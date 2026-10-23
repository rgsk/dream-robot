"""VPP2 stage B: fine-tune the LIBERO Action expert on T5, the 14B video model frozen (their recipe).

Run inside the VPP2 env, from its repo root (envs/vpp2/src), PYTHONPATH=src:

    python $EXP/train_t5.py text-cache                 # UMT5 contexts for our 6 prompts (once)
    python $EXP/train_t5.py init                       # Action-30k -> 8-D action layers (once)
    python $EXP/train_t5.py train --steps 2000 --batch 8 [--hf-repo rgsk/dream-robot-checkpoints]

Their CLI (vpp2/libero/cli.py) pins the LIBERO paper setup (7-D actions, the four LIBERO suites,
global batch 256), so this calls their pieces directly with ours swapped in:
- data: data/vpp/t5_lerobot21 (to_lerobot21.py), 8-D state and action, no delta dims (ours are
  absolute joint targets: padded steps repeat the last target); min/max stats computed from our
  data by their dataset on first use and saved in the run dir (the eval reads them from there);
- init: their Action-30k, *every* trained tensor kept -- the 40-layer backbone, the layers that
  read the video model's K/V, the proprio encoder (8-D, same width) -- except the 7-D action
  input / output layers, which become fresh 8-D ones. (Their init path would drop the K/V
  readers too, and stage A showed the reading works.)
- train: their LiberoTrainer, action-only, video frozen, one GPU, no DeepSpeed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / "data/vpp/t5_lerobot21"
WEIGHTS = Path("weights").resolve()
INIT = WEIGHTS / "t5/action_init_8d.pt"
CACHE = REPO / "data/vpp/text_cache"


def config(out: Path | None = None, steps: int = 2000, batch: int = 8) -> OmegaConf:
    cfg = OmegaConf.load("configs/libero/train_action.yaml")
    d, proc, m = cfg.data.train, cfg.data.train.processor, cfg.model
    d.dataset_dirs = [str(DATA)]
    d.pretrained_norm_stats = None  # computed from our data, saved as <out>/dataset_stats.json
    d.text_embedding_cache_dir = str(CACHE)
    for meta in (d.shape_meta, proc.shape_meta):
        meta.action[0].raw_shape = meta.action[0].shape = 8
        meta.state[0].raw_shape = meta.state[0].shape = 8
        for image in meta.images:
            image.raw_shape = [3, 224, 224]
    proc.action_output_dim, proc.proprio_output_dim = 8, 8
    proc.delta_action_dim_mask = None
    m.model_id = str(WEIGHTS / "Wan2.1-I2V-14B-480P")
    m.tokenizer_model_id = str(WEIGHTS / "Wan2.1-I2V-14B-480P/google/umt5-xxl")
    m.dit_checkpoint_path = str(WEIGHTS / "libero/video_step010000.pt")
    m.action_dit_config.action_dim = 8
    m.action_dit_pretrained_path = None  # as their eval: parameters come from resume_ckpt
    m.skip_dit_load_from_pretrain = True
    cfg.resume_ckpt = str(INIT)
    cfg.frozen_video_checkpoint = m.dit_checkpoint_path
    cfg.output_dir = str(out) if out else "runs/t5/action"
    cfg.max_steps, cfg.batch_size = steps, batch
    cfg.warmup_steps, cfg.save_every, cfg.eval_every, cfg.log_every = 100, 500, 500, 10
    cfg.num_workers = 8
    cfg.wandb.enabled = False
    return cfg


def text_cache() -> None:
    import vpp2.libero.data as data
    data.check_data = lambda cfg: None  # their check counts the LIBERO suites' frames
    print(data.text_cache(config(), "cuda", 8))


def init() -> None:
    """Action-30k with fresh 8-D action layers; everything else as trained on LIBERO."""
    from vpp2.models.wan21_14b.action_dit import ActionDiT
    src = WEIGHTS / "libero/action_step030000.pt"
    payload = torch.load(src, map_location="cpu", weights_only=True)
    cfg = config()
    fresh = ActionDiT(**OmegaConf.to_container(cfg.model.action_dit_config, resolve=True)).state_dict()
    state, replaced = dict(payload["action_expert"]), []
    for key, value in state.items():
        if value.shape != fresh[key].shape:
            state[key] = fresh[key].to(value.dtype)
            replaced.append(f"{key} {tuple(value.shape)} -> {tuple(fresh[key].shape)}")
    assert set(state) == set(fresh), "key sets differ"
    assert all(k.startswith(("action_encoder.", "head.")) for k in (r.split()[0] for r in replaced)), replaced
    payload["action_expert"] = state
    payload["resolved_config"]["model"]["action_dit_config"]["action_dim"] = 8
    payload["video_checkpoint"]["path"] = str(WEIGHTS / "libero/video_step010000.pt")
    payload["step"] = 0
    INIT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, INIT)
    print(f"{INIT}: kept {len(state) - len(replaced)} tensors, fresh 8-D: {replaced}")


def train(args) -> None:
    """Their training, saving only the compact action checkpoint (1.8 GB) -- their full resume
    state is 38 GB a save (frozen video + optimizer) -- and pushing each one to the hub."""
    from vpp2.libero.runtime import run_training
    from vpp2.trainer import JointTrainer

    api, uploads = None, []
    if args.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi()

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"
        path = self._save_weights_checkpoint(step_tag=step_tag, state_dict_overrides=None)
        if api is not None:  # pods are deleted; the hub keeps it
            for local in (Path(path), Path(self.output_dir) / "dataset_stats.json"):
                uploads.append(api.upload_file(path_or_fileobj=str(local), repo_id=args.hf_repo,
                                               path_in_repo=f"vpp2/{args.out.name}/{local.name}",
                                               run_as_future=True))
        return {"weights_path": path, "state_path": None}

    JointTrainer.save_checkpoint = save_checkpoint
    cfg = config(args.out, args.steps, args.batch)
    if args.init:  # continue from one of our checkpoints (weights only: fresh optimizer + warm-up)
        cfg.resume_ckpt = str(Path(args.init).resolve())
        cfg.resume_ckpt_load_video = False  # the frozen video comes from frozen_video_checkpoint
    run_training(cfg)
    for upload in uploads:  # don't exit before the last checkpoint reaches the hub
        upload.result()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("text-cache")
    sub.add_parser("init")
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, default=Path("runs/t5/action"))
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--init", default=None, help="start from this compact checkpoint instead of Action-30k")
    t.add_argument("--hf-repo", default=None, help="push each checkpoint here, e.g. rgsk/dream-robot-checkpoints")
    c = sub.add_parser("show-config")
    c.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.cmd == "text-cache":
        text_cache()
    elif args.cmd == "init":
        init()
    elif args.cmd == "train":
        train(args)
    else:
        print(OmegaConf.to_yaml(config(args.out)))


if __name__ == "__main__":
    sys.exit(main())
