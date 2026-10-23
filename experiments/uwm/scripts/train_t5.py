"""Fine-tune UWM (from the LIBERO-90 checkpoint) on T5, with the instruction; optionally video-drop.

    PYTHONPATH=envs/uwm/src:experiments/uwm/scripts envs/uwm/.venv/bin/python \
        experiments/uwm/scripts/train_t5.py --out experiments/uwm/runs/b1 --steps 20000
    # B2: --video-drop 0.5 --registers 64

What differs from their train_robomimic.py, and why:
- **Instruction**: CLIP text vector on (their LIBERO-90 model ran without one), from
  t5_dataset.py. It widens the condition every block reads (3,584 -> 4,352), so each block's
  AdaLN input layer is loaded with the pretrained columns and **zeros for the new ones**: at step 0
  the model is exactly the pretrained one, and the instruction's influence is learnt from 0.
- **Actions**: 8-D joints + gripper instead of 7-D deltas -> the action input / output layers start
  fresh. Scaled per dimension to [-1, 1] (min / max over the data): their sampler clips every
  sample to [-1, 1] (clip_sample) and their LIBERO actions were already in range; ours (joints up
  to 3.4 rad) would be clipped silently. Stats saved in the checkpoint.
- **Video-drop** (experiments/uwm/notes.md step 2): with probability --video-drop a training step
  removes the video tokens from the sequence and trains actions only; at run time the same model
  can then act with no video tokens. --registers adds scratch tokens (24 -> N; the pretrained 24
  load, the rest start fresh).
- One GPU, plain loop, bf16 autocast, no wandb: losses and validation action error to stdout and
  <out>/log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from t5_dataset import make_t5_dataset

SRC = Path(__file__).resolve().parents[3] / "envs/uwm/src"
REPO = Path(__file__).resolve().parents[3]
SHAPE_META = {
    "obs": {k: {"shape": [224, 224, 3], "type": "rgb"} for k in ("agentview_rgb", "eye_in_hand_rgb")},
    "action": {"shape": [8]},
}


def build_model(registers: int, patches: bool = False, joints: bool = False):
    with initialize_config_dir(config_dir=str(SRC / "configs"), version_base=None):
        cfg = compose(config_name="train_uwm_robomimic")
    meta = SHAPE_META if not joints else {  # + 7 joints + gripper, both frames, into AdaLN
        **SHAPE_META, "obs": {**SHAPE_META["obs"], "robot_states": {"shape": [8], "type": "low_dim"}}}
    cfg.dataset.shape_meta = OmegaConf.create(meta)
    cfg.model.obs_encoder.use_language = True
    cfg.model.obs_encoder.use_low_dim = joints  # their config: off
    cfg.model.num_registers = registers
    model = instantiate(cfg.model)
    if patches:  # image patches + instruction words as cross-attention context (uwm_patch.py)
        from uwm_patch import add_patch_context
        add_patch_context(model)
    return model, cfg


def load_pretrained(model, path: Path) -> dict:
    """Pretrained weights into a model whose condition, action width or registers grew.

    same shape -> copied; condition grew (AdaLN input layers) -> old columns copied, new ones 0;
    registers / position table grew -> the pretrained prefix copied, the rest keeps its init;
    anything else that differs (7-D -> 8-D action layers) -> left at its fresh init.
    """
    pretrained = torch.load(path, map_location="cpu")["model"]
    own = model.state_dict()
    report = {"copied": 0, "cond_widened": 0, "prefix": 0, "fresh": [], "unused": []}
    for key, value in pretrained.items():
        if key not in own:
            report["unused"].append(key)
            continue
        target = own[key]
        if target.shape == value.shape:
            target.copy_(value)
            report["copied"] += 1
        elif "adaLN_modulation" in key and key.endswith("weight") and target.shape[0] == value.shape[0]:
            old = value.shape[1] - 512           # condition = [image (+ joints, text) features, time 512]
            target.zero_()
            target[:, :old] = value[:, :old]
            target[:, -512:] = value[:, -512:]
            report["cond_widened"] += 1
        elif key.endswith(("registers", "pos_embed")) and target.shape[2] == value.shape[2]:
            target[:, : value.shape[1]] = value
            report["prefix"] += 1
        else:
            report["fresh"].append(key)
    model.load_state_dict(own)
    return report


def forward_dropped(self, global_cond, action, action_t, next_obs, next_obs_t):
    """DualNoisePredictionNet.forward without the video tokens: actions + registers only."""
    if action_t.dim() == 0:
        action_t = action_t.expand(action.shape[0]).to(dtype=torch.long, device=action.device)
    if next_obs_t.dim() == 0:
        next_obs_t = next_obs_t.expand(action.shape[0]).to(dtype=torch.long, device=action.device)
    a0, a1 = self.action_inds
    _, o1 = self.next_obs_inds
    registers = self.registers.expand(action.shape[0], -1, -1)
    x = torch.cat((self.action_encoder(action), registers), dim=1)
    x = x + torch.cat((self.pos_embed[:, a0:a1], self.pos_embed[:, o1:]), dim=1)
    cond = torch.cat((global_cond, self.timestep_embedding(action_t, next_obs_t)), dim=-1)
    for block in self.blocks:
        x = block(x, cond)
    x = self.head(x, cond)
    return self.action_decoder(x[:, a0:a1]), None


def dropped(model):
    """A context to run model.noise_pred_net with the video tokens removed."""
    net = model.noise_pred_net

    class _Ctx:
        def __enter__(self):
            if hasattr(net, "drop_video"):   # patch model: its forward has a switch
                net.drop_video = True
                return
            self.original = net.forward
            net.forward = types.MethodType(forward_dropped, net)

        def __exit__(self, *exc):
            if hasattr(net, "drop_video"):
                net.drop_video = False
                return
            net.forward = self.original

    return _Ctx()


def split(batch, device):
    """Their process_batch: 2 current frames, 16 actions, 2 frames after the actions."""
    obs, actions = batch["obs"], batch["action"]
    curr = {k: v[:, :2].to(device) for k, v in obs.items()}
    nxt = {k: v[:, 17:].to(device) for k, v in obs.items()}
    curr["input_ids"] = batch["input_ids"].to(device)
    curr["attention_mask"] = batch["attention_mask"].to(device)
    return curr, nxt, actions[:, 1:17].to(device)


def action_only_loss(model, curr, action):
    """The action half of model.forward, with no video tokens and no VAE pass."""
    obs = model.obs_encoder.encode_curr_obs(curr)
    noise = torch.randn_like(action)
    t = torch.randint(0, model.num_train_steps, (action.shape[0],), device=action.device)
    noisy = model.noise_scheduler.add_noise(action, noise, t)
    last = torch.full_like(t, model.num_train_steps - 1)
    with dropped(model):
        pred, _ = model.noise_pred_net(obs, noisy, t, None, last)
    return F.mse_loss(pred, noise)


@torch.no_grad()
def val_action_error(model, loader, device, scale, offset, drop: bool, batches: int = 4) -> float:
    """Mean |sampled - true| action per dimension, in data units (policy mode, 10 steps)."""
    model.eval()
    errors = []
    for i, batch in enumerate(loader):
        if i == batches:
            break
        curr, _, action = split(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if drop:
                with dropped(model):
                    pred = model.sample_marginal_action(curr)
            else:
                pred = model.sample_marginal_action(curr)
        errors.append(((pred.float() - (action - offset) / scale).abs() * scale).mean().item())
    model.train()
    return float(np.mean(errors))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=REPO / "envs/uwm/ckpt/libero_90.pt")
    p.add_argument("--data", default=str(REPO / "data/cosmos_policy/t5_224/*.hdf5"))
    p.add_argument("--buffer", default=str(REPO / "data/uwm/t5_224/buffer.zarr"))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--video-drop", type=float, default=0.0, help="share of steps without video tokens")
    p.add_argument("--registers", type=int, default=24)
    p.add_argument("--patches", action="store_true",
                   help="image patches + instruction words as cross-attention context (uwm_patch.py)")
    p.add_argument("--joints", action="store_true", help="the robot's joints + gripper as input")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--vae-microbatch", type=int, default=16, help="frames per frozen-VAE pass")
    p.add_argument("--hf-repo", default=None,
                   help="push each checkpoint + log here (e.g. rgsk/dream-robot-checkpoints, private)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(0)

    train_set, val_set = make_t5_dataset("t5", args.data, args.buffer, SHAPE_META, seq_len=19,
                                         val_ratio=0.05)
    actions = train_set.buffer["action"][:]
    lo, hi = actions.min(0), actions.max(0)
    scale = torch.tensor((hi - lo) / 2, device=device).clamp(min=1e-6)
    offset = torch.tensor((hi + lo) / 2, device=device)
    loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, drop_last=True,
                                         num_workers=args.workers, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch, shuffle=True,
                                             num_workers=2)

    model, cfg = build_model(args.registers, args.patches, args.joints)
    report = load_pretrained(model, args.ckpt)
    print(f"loaded: copied {report['copied']}, condition widened {report['cond_widened']}, "
          f"prefix {report['prefix']}, fresh {report['fresh']}, unused {report['unused']}", flush=True)
    model = model.to(device).train()
    # Their VAE pass encodes up to 72 frames at once ("tuned for 40GB"); the 4060 has 8.
    vae_pass = model.obs_encoder.apply_vae
    model.obs_encoder.apply_vae = lambda imgs, inverse=False: vae_pass(
        imgs, inverse=inverse, microbatch_size=args.vae_microbatch)
    encoder =[p for n, p in model.named_parameters() if p.requires_grad and n.startswith("obs_encoder.img_encoder")]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("obs_encoder.img_encoder")]
    optimizer = torch.optim.AdamW([{"params": encoder, "lr": args.lr * 0.05}, {"params": rest, "lr": args.lr}],
                                  weight_decay=1e-6,  # their fine-tune: encoder at 0.05x
                                  foreach=False)  # one tensor at a time: no extra 1.2 GB copy on the 4060

    api, uploads = (HfApi() if args.hf_repo else None), []
    log = (args.out / "log.jsonl").open("a")
    rng = np.random.default_rng(0)
    step, start, sums = 0, time.time(), {}
    while step < args.steps:
        for batch in loader:
            batch["action"] = (batch["action"].to(device) - offset) / scale
            curr, nxt, action = split(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if rng.random() < args.video_drop:
                    loss = action_only_loss(model, curr, action)
                    info = {"drop_action_loss": loss.item()}
                else:
                    loss, info = model(curr, nxt, action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            for k, v in info.items():
                sums.setdefault(k, []).append(v)

            if step % args.log_every == 0:
                row = {"step": step, "s_per_step": round((time.time() - start) / step, 3),
                       **{k: round(float(np.mean(v)), 4) for k, v in sums.items()},
                       "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
                sums = {}
                if step % args.val_every == 0:
                    row["val_action_err"] = round(val_action_error(model, val_loader, device, scale, offset, False), 4)
                    if args.video_drop:
                        row["val_action_err_dropped"] = round(
                            val_action_error(model, val_loader, device, scale, offset, True), 4)
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"model": model.state_dict(), "step": step, "registers": args.registers,
                            "video_drop": args.video_drop, "patches": args.patches, "action_scale": scale.cpu(),
                            "action_offset": offset.cpu(), "joints": args.joints,
                            "state_scale": torch.tensor(train_set.state_scale),
                            "state_offset": torch.tensor(train_set.state_offset)}, args.out / "model.pt")
                if args.hf_repo:  # pods are deleted; the hub keeps it (uploads in the background)
                    for name in ("model.pt", "log.jsonl"):
                        uploads.append(api.upload_file(
                            path_or_fileobj=str(args.out / name), repo_id=args.hf_repo,
                            path_in_repo=f"uwm/{args.out.name}/{name}", run_as_future=True))
            if step == args.steps:
                break
    for upload in uploads:  # don't exit before the last checkpoint reaches the hub
        upload.result()


if __name__ == "__main__":
    main()
