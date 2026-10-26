"""Fine-tune UVA (from the LIBERO-10 checkpoint) on one of our tasks, with the robot's joints.

    envs/uva/.venv/bin/python experiments/uva/scripts/train_uva.py \
        --data data/uva/pick_place_cube --name pick_place_cube --out experiments/uva/runs/b1 \
        --steps 5000 --batch 32 [--hf-repo rgsk/dream-robot-checkpoints]
    # T5, both cameras: --data data/uva/t5 --name t5 --image 224 --wrist [--imagine-wrist]

What differs from their workspace (train_unified_video_action_workspace.py), and why:
- **Joints (+ wrist camera with --wrist) in, 8-D actions out, actions scaled to [-1, 1]**:
  uva_patch.py, task_cfg.py.
- **Every training mode, as theirs**: each batch draws one of policy / video / forward / inverse /
  full dynamics at random (``task_modes: []``) -- video is the teacher even for the policy.
- One GPU, plain loop, bf16 autocast (their fp16 needs a grad scaler), cosine LR with warmup,
  their EMA; no accelerate, no wandb, no LIBERO rollouts. Losses and a validation action error
  to stdout and <out>/log.jsonl.
- **val_action_err**: the EMA policy, as at run time (16 past steps -> 16 actions), on held-out
  demos, each with its own instruction; mean |predicted - recorded| over the chunk, in radians.
- Checkpoint = EMA weights + normalizer + cfg (<out>/model.pt); with --hf-repo each one is also
  pushed (pods are deleted; the hub keeps it).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from huggingface_hub import HfApi
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[3] / "envs/uva/src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_cfg import build_cfg  # noqa: E402
from uva_patch import (  # noqa: E402
    WRIST_KEY, add_inputs, decode_into_ram, one_thread, scale_actions, skip_threadpool_limits)
from unified_video_action.model.common.lr_scheduler import get_scheduler  # noqa: E402
from unified_video_action.utils.data_utils import resize_image  # noqa: E402

def to_cuda(batch):
    return {"obs": {k: v.cuda(non_blocking=True) for k, v in batch["obs"].items()},
            "action": batch["action"].cuda(non_blocking=True)}


@torch.no_grad()
def val_action_error(policy, loader, batches: int) -> float:
    """Run-time call on held-out windows: first 16 steps in, 16 actions out vs the recorded ones."""
    policy.eval()
    errors = []
    for i, batch in enumerate(loader):
        if i == batches:
            break
        batch = to_cuda(batch)
        obs = {"agentview_image": batch["obs"]["agentview_rgb"][:, :16].contiguous(),  # their resize uses .view
               **{k: v[:, :16].contiguous() for k, v in batch["obs"].items() if k in ("joint_qpos", WRIST_KEY)}}
        ids = batch["obs"]["language"][:, 0, 0].long()  # [B, 30] CLIP token ids (+ mask in [:, 0, 1])
        prompts = policy.tokenizer.batch_decode(ids, skip_special_tokens=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = policy.predict_action(obs, language_goal=prompts)["action_pred"]
        recorded = batch["action"][:, 15:31]  # their shift_action: the chunk starts at the last past step
        errors.append((pred.float() - recorded).abs().mean().item())
    return float(np.mean(errors))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=Path("envs/uva/ckpt/libero10.ckpt"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--name", required=True, help="task name (pick_place_cube, t5)")
    p.add_argument("--image", type=int, default=128, help="frame size the data was recorded at")
    p.add_argument("--wrist", action="store_true", help="also feed the wrist camera")
    p.add_argument("--imagine-wrist", action="store_true", help="also predict the future wrist frames")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--hf-repo", default=None, help="push each checkpoint + log, e.g. rgsk/dream-robot-checkpoints")
    args = p.parse_args()
    ckpt, data, out = args.ckpt.resolve(), args.data.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(SRC)  # their config points at pretrained_models/... relative to the repo

    cfg = build_cfg(ckpt, data, args.name, out, image=args.image, wrist=args.wrist,
                    imagine_wrist=args.imagine_wrist)
    policy = hydra.utils.instantiate(
        cfg.model.policy, task_name=cfg.task.name, task_modes=cfg.task.task_modes,
        normalizer_type=cfg.task.dataset.normalizer_type,
        language_emb_model=cfg.task.dataset.language_emb_model)
    train_set = hydra.utils.instantiate(cfg.task.dataset)
    val_set = train_set.get_validation_dataset()
    t0 = time.time()
    for dataset in (train_set, val_set):
        decode_into_ram(dataset)
    print(f"frames decoded into RAM in {time.time() - t0:.0f} s", flush=True)
    normalizer = train_set.get_normalizer()
    scale_actions(normalizer, train_set.replay_buffer["action"])
    policy.set_normalizer(normalizer)
    ema_policy = copy.deepcopy(policy)  # before patching: the patch binds methods to its policy
    for model in (policy, ema_policy):
        add_inputs(model, wrist=args.wrist, imagine_wrist=args.imagine_wrist)
        model.cuda()
    ema = hydra.utils.instantiate(cfg.ema, model=ema_policy)

    skip_threadpool_limits()
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True, worker_init_fn=one_thread)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=True, num_workers=2,
                            worker_init_fn=one_thread)
    optimizer = policy.get_optimizer(**{**cfg.model.policy.optimizer, "learning_rate": args.lr})
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=args.warmup,
                              num_training_steps=args.steps)
    print(f"train {len(train_set)} windows, val {len(val_set)}; batch {args.batch}, {args.steps} steps",
          flush=True)

    api, uploads = (HfApi() if args.hf_repo else None), []
    log = (out / "log.jsonl").open("a")
    step, start, sums = 0, time.time(), {}
    while step < args.steps:
        waited = time.time()
        for batch in loader:
            sums.setdefault("data_s", []).append(time.time() - waited)  # time spent waiting on the loader
            policy.train()
            batch = resize_image(cfg, to_cuda(batch))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, (video_loss, action_loss) = policy(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.step(policy)
            step += 1
            sums.setdefault("loss", []).append(float(loss))
            if float(video_loss) > 0:
                sums.setdefault("video_loss", []).append(float(video_loss))
            if float(action_loss) > 0:  # only modes that predict actions
                sums.setdefault("action_loss", []).append(float(action_loss))

            if step % args.log_every == 0:
                row = {"step": step, "s_per_step": round((time.time() - start) / step, 3),
                       **{k: round(float(np.mean(v)), 4) for k, v in sums.items()},
                       "lr": scheduler.get_last_lr()[0], "gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
                sums = {}
                if step % args.val_every == 0:
                    row["val_action_err"] = round(val_action_error(ema_policy, val_loader, 4), 4)
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"ema": ema_policy.state_dict(), "normalizer": normalizer.state_dict(),
                            "cfg": cfg, "step": step}, out / "model.pt")
                if args.hf_repo:
                    for name in ("model.pt", "log.jsonl"):
                        uploads.append(api.upload_file(
                            path_or_fileobj=str(out / name), repo_id=args.hf_repo,
                            path_in_repo=f"uva/{out.name}/{name}", run_as_future=True))
            if step == args.steps:
                break
            waited = time.time()
    for upload in uploads:  # don't exit before the last checkpoint reaches the hub
        upload.result()


if __name__ == "__main__":
    main()
