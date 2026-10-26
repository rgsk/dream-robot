"""Stage B, piece 2: the patched UVA builds from LIBERO-10, and trains on our data.

Checks, on pick_place_cube:
1. which weights did NOT come from the checkpoint (should be only the 8-D action layers);
2. scaled actions / joints sit in [-1, 1];
3. at the start, the joint input changes nothing (its layer is zero): same seed, two different
   joint readings -> identical actions;
4. one training step (fp16 autocast, AdamW) per batch size: losses, seconds, peak GPU memory.

    envs/uva/.venv/bin/python experiments/uva/scripts/check_model.py \
        --ckpt envs/uva/ckpt/libero10.ckpt --data data/uva/pick_place_cube
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import hydra
import torch
from torch.utils.data import DataLoader

SRC = Path(__file__).resolve().parents[3] / "envs/uva/src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_cfg import build_cfg  # noqa: E402
from uva_patch import add_inputs, scale_actions  # noqa: E402
from unified_video_action.utils.data_utils import resize_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()
    ckpt, data = args.ckpt.resolve(), args.data.resolve()
    os.chdir(SRC)  # their config points at pretrained_models/... relative to the repo

    cfg = build_cfg(ckpt, data, "pick_place_cube", Path("/tmp/uva_check_model"))
    policy = hydra.utils.instantiate(
        cfg.model.policy, task_name=cfg.task.name, task_modes=cfg.task.task_modes,
        normalizer_type=cfg.task.dataset.normalizer_type,
        language_emb_model=cfg.task.dataset.language_emb_model)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    scale_actions(normalizer, dataset.replay_buffer["action"])
    policy.set_normalizer(normalizer)
    add_inputs(policy)
    policy.cuda()

    # 2. ranges after scaling
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True)))
    for key in ("action", "joint_qpos"):
        raw = batch["action"] if key == "action" else batch["obs"][key]
        scaled = normalizer[key].normalize(raw)
        print(f"{key:<10} raw [{raw.min():.2f}, {raw.max():.2f}] -> scaled [{scaled.min():.2f}, {scaled.max():.2f}]")

    # 3. the zero joint layer: two different readings, same seed -> same actions
    policy.eval()
    frames = batch["obs"]["agentview_rgb"][:1, :16].cuda()
    joints = batch["obs"]["joint_qpos"][:1, :16].cuda()
    goal = ["put the cube in the bin"]
    outs = []
    for j in (joints, joints + 0.5):
        torch.manual_seed(0)
        with torch.no_grad():
            outs.append(policy.predict_action(
                {"agentview_image": frames.clone(), "joint_qpos": j}, language_goal=goal)["action_pred"])
    print(f"zero joint layer: actions differ by {(outs[0] - outs[1]).abs().max():.2e} "
          f"(shape {tuple(outs[0].shape)})")

    # 4. one training step per batch size
    policy.train()
    optimizer = policy.get_optimizer(**cfg.model.policy.optimizer)
    for size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = next(iter(DataLoader(dataset, batch_size=size, shuffle=True)))
        batch = resize_image(cfg, {"obs": {k: v.cuda() for k, v in batch["obs"].items()},
                                   "action": batch["action"].cuda()})
        try:
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                loss, (video_loss, action_loss) = policy(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            print(f"batch {size}: loss {float(loss):.3f} (video {float(video_loss):.3f}, "
                  f"action {float(action_loss):.3f}), {time.perf_counter() - t0:.2f} s, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        except torch.cuda.OutOfMemoryError:
            print(f"batch {size}: out of memory")
            break


if __name__ == "__main__":
    main()
