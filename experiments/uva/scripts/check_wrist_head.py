"""c2's wrist video head, checked on a trained c1 checkpoint before a pod run.

The c1 policy, with a wrist head added as training would (a copy of the top-camera video head):
1. one run-time call + imagine: a top and a wrist frame come back (saved side by side);
2. one T5 training window (batch 1) per mode: the wrist loss is added in the video modes only, and
   gradients reach the wrist head.

    envs/uva/.venv/bin/python experiments/uva/scripts/check_wrist_head.py \
        --ckpt envs/uva/runs/uva/t5_c1/model.pt --data data/uva/t5 --out scratch/wrist_head.png
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import open_dict
from PIL import Image

SRC = Path(__file__).resolve().parents[3] / "envs/uva/src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from serve import imagine, record_policy_calls  # noqa: E402
from uva_patch import WRIST_KEY, add_inputs, skip_threadpool_limits  # noqa: E402
from unified_video_action.model.common.normalizer import LinearNormalizer  # noqa: E402
from unified_video_action.utils.data_utils import resize_image  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    ckpt_path, data, out = args.ckpt.resolve(), args.data.resolve(), args.out.resolve()
    os.chdir(SRC)

    ckpt = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = ckpt["cfg"]
    with open_dict(cfg):
        cfg.model.policy.autoregressive_model_params.pretrained_model_path = None
        cfg.task.dataset.dataset_path = str(data)
    policy = hydra.utils.instantiate(
        cfg.model.policy, task_name=cfg.task.name, task_modes=cfg.task.task_modes,
        normalizer_type=cfg.task.dataset.normalizer_type,
        language_emb_model=cfg.task.dataset.language_emb_model)
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    policy.set_normalizer(normalizer)
    add_inputs(policy, wrist=True, imagine_wrist=True)
    missing, unexpected = policy.load_state_dict(ckpt["ema"], strict=False)
    policy.model.diffloss_wrist.load_state_dict(policy.model.diffloss.state_dict())  # as at training start
    print(f"not in c1: {sorted({k.split('.')[1] for k in missing})}, unexpected: {unexpected}")
    policy.cuda()

    skip_threadpool_limits()
    ds = hydra.utils.instantiate(cfg.task.dataset)
    sample = ds[len(ds) // 3]
    obs = {k: v[None].cuda() for k, v in sample["obs"].items()}
    prompt = policy.tokenizer.decode(obs["language"][0, 0, 0].long(), skip_special_tokens=True)

    # 1. run time + imagine
    policy.eval()
    last = record_policy_calls(policy)
    run_obs = {"agentview_image": obs["agentview_rgb"][:, :16].contiguous(),
               "joint_qpos": obs["joint_qpos"][:, :16].contiguous(), WRIST_KEY: obs[WRIST_KEY][:, :16].contiguous()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        policy.predict_action(run_obs, language_goal=[prompt])
        frames = imagine(policy, last, (224, 224))
    print(f"imagined: {sorted(frames)}  ({prompt!r})")
    pics = [np.frombuffer(frames[k], np.uint8).reshape(224, 224, 3) for k in ("future_top", "future_wrist")]
    Image.fromarray(np.concatenate(pics, axis=1)).save(out)
    print(f"-> {out}")

    # 2. one training window per mode
    policy.train()
    for mode in ("policy_model", "video_model", "dynamic_model", "full_dynamic_model"):
        policy.task_modes = [mode]
        batch = resize_image(cfg, {"obs": {k: v.clone() for k, v in obs.items()}, "action": sample["action"][None].cuda()})
        policy.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, (video_loss, action_loss) = policy(batch)
        loss.backward()
        grad = policy.model.diffloss_wrist.net.input_proj.weight.grad
        print(f"{mode:<19} loss {float(loss):.3f}  video {float(video_loss):.3f}  action {float(action_loss):.3f}  "
              f"wrist head grad {0.0 if grad is None else float(grad.abs().sum()):.2e}")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
