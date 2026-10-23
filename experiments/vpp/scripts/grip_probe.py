"""Does a VPP2 checkpoint close the gripper where the demos do? Recorded frames in, gripper out.

The step-500 eval reached the named cube, opened around it and never closed (0/10 NO_GRASP). On
recorded demos, around the moment the demo's gripper action flips open -> closed, this asks the
model for its chunk (exactly as serve.py does) and prints its gripper channel next to the demo's,
plus the joint error over the chunk. ~0.5 where the demo says 0 -> undertrained; a clean wrong
value -> a convention bug.

    cd envs/vpp2/src && PYTHONPATH=src ../.venv/bin/python $EXP/grip_probe.py \
        --ckpt runs/t5/b/checkpoints/weights/step_000500.pt --stats runs/t5/b/dataset_stats.json
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import call, normalize_proprio  # noqa: E402
from train_t5 import config  # noqa: E402

from vpp2.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402

DATA = Path("/workspace/dream-robot/data/cosmos_policy/t5_224")
OFFSETS = (-24, -12, -4, 0, 8)  # steps relative to the demo's close


def jpeg(group, key, t):
    return np.array(Image.open(io.BytesIO(group[key][t].tobytes())).convert("RGB"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--demos", type=int, default=4)
    args = p.parse_args()

    cfg = config()
    cfg.model.load_text_encoder = True
    cfg.model.tokenizer_max_len = 128
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(args.ckpt, video_checkpoint_override=cfg.model.dit_checkpoint_path)
    model = model.to("cuda").eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(args.stats))
    normalizer = processor.normalizer.normalizers["action"]["default"]

    for path in sorted(DATA.glob("*_demo.hdf5"))[: args.demos]:
        prompt = path.name[: -len("_demo.hdf5")].replace("_", " ")
        with h5py.File(path) as f:
            g = f["data/demo_0"]
            actions, states = g["actions"][:], g["robot_states"][:]
            close = int(np.flatnonzero(actions[:, 7] < 0.5)[0])  # first closed command
            print(f"\n{prompt}: demo closes at step {close} (gripper action before {actions[close - 1, 7]:.2f}, "
                  f"after {actions[close, 7]:.2f})", flush=True)
            for off in OFFSETS:
                t = close + off
                top, wrist = jpeg(g, "obs/agentview_rgb_jpeg", t), jpeg(g, "obs/eye_in_hand_rgb_jpeg", t)
                proprio = normalize_proprio(states[t].astype(np.float32), processor)
                out, _ = call(model, cfg, processor, top, wrist, prompt, proprio, False)
                pred = normalizer.backward(out["action"].reshape(1, 32, -1).float())[0].numpy()
                truth = actions[t: t + 32]
                n = len(truth)
                joint_err = np.abs(pred[:n, :7] - truth[:, :7]).mean()
                print(f"  t={t:3d} ({off:+3d})  gripper model {np.round(pred[:16:3, 7], 2)}  "
                      f"demo {np.round(truth[:16:3, 7], 2)}  joint err {joint_err:.3f} rad", flush=True)


if __name__ == "__main__":
    main()
