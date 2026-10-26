"""One sample of our converted demos through UVA's own loader: shapes, ranges, and the frames.

    envs/uva/.venv/bin/python experiments/uva/scripts/check_data.py \
        --data data/uva/pick_place_cube --out experiments/uva/data_check.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[3] / "envs/uva/src"
sys.path.insert(0, str(SRC))

from unified_video_action.dataset.libero_replay_image_dataset import LiberoReplayImageDataset  # noqa: E402

SHAPE_META = {
    "image_resolution": 128,
    "action": {"shape": [8]},
    "obs": {
        "agentview_rgb": {"shape": [3, 128, 128], "type": "rgb"},
        "joint_qpos": {"shape": [8], "type": "low_dim"},
        "language": {"shape": [15]},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    ds = LiberoReplayImageDataset(
        shape_meta=SHAPE_META, dataset_path=str(args.data.resolve()), horizon=32, n_obs_steps=16,
        pad_before=1, pad_after=7, abs_action=False, language_emb_model="clip", data_aug=False,
    )
    sample = ds[len(ds) // 2]
    print(f"{len(ds)} windows")
    for key, value in [*sample["obs"].items(), ("action", sample["action"])]:
        print(f"  {key:<14} {tuple(value.shape)}  {value.dtype}  [{value.min():.3f}, {value.max():.3f}]")

    frames = sample["obs"]["agentview_rgb"]  # [T, 3, H, W]
    shown = range(0, frames.shape[0], 4)
    fig, axes = plt.subplots(1, len(shown), figsize=(3 * len(shown), 3))
    for ax, t in zip(axes, shown):
        ax.imshow(np.asarray(frames[t]).transpose(1, 2, 0))
        ax.set_title(f"t={t}")
        ax.axis("off")
    fig.suptitle("top camera as UVA's loader returns it (should be upright)")
    fig.savefig(args.out, dpi=80, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
