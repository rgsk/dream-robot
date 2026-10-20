"""Decode a dataset's frames once into <root>/decoded/<camera>.npy, and spot-check them.

    uv run python experiments/idm/scripts/build_cache.py
"""
import time

import numpy as np
import torch

from dream_robot.core.dataset import image_feature, open_dataset
from dream_robot.policies.idm.data import decoded_frames

ROOT = "data/robosuite/instructed_pick_place_clean_n150"
REPO = "dream_robot/instructed_pick_place_clean_n150"

if __name__ == "__main__":
    t = time.time()
    frames = {c: np.load(p, mmap_mode="r") for c, p in decoded_frames(REPO, ROOT, ("top", "wrist")).items()}
    print(f"cache ready in {time.time() - t:.0f} s:", {k: v.shape for k, v in frames.items()})
    ds = open_dataset(REPO, ROOT)
    for row in (0, 12345, len(ds) - 1):
        item = ds[row]
        for cam in ("top", "wrist"):
            ref = (item[image_feature(cam)] * 255).round().to(torch.uint8).numpy()
            print(f"row {row} {cam}: matches LeRobot's decode = {np.array_equal(ref, frames[cam][row])}")
