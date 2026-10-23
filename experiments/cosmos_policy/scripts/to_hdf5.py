"""T5 demos (our LeRobot dataset) -> Cosmos Policy's LIBERO-style HDF5 files.

    uv run --with h5py python experiments/cosmos_policy/scripts/to_hdf5.py \
        --root data/robosuite/instructed_pick_place_clean_n150_224 --out data/cosmos_policy/t5_224

What their LIBERODataset reads (cosmos_policy/datasets/libero_dataset.py), and what goes in it:
- one file per instruction, ``<words_joined_by_underscores>_demo.hdf5``: the loader takes the
  instruction from the file name (it strips the last 10 characters, ``_demo.hdf5``);
- ``data/demo_<i>/obs/agentview_rgb_jpeg`` / ``eye_in_hand_rgb_jpeg``: our top / wrist frames,
  JPEG quality 95 -- their eval JPEG-compresses live frames at 95 to match;
- ``data/demo_<i>/actions`` (T, 8): our absolute joint targets + gripper, as recorded;
- ``data/demo_<i>/robot_states`` (T, 8): our ``observation.state``.
Frames are upright already (our env flips robosuite's renders), so evals run without --flip_images.
A ``manifest.json`` maps every (file, demo) back to our episode index and seed.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from dream_robot.core.dataset import ACTION_FEATURE, STATE_FEATURE, open_dataset
from dream_robot.policies.idm.data import decoded_frames

REPO_ID = "dream_robot/instructed_pick_place"
CAMERAS = {"top": "agentview_rgb_jpeg", "wrist": "eye_in_hand_rgb_jpeg"}
JPEG_QUALITY = 95


def jpeg(frame_chw: np.ndarray) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame_chw.transpose(1, 2, 0))).save(
        buffer, format="JPEG", quality=JPEG_QUALITY)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    table = open_dataset(args.repo_id, args.root).hf_dataset
    episode_index = np.asarray(table["episode_index"])
    states = np.asarray(table[STATE_FEATURE], dtype=np.float32)
    actions = np.asarray(table[ACTION_FEATURE], dtype=np.float32)
    frames = {c: np.load(p, mmap_mode="r")
              for c, p in decoded_frames(args.repo_id, args.root, tuple(CAMERAS)).items()}
    summary = json.loads((args.root / "recording_summary.json").read_text())
    kept = [e for e in summary["episodes"] if e["kept"]]   # dataset episode i = i-th kept

    by_prompt: dict[str, list[int]] = defaultdict(list)
    for episode, row in enumerate(kept):
        by_prompt[row["prompt"]].append(episode)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for prompt, episodes in sorted(by_prompt.items()):
        name = prompt.replace(" ", "_") + "_demo.hdf5"
        with h5py.File(args.out / name, "w") as f:
            for demo, episode in enumerate(episodes):
                rows = np.flatnonzero(episode_index == episode)
                group = f.create_group(f"data/demo_{demo}")
                for camera, key in CAMERAS.items():
                    data = group.create_dataset(
                        f"obs/{key}", (rows.size,), dtype=h5py.vlen_dtype(np.uint8))
                    for i, row in enumerate(rows):
                        data[i] = jpeg(frames[camera][row])
                group.create_dataset("actions", data=actions[rows])
                group.create_dataset("robot_states", data=states[rows])
                manifest.append({"file": name, "demo": demo, "episode": episode,
                                 "seed": kept[episode]["seed"], "steps": int(rows.size)})
        print(f"{name}: {len(episodes)} demos", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest)} demos in {len(by_prompt)} files -> {args.out}")


if __name__ == "__main__":
    main()
