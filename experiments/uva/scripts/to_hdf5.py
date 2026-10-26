"""Our LeRobot demos -> the LIBERO-style HDF5 files UVA's loader reads.

    uv run --with h5py python experiments/uva/scripts/to_hdf5.py \
        --root data/robosuite/pick_place_cube --out data/uva/pick_place_cube

What their LiberoReplayImageDataset reads (unified_video_action/dataset/libero_replay_image_dataset.py),
and what goes in it:
- one file per instruction, ``<words_joined_by_underscores>_demo.hdf5``: the loader takes the
  instruction from the file name (it strips the last 10 characters, ``_demo.hdf5``);
- ``data/demo_<i>/obs/agentview_rgb`` / ``eye_in_hand_rgb`` (T, H, W, 3) uint8: our top / wrist
  frames, raw (their loader copies them into Zarr as is; Cosmos's files hold JPEG bytes instead),
  stored **upside down**: LIBERO renders are, and their loader flips every frame vertically
  (rot90 x2 then a left-right flip), so ours come out upright as the checkpoint saw LIBERO.
  At run time, feed upright frames straight to the policy (no flip);
- ``data/demo_<i>/obs/joint_qpos`` (T, 8): our ``observation.state``, 7 joints + gripper; read
  as a ``low_dim`` key once the task config lists it (the name must end in ``qpos``: their
  normalizer range-scales those and rejects names it doesn't know);
- ``data/demo_<i>/actions`` (T, 8): our absolute joint targets + gripper, as recorded. The config
  must set ``abs_action: false``, else their loader converts 7-D pose actions to 10-D.
A ``manifest.json`` maps every (file, demo) back to our episode index and seed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from dream_robot.core.dataset import ACTION_FEATURE, STATE_FEATURE, open_dataset
from dream_robot.policies.idm.data import decoded_frames

CAMERAS = {"top": "agentview_rgb", "wrist": "eye_in_hand_rgb"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.root / "recording_summary.json").read_text())
    repo_id = summary["repo_id"]
    table = open_dataset(repo_id, args.root).hf_dataset
    episode_index = np.asarray(table["episode_index"])
    states = np.asarray(table[STATE_FEATURE], dtype=np.float32)
    actions = np.asarray(table[ACTION_FEATURE], dtype=np.float32)
    frames = {c: np.load(p, mmap_mode="r")
              for c, p in decoded_frames(repo_id, args.root, tuple(CAMERAS)).items()}
    kept = [e for e in summary["episodes"] if e["kept"]]   # dataset episode i = i-th kept

    # one prompt per run (pick_place_cube) or one per episode (instructed tasks)
    by_prompt: dict[str, list[int]] = defaultdict(list)
    for episode, row in enumerate(kept):
        by_prompt[row.get("prompt", summary.get("task_prompt"))].append(episode)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for prompt, episodes in sorted(by_prompt.items()):
        name = prompt.replace(" ", "_") + "_demo.hdf5"
        with h5py.File(args.out / name, "w") as f:
            for demo, episode in enumerate(episodes):
                rows = np.flatnonzero(episode_index == episode)
                group = f.create_group(f"data/demo_{demo}")
                for camera, key in CAMERAS.items():
                    upside_down = frames[camera][rows][:, :, ::-1]  # (T, 3, H, W), H reversed
                    group.create_dataset(f"obs/{key}", data=upside_down.transpose(0, 2, 3, 1))
                group.create_dataset("obs/joint_qpos", data=states[rows])
                group.create_dataset("actions", data=actions[rows])
                manifest.append({"file": name, "demo": demo, "episode": episode,
                                 "seed": kept[episode]["seed"], "steps": int(rows.size)})
        print(f"{name}: {len(episodes)} demos", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(manifest)} demos in {len(by_prompt)} files -> {args.out}")


if __name__ == "__main__":
    main()
