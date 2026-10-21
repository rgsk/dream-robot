"""T5 demos (our Cosmos-format HDF5) -> the LeRobot v2.1 layout VPP2's reader expects.

    envs/vpp2/.venv-cpu/bin/python experiments/vpp/scripts/to_lerobot21.py \
        --data data/cosmos_policy/t5_224 --out data/vpp/t5_lerobot21

What their reader (vpp2/datasets/lerobot/lerobot, a vendored LeRobot v2.1) loads, and what goes in:
- ``meta/info.json``: codebase v2.1, fps 30, features ``observation.images.image`` (top) and
  ``observation.images.wrist_image`` (wrist) as 224x224 videos, ``observation.state`` (our 7
  joints + gripper, 8-D: the same width as LIBERO's EEF-pose proprio), ``action`` (our absolute
  joint targets + gripper, 8-D), plus the index columns;
- ``meta/tasks.jsonl``: one task per instruction (the file name, as Cosmos and UWM read it);
- ``meta/episodes.jsonl`` / ``meta/episodes_stats.jsonl``: per-episode length, task and min / max /
  mean / std of state and action (their loader aggregates these);
- ``data/chunk-000/episode_XXXXXX.parquet``: one row per step;
- ``videos/chunk-000/<key>/episode_XXXXXX.mp4``: H.264, every frame a key frame, so their
  timestamp seeks land exactly (tolerance 1e-4 s).
Their own writer in that copy is modified (it no longer saves images), so this writes the files
directly. Frames are upright already; no flip.
"""

from __future__ import annotations

import argparse
import io
import json
from fractions import Fraction
from pathlib import Path

import av
import h5py
import numpy as np
import pandas as pd
from PIL import Image

FPS = 30
CAMERAS = {"observation.images.image": "obs/agentview_rgb_jpeg",
           "observation.images.wrist_image": "obs/eye_in_hand_rgb_jpeg"}
NAMES = [f"joint_{i}" for i in range(7)] + ["gripper"]


def write_video(frames: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        stream.time_base = Fraction(1, FPS)
        stream.options = {"crf": "18", "g": "1"}  # all key frames: exact seeks
        for i, f in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(f, format="rgb24")
            frame.pts = i
            out.mux(stream.encode(frame))
        out.mux(stream.encode())


def stats(x: np.ndarray) -> dict:
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; remove it first")
    (args.out / "meta").mkdir(parents=True)
    (args.out / "data/chunk-000").mkdir(parents=True)

    tasks, episodes, ep_stats, total = {}, [], [], 0
    for path in sorted(args.data.glob("*_demo.hdf5")):
        prompt = path.name[: -len("_demo.hdf5")].replace("_", " ")
        task_index = tasks.setdefault(prompt, len(tasks))
        with h5py.File(path) as f:
            for d in range(len(f["data"])):
                g = f[f"data/demo_{d}"]
                ep = len(episodes)
                state = g["robot_states"][:].astype(np.float32)
                action = g["actions"][:].astype(np.float32)
                n = len(action)
                for key, src in CAMERAS.items():
                    frames = np.stack([np.array(Image.open(io.BytesIO(j.tobytes())).convert("RGB")) for j in g[src]])
                    write_video(frames, args.out / f"videos/chunk-000/{key}/episode_{ep:06d}.mp4")
                pd.DataFrame({
                    "observation.state": list(state), "action": list(action),
                    "timestamp": np.arange(n, dtype=np.float32) / FPS,
                    "frame_index": np.arange(n), "episode_index": np.full(n, ep),
                    "index": np.arange(total, total + n), "task_index": np.full(n, task_index),
                }).to_parquet(args.out / f"data/chunk-000/episode_{ep:06d}.parquet")
                episodes.append({"episode_index": ep, "tasks": [prompt], "length": n})
                ep_stats.append({"episode_index": ep, "stats": {"observation.state": stats(state),
                                                                 "action": stats(action)}})
                total += n
        print(f"{prompt}: {len(episodes)} episodes so far", flush=True)

    video = {"dtype": "video", "shape": [224, 224, 3], "names": ["height", "width", "channels"],
             "info": {"video.fps": FPS, "video.height": 224, "video.width": 224, "video.channels": 3,
                      "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                      "has_audio": False}}
    scalar = lambda dtype: {"dtype": dtype, "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.1", "robot_type": "panda", "fps": FPS,
        "total_episodes": len(episodes), "total_frames": total, "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(CAMERAS), "total_chunks": 1, "chunks_size": 1000,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            **{key: video for key in CAMERAS},
            "observation.state": {"dtype": "float32", "shape": [8], "names": NAMES},
            "action": {"dtype": "float32", "shape": [8], "names": NAMES},
            "timestamp": scalar("float32"), "frame_index": scalar("int64"),
            "episode_index": scalar("int64"), "index": scalar("int64"), "task_index": scalar("int64"),
        },
    }
    (args.out / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    jsonl = lambda rows: "".join(json.dumps(r) + "\n" for r in rows)
    (args.out / "meta/tasks.jsonl").write_text(jsonl({"task_index": i, "task": t} for t, i in tasks.items()))
    (args.out / "meta/episodes.jsonl").write_text(jsonl(episodes))
    (args.out / "meta/episodes_stats.jsonl").write_text(jsonl(ep_stats))
    print(f"done: {len(episodes)} episodes, {total} frames, {len(tasks)} tasks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
