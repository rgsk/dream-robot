"""Build their loader's Zarr cache for one of our converted datasets (the pod only needs this file).

    envs/uva/.venv/bin/python experiments/uva/scripts/build_cache.py --data data/uva/t5 --image 224 --wrist
"""

import argparse
import sys
import time
from pathlib import Path

import hydra

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "envs/uva/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_cfg import build_cfg  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", type=Path, required=True)
p.add_argument("--image", type=int, default=128)
p.add_argument("--wrist", action="store_true")
args = p.parse_args()
repo = Path(__file__).resolve().parents[3]
cfg = build_cfg(repo / "envs/uva/ckpt/libero10.ckpt", args.data.resolve(), args.data.name, Path("/tmp/c"),
                image=args.image, wrist=args.wrist)
t0 = time.time()
ds = hydra.utils.instantiate(cfg.task.dataset)
print(f"{len(ds)} windows, keys {list(ds.replay_buffer.keys())}, {time.time() - t0:.0f} s")
