"""Where a training sample's time goes in their loader (run on the pod, data cache present)."""

import cProfile
import os
import pstats
import sys
import time
from pathlib import Path

import hydra

SRC = Path(__file__).resolve().parents[3] / "envs/uva/src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from task_cfg import build_cfg  # noqa: E402
from uva_patch import decode_into_ram  # noqa: E402

repo = Path(__file__).resolve().parents[3]
os.chdir(SRC)
cfg = build_cfg(repo / "envs/uva/ckpt/libero10.ckpt", repo / "data/uva/pick_place_cube", "pick_place_cube", Path("/tmp/p"))
ds = hydra.utils.instantiate(cfg.task.dataset)
decode_into_ram(ds)
ds[0]
t0 = time.perf_counter()
for i in range(20):
    ds[i * 100]
print(f"{(time.perf_counter() - t0) / 20 * 1000:.0f} ms per sample")
cProfile.run("for i in range(5): ds[i * 50]", "/tmp/prof")
pstats.Stats("/tmp/prof").sort_stats("cumulative").print_stats(12)
