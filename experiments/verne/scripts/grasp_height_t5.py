# Where does the T5 square-up + shake expert close? Grip-site height above the named cube's centre at the
# first close command, and the xy miss, for placed vs failed episodes. Same noise as record.py (rng per seed).
# usage: python experiments/verne/scripts/grasp_height_t5.py 180 [grasp_dz] [fine_tolerance]
import logging
import os
import sys
from dataclasses import replace

os.environ.setdefault('MUJOCO_GL', 'egl')
logging.disable(logging.WARNING)
import robosuite  # noqa: F401,E402

logging.disable(logging.NOTSET)
logging.getLogger('robosuite_logs').setLevel(logging.ERROR)
import numpy as np  # noqa: E402

from dream_robot.core.record import JointNoise  # noqa: E402
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import (  # noqa: E402
    InstructedPickPlace,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.instructed_pick_place.expert import (  # noqa: E402
    TASK_YAML,
    ExpertConfig,
    ExpertPolicy,
)

n = int(sys.argv[1])
cfg = ExpertConfig.load(TASK_YAML)
if len(sys.argv) > 2:
    cfg = replace(cfg, grasp_dz=float(sys.argv[2]))
if len(sys.argv) > 3:
    cfg = replace(cfg, fine_tolerance=float(sys.argv[3]))
env = InstructedPickPlace(TaskConfig.load(), pairs='seen', wording='template')
policy, noise = ExpertPolicy(env, cfg, square_up=True), JointNoise(0.05, env.embodiment)
rows = []  # placed, dz (cm), dxy (cm), failure
for seed in range(n):
    obs = env.reset(seed=seed)
    policy.reset()
    rng, closed, was_open, dz, dxy = np.random.default_rng(seed), False, False, np.nan, np.nan
    while True:
        a, _ = policy(obs)
        was_open |= a[7] >= 0.5
        if a[7] < 0.5 and was_open and not closed:
            d = policy._ik.site_pos - env.cube_pos
            dz, dxy, closed = 100 * d[2], 100 * np.linalg.norm(d[:2]), True
        r = env.step(noise(a, rng))
        obs = r.observation
        if r.terminated or r.truncated:
            break
    rows.append((bool(r.success), dz, dxy, str(r.failure_mode.value)))
ok = np.array([r[0] for r in rows])
dz, dxy = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
print(f'grasp_dz {cfg.grasp_dz}, fine_tolerance {cfg.fine_tolerance}: placed {ok.sum()}/{n}')
for name, m in (('placed', ok), ('failed', ~ok)):
    print(f'  {name:6s} {m.sum():3d}: height above centre at close {np.nanmean(dz[m]):+.2f} cm '
          f'(range {np.nanmin(dz[m]):+.2f} .. {np.nanmax(dz[m]):+.2f}), xy miss {np.nanmean(dxy[m]):.2f} cm')
bins = [-9, 0.0, 0.5, 1.0, 1.5, 9]
for lo, hi in zip(bins, bins[1:]):
    m = (dz >= lo) & (dz < hi)
    print(f'  close at {lo:+.1f}..{hi:+.1f} cm: {m.sum():3d} eps, placed {ok[m].mean() if m.any() else float("nan"):.0%}')
env.close()
