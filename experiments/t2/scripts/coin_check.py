"""T2 step 1b: is the coin fair, and independent of cube placement?

    MUJOCO_GL=glfw uv run python experiments/t2/scripts/coin_check.py

A coin correlated with the cube's spawn would hand the policy a visible cue for
which bin, and T2 would stop testing multimodality.
"""

import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import PickPlaceTwoBins
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.expert import ExpertPolicy

env = PickPlaceTwoBins()
expert = ExpertPolicy(env)
for name, seeds in (("record 0-99", range(100)), ("eval 1000-1019", range(1000, 1020))):
    coins, cubes = [], []
    for s in seeds:
        env.reset(seed=s)
        expert.reset()
        coins.append(expert.target_bin)
        cubes.append(env.cube_pos[:2])
    coins, cubes = np.array(coins), np.array(cubes)
    corr = [float(np.corrcoef(coins, cubes[:, k])[0, 1]) for k in (0, 1)]
    print(f"{name}: left={int(coins.sum())}/{len(coins)} "
          f"corr(coin, cube x)={corr[0]:+.2f} corr(coin, cube y)={corr[1]:+.2f} "
          f"first 12: {''.join('LR'[1 - c] for c in coins[:12])}")
env.close()
