"""T2 step 6b: per-seed agreement between path runs, and each run's side vs cube spawn x.

    MUJOCO_GL=glfw uv run python experiments/t2/scripts/compare_runs.py
"""

import json

import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import PickPlaceTwoBins

names = ("expert", "bc", "act", "act_sampled_z")
runs = {n: {e["seed"]: e for e in json.load(open(f"experiments/t2/paths/{n}.json"))["episodes"]} for n in names}
seeds = sorted(runs["bc"])
env = PickPlaceTwoBins()
cube_x = {}
for s in seeds:
    env.reset(seed=s)
    cube_x[s] = float(env.cube_pos[0])
env.close()

def side(n, s):
    return runs[n][s]["cube_where"]

for a, b in (("act", "act_sampled_z"), ("act", "bc"), ("bc", "expert"), ("act", "expert")):
    print(f"same bin, {a} vs {b}: {sum(side(a, s) == side(b, s) for s in seeds)}/20")
for n in names:
    right = np.array([side(n, s) == "right" for s in seeds], float)
    x = np.array([cube_x[s] for s in seeds])
    pos = [side(n, s) for s in seeds if cube_x[s] > 0]
    print(f"{n}: corr(right, cube x) {np.corrcoef(right, x)[0, 1]:+.2f}; cube x>0 -> "
          f"{pos.count('right')} right / {pos.count('left')} left")
# mid-carry reversals: gripper first heads >0.05 m one way, then ends in the other bin
for n in ("bc", "act"):
    flips = []
    for s in seeds:
        y = np.array(runs[n][s]["path"])[:, 1]
        end = np.sign(y[-1])
        if (end > 0 and y.min() < -0.05) or (end < 0 and y.max() > 0.05):
            flips.append(s)
    print(f"{n}: reversals mid-carry on seeds {flips}")
