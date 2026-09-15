"""T2 step 4c: do the training demos themselves lean one way by cube spawn?

    MUJOCO_GL=glfw uv run python experiments/t2/scripts/demo_side_vs_spawn.py

Reads the recorded coin per seed (bin_choices.json, kept episodes only), resets
each seed to get the cube spawn, and tabulates bin by spawn region. If BC's
rule on the eval seeds ("cube x > 0 -> right") is already a majority in the
demos, BC is copying a sampling accident of 100 coin flips.
"""

import json

import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import PickPlaceTwoBins

ROOT = "data/robosuite/pick_place_two_bins_noise0025_n100"
rows = [r for r in json.load(open(f"{ROOT}/bin_choices.json")) if r["kept"]]
env = PickPlaceTwoBins()
xy = []
for r in rows:
    env.reset(seed=r["seed"])
    xy.append(env.cube_pos[:2])
env.close()
xy = np.array(xy)
right = np.array([r["target_bin"] == "right" for r in rows])

print(f"kept demos: {len(rows)}, right {right.sum()} / left {(~right).sum()}")
for name, col in (("x", 0), ("y", 1)):
    v = xy[:, col]
    print(f"corr(demo goes right, cube {name}) = {np.corrcoef(right.astype(float), v)[0, 1]:+.2f}")
    for label, mask in ((f"cube {name} > 0", v > 0), (f"cube {name} < 0", v < 0)):
        print(f"  {label}: {right[mask].sum()} right / {(~right[mask]).sum()} left")
for label, mask in (("x>0,y>0", (xy[:, 0] > 0) & (xy[:, 1] > 0)), ("x>0,y<0", (xy[:, 0] > 0) & (xy[:, 1] < 0)),
                    ("x<0,y>0", (xy[:, 0] < 0) & (xy[:, 1] > 0)), ("x<0,y<0", (xy[:, 0] < 0) & (xy[:, 1] < 0))):
    print(f"  quadrant {label}: {right[mask].sum()} right / {(~right[mask]).sum()} left")
