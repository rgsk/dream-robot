"""T2 step 4b: what decides the bin a policy picks? Test the cube's spawn position.

    MUJOCO_GL=glfw uv run python experiments/t2/scripts/side_vs_spawn.py experiments/t2/paths/bc.json

Per eval seed: cube spawn (x, y) at reset, the bin the rollout used, and the
expert's hidden coin. A strong spawn-to-side relation means the policy found a
visible cue; none means the tie is broken some other way (e.g. closed loop).
"""

import json
import sys

import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import PickPlaceTwoBins
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.expert import ExpertPolicy

run = json.load(open(sys.argv[1]))
env = PickPlaceTwoBins()
expert = ExpertPolicy(env)
rows = []
for e in run["episodes"]:
    env.reset(seed=e["seed"])
    expert.reset()
    x, y = env.cube_pos[:2]
    rows.append((e["seed"], x, y, e["cube_where"], env.bin_names[expert.target_bin], e["commit_step"]))
env.close()

print("seed   cube_x   cube_y  policy  coin   commit")
for s, x, y, w, c, k in sorted(rows, key=lambda r: r[2]):
    print(f"{s}  {x:+.3f}  {y:+.3f}  {w:6s}  {c:5s}  {k}")
right = np.array([r[3] == "right" for r in rows], dtype=float)
for name, col in (("x", 1), ("y", 2)):
    v = np.array([r[col] for r in rows])
    print(f"corr(policy goes right, cube {name}) = {np.corrcoef(right, v)[0, 1]:+.2f}")
print(f"agreement with the hidden coin: {sum(r[3] == r[4] for r in rows)}/{len(rows)}")
