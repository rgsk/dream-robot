# Can the scripted expert still pick and place when the cube spawns over a wider range?
# usage: python experiments/verne/scripts/expert_wide.py -0.10 0.10 -0.10 0.06 30   (x min max, y min max in m, episodes)
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import replace

os.environ.setdefault('MUJOCO_GL', 'egl')
logging.disable(logging.WARNING)
import robosuite  # noqa: F401

logging.disable(logging.NOTSET)
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube, TaskConfig
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertPolicy

x0, x1, y0, y1, n = *map(float, sys.argv[1:5]), int(sys.argv[5])
env = PickPlaceCube(replace(TaskConfig.load(), cube_xy_range=((x0, x1), (y0, y1))))
policy = ExpertPolicy(env)
outcomes, t0 = Counter(), time.time()
for seed in range(n):
    obs = env.reset(seed=seed)
    policy.reset()
    cube = env._raw['cube_pos'][:2]
    for _ in range(env.horizon_steps):
        action, command = policy(obs)
        result = env.step(action)
        obs = result.observation
        if result.success or result.truncated or command.timed_out:
            break
    why = 'placed' if result.success else (f'stuck in {command.phase}' if command.timed_out else str(result.failure_mode))
    outcomes[why] += 1
    print(f'seed {seed:3d} cube xy ({cube[0]:+.3f}, {cube[1]:+.3f}): {why}', flush=True)
print(f'x [{x0}, {x1}], y [{y0}, {y1}]: {outcomes["placed"]}/{n} placed  {dict(outcomes)}  {time.time() - t0:.0f} s')
