# Film the scripted expert at a wider cube range: render view + the recorded cameras.
# usage: python experiments/verne/scripts/expert_video.py 0.10 0.10 5 11   (x range, y range, seeds...)
import logging
import os
import sys
from dataclasses import replace

os.environ.setdefault('MUJOCO_GL', 'egl')
logging.disable(logging.WARNING)
import robosuite  # noqa: F401

logging.disable(logging.NOTSET)
from dream_robot.core.video import write_video
from dream_robot.sims.robosuite.tasks.pick_place_cube.demo import rollout_frames
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube, TaskConfig
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertPolicy

rx, ry, seeds = float(sys.argv[1]), float(sys.argv[2]), [int(s) for s in sys.argv[3:]]
env = PickPlaceCube(replace(TaskConfig.load(), cube_xy_range=(rx, ry)))
policy = ExpertPolicy(env)
for seed in seeds:
    frames, placed, why = rollout_frames(env, policy, seed, hold=15)
    path = write_video(frames, f'scratch/expert_x{rx}_y{ry}_seed{seed}_{why.replace(" ", "_")}.mp4', fps=30)
    print(seed, why, path)
