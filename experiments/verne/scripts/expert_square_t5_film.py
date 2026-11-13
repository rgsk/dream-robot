# Films the T5 square-up expert on the 3 seeds (of 0-29) whose named cube is most diagonal
# -> scratch/verne07/expert_t5_square_seed<s>_<deg>deg_<color>_<side>.mp4
import logging
import os

os.environ.setdefault('MUJOCO_GL', 'egl')
logging.disable(logging.WARNING)
import robosuite  # noqa: F401,E402

logging.disable(logging.NOTSET)
logging.getLogger('robosuite_logs').setLevel(logging.ERROR)
import numpy as np  # noqa: E402

from dream_robot.core.video import write_video  # noqa: E402
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import (  # noqa: E402
    InstructedPickPlace,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.instructed_pick_place.expert import ExpertPolicy  # noqa: E402
from dream_robot.sims.robosuite.tasks.pick_place_cube.demo import rollout_frames  # noqa: E402

env = InstructedPickPlace(TaskConfig.load(), pairs='seen', wording='template')
off = {}
for seed in range(30):
    env.reset(seed=seed)
    off[seed] = abs(np.degrees((env.cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4))
policy = ExpertPolicy(env, square_up=True)
for seed in sorted(off, key=off.get)[-3:]:
    frames, placed, why = rollout_frames(env, policy, seed, hold=15)
    ins = env.instruction
    path = write_video(
        frames,
        f'scratch/verne07/expert_t5_square_seed{seed}_{off[seed]:.0f}deg_{ins.color}_{ins.side}.mp4',
        fps=env.control_hz,
    )
    print(f'seed {seed}: "{ins.prompt}", cube {off[seed]:.0f} deg off square, {why} -> {path}', flush=True)
env.close()
