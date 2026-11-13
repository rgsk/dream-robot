# The T5 square-up expert without shake, target 0.5 cm below the cube centre (grasp_dz -0.005): placed of 50,
# then films seeds 0, 19, 2 (the most diagonal named cubes) -> scratch/verne07/clean_low_seed<s>_<result>.mp4
import logging
import os
from dataclasses import replace

os.environ.setdefault('MUJOCO_GL', 'egl')
logging.disable(logging.WARNING)
import robosuite  # noqa: F401,E402

logging.disable(logging.NOTSET)
logging.getLogger('robosuite_logs').setLevel(logging.ERROR)

from dream_robot.core.video import write_video  # noqa: E402
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import (  # noqa: E402
    InstructedPickPlace,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.instructed_pick_place.expert import (  # noqa: E402
    TASK_YAML,
    ExpertConfig,
    ExpertPolicy,
)
from dream_robot.sims.robosuite.tasks.pick_place_cube.demo import rollout_frames  # noqa: E402

env = InstructedPickPlace(TaskConfig.load(), pairs='seen', wording='template')
policy = ExpertPolicy(env, replace(ExpertConfig.load(TASK_YAML), grasp_dz=-0.005), square_up=True)
ok = 0
for seed in range(50):
    obs = env.reset(seed=seed)
    policy.reset()
    while True:
        r = env.step(policy(obs)[0])
        obs = r.observation
        if r.terminated or r.truncated:
            break
    ok += bool(r.success)
print(f'clean, target -0.5 cm: {ok}/50 placed', flush=True)
for seed in (0, 19, 2):
    frames, placed, why = rollout_frames(env, policy, seed, hold=15)
    path = write_video(frames, f'scratch/verne07/clean_low_seed{seed}_{why.replace(" ", "_")}.mp4', fps=env.control_hz)
    print(f'seed {seed} "{env.instruction.prompt}": {why} -> {path}', flush=True)
env.close()
