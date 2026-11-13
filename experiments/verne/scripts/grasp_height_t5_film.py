# Films the T5 square-up + shake expert with the old target (grasp_dz +0.5 cm) and the new one (-0.5 cm) on the
# same seeds: the first 3 seeds where the old one fails -> scratch/verne07/grasp_seed<s>_dz<+-x>_<result>.mp4
import logging
import os
from dataclasses import replace

os.environ.setdefault('MUJOCO_GL', 'egl')
logging.disable(logging.WARNING)
import robosuite  # noqa: F401,E402

logging.disable(logging.NOTSET)
logging.getLogger('robosuite_logs').setLevel(logging.ERROR)
import numpy as np  # noqa: E402

from dream_robot.core.record import JointNoise  # noqa: E402
from dream_robot.core.video import observation_panel, write_video  # noqa: E402
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import (  # noqa: E402
    InstructedPickPlace,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.instructed_pick_place.expert import (  # noqa: E402
    TASK_YAML,
    ExpertConfig,
    ExpertPolicy,
)

env = InstructedPickPlace(TaskConfig.load(), pairs='seen', wording='template')
noise = JointNoise(0.05, env.embodiment)


def episode(dz, seed, film):
    policy = ExpertPolicy(env, replace(ExpertConfig.load(TASK_YAML), grasp_dz=dz), square_up=True)
    obs = env.reset(seed=seed)
    policy.reset()
    rng, frames = np.random.default_rng(seed), []
    while True:
        a, _ = policy(obs)
        if film:
            frames.append(observation_panel(env.render(), obs.images))
        r = env.step(noise(a, rng))
        obs = r.observation
        if r.terminated or r.truncated:
            break
    return r, frames


found = 0
for seed in range(180):
    r, _ = episode(0.005, seed, film=False)
    if r.success:
        continue
    for dz in (0.005, -0.005):
        r, frames = episode(dz, seed, film=True)
        why = 'placed' if r.success else str(r.failure_mode.value)
        path = write_video(frames + [frames[-1]] * 15, f'scratch/verne07/grasp_seed{seed}_dz{100 * dz:+.1f}cm_{why}.mp4', fps=env.control_hz)
        print(f'seed {seed} "{env.instruction.prompt}", target {100 * dz:+.1f} cm: {why} -> {path}', flush=True)
    found += 1
    if found == 3:
        break
env.close()
