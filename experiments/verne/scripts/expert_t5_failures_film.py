# Films failed attempts of the T5 square + shake (sigma 0.05) recording, replayed exactly (rng per seed as the
# recorder) -> scratch/verne07/fail_seed<s>_<mode>.mp4. usage: ... expert_t5_failures_film.py 0 3 105 18 41 54
import logging
import os
import sys

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
from dream_robot.sims.robosuite.tasks.instructed_pick_place.expert import ExpertPolicy  # noqa: E402

env = InstructedPickPlace(TaskConfig.load(), pairs='seen', wording='template')
policy, noise = ExpertPolicy(env, square_up=True), JointNoise(0.05, env.embodiment)
for seed in map(int, sys.argv[1:]):
    obs = env.reset(seed=seed)
    policy.reset()
    rng, frames, phases = np.random.default_rng(seed), [], []
    while True:
        a, cmd = policy(obs)
        phases.append(cmd.phase)
        frames.append(observation_panel(env.render(), obs.images))
        r = env.step(noise(a, rng))
        obs = r.observation
        if r.terminated or r.truncated:
            break
    why = 'placed' if r.success else str(r.failure_mode.value)
    path = write_video(frames, f'scratch/verne07/fail_seed{seed}_{why}.mp4', fps=env.control_hz)
    seq = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    print(f'seed {seed} "{env.instruction.prompt}": {why}; phases {" > ".join(seq)} (ends in {phases[-1]} for '
          f'{len(phases) - max(i for i, p in enumerate(phases) if i == 0 or p != phases[i - 1])} steps) -> {path}', flush=True)
env.close()
