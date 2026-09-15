"""Step 3 robustness probe: expert success when the executed arm joints get Gaussian noise.
Eval seeds 1000-1009. Result logged in ../notes.md."""
import os, time, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertPolicy
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

env = PickPlaceCube(); expert = ExpertPolicy(env)
rng = np.random.default_rng(0)
for sigma in [0.0, 0.02, 0.05, 0.1, 0.2]:
    wins, steps, modes, t0 = 0, [], {}, time.time()
    for seed in range(1000, 1010):
        obs = env.reset(seed=seed); expert.reset()
        while True:
            clean, _ = expert(obs)
            executed = clean.copy()
            executed[:7] += rng.normal(0, sigma, 7)   # arm only; label would stay `clean`
            r = env.step(executed); obs = r.observation
            if r.terminated or r.truncated: break
        wins += r.success; steps.append(env._t)
        if not r.success: modes[str(r.failure_mode)] = modes.get(str(r.failure_mode), 0) + 1
    print(f"sigma={sigma:<5} success {wins}/10  mean_steps {np.mean(steps):.0f}  fails {modes}  ({time.time()-t0:.0f}s)", flush=True)
env.close()
