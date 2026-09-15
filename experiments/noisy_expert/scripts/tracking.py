import os, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np
from dream_robot.core.schema import joint_tracking_error
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

env = PickPlaceCube(); expert = ExpertActor(ExpertPolicy(env))
for sigma in [0.0, 0.03, 0.05]:
    rng = np.random.default_rng(0); lab_m, lab_x, exe_m, exe_x, wins = [], [], [], [], 0
    for seed in range(5):
        obs = env.reset(seed=seed); expert.reset(); S, L, E = [], [], []
        while True:
            clean = expert(obs); executed = clean.copy(); executed[:7] += rng.normal(0, sigma, 7)
            S.append(obs.state.copy()); L.append(clean); E.append(executed)
            r = env.step(executed); obs = r.observation
            if r.terminated or r.truncated: break
        wins += r.success
        m, x = joint_tracking_error(np.array(L), np.array(S), env.embodiment); lab_m.append(m); lab_x.append(x)
        m, x = joint_tracking_error(np.array(E), np.array(S), env.embodiment); exe_m.append(m); exe_x.append(x)
    print(f"sigma {sigma}: success {wins}/5 | clean-label tracking mean {np.mean(lab_m):.3f} max {np.max(lab_x):.3f}"
          f" | executed tracking mean {np.mean(exe_m):.3f} max {np.max(exe_x):.3f}  (limit 0.25)")
env.close()
