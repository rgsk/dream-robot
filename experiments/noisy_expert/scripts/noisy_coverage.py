import os, sys, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np, mujoco
from dream_robot.core.record import JointNoise
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy, EEF_SITE
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

LOW = 3.0
env = PickPlaceCube(); expert = ExpertActor(ExpertPolicy(env))
for sigma in [float(s) for s in sys.argv[1:]]:
    noise = JointNoise(sigma, env.embodiment)
    wins, ep_max, off1, steps = 0, [], [], []
    for seed in range(25):
        obs = env.reset(seed=seed); expert.reset(); rng = np.random.default_rng(seed)   # same seeding as the recorder
        m, d = env.mujoco; sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        rows = []
        while True:
            g = d.site_xpos[sid] - env.cube_pos
            rows.append((np.linalg.norm(g[:2]) * 100, g[2] * 100, obs.state[7]))
            r = env.step(noise(expert(obs), rng)); obs = r.observation
            if r.terminated or r.truncated: break
        wins += r.success; steps.append(len(rows))
        if not r.success: continue                      # recorder keeps successes only
        rows = np.array(rows); near = rows[(rows[:, 1] < LOW) & (rows[:, 2] > 0.9)]
        ep_max.append(near[:, 0].max() if len(near) else 0.0)
        off1.append(int((near[:, 0] > 1.0).sum()))
    ep_max = np.array(ep_max)
    print(f"sigma {sigma:<5} success {wins}/25  mean steps {np.mean(steps):.0f} | kept eps: "
          f"max off-centre at grasp height median {np.median(ep_max):.2f} cm, worst {ep_max.max():.2f} cm, "
          f"eps with >1 cm moment {np.sum(ep_max > 1.0)}/{len(ep_max)}, >2 cm {np.sum(ep_max > 2.0)}; "
          f"steps >1 cm total {sum(off1)}", flush=True)
env.close()
