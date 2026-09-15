"""Where does BC freeze, vs where the (shaky) expert closes? xy error and height above cube centre."""
import json, os, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np, mujoco
from dream_robot.core.record import JointNoise
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy, EEF_SITE
from dream_robot.policies.bc.policy import BCPolicy
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

env = PickPlaceCube()
def site(): m, d = env.mujoco; return d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)

# expert pose at its first close command, on the shaky dataset's kept seeds
expert = ExpertActor(ExpertPolicy(env)); noise = JointNoise(0.05, env.embodiment)
seeds = [e["seed"] for e in json.load(open("data/robosuite/pick_place_cube_noise005/recording_summary.json"))["episodes"] if e["kept"]]
xy, dz = [], []
for seed in seeds:
    obs = env.reset(seed=seed); expert.reset(); rng = np.random.default_rng(seed); d, sid = site()
    while True:
        a = expert(obs)
        if a[7] < 0.5:
            g = d.site_xpos[sid] - env.cube_pos; xy.append(np.linalg.norm(g[:2]) * 100); dz.append(g[2] * 100); break
        r = env.step(noise(a, rng)); obs = r.observation
print(f"shaky EXPERT at close: xy median {np.median(xy):.2f} (max {np.max(xy):.2f}) cm | height median {np.median(dz):.2f} cm (range {np.min(dz):.2f}..{np.max(dz):.2f})", flush=True)

for name, ckpt in [("clean BC", "experiments/bc_pick_place_cube/checkpoint.pt"), ("shaky BC", "experiments/bc_noise005/checkpoint.pt")]:
    bc = BCPolicy.load(ckpt); fx, fz = [], []
    for seed in range(1000, 1020):
        obs = env.reset(seed=seed); bc.reset(); d, sid = site(); closed = False
        while True:
            a = bc(obs); closed |= a[7] < 0.5
            r = env.step(a); obs = r.observation
            if r.terminated or r.truncated: break
        if closed: continue                                   # only the frozen-open episodes
        g = d.site_xpos[sid] - env.cube_pos; fx.append(np.linalg.norm(g[:2]) * 100); fz.append(g[2] * 100)
        print(f"  {name} seed {seed}: frozen at xy {fx[-1]:.2f} cm, height {fz[-1]:+.2f} cm", flush=True)
    print(f"{name} FROZEN ({len(fx)} eps): xy median {np.median(fx):.2f} cm | height median {np.median(fz):+.2f} cm (range {np.min(fz):+.2f}..{np.max(fz):+.2f})", flush=True)
env.close()
