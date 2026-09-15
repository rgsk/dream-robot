import os, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np, mujoco
from PIL import Image, ImageDraw, ImageFont
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy, EEF_SITE
from dream_robot.policies.bc.policy import BCPolicy
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

env = PickPlaceCube()
pols = {"expert": ExpertActor(ExpertPolicy(env)), "BC": BCPolicy.load("experiments/bc_pick_place_cube/checkpoint.pt")}

def wrist_at_arrival(p, seed):
    obs = env.reset(seed=seed); p.reset(); frames, xy, arrival = [], None, None
    m, d = env.mujoco; sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    while True:
        g = d.site_xpos[sid] - env.cube_pos
        if arrival is None and g[2] < 0.03: arrival, xy = len(frames), np.linalg.norm(g[:2]) * 100
        frames.append(obs.images["wrist"].copy())
        r = env.step(p(obs)); obs = r.observation
        if r.terminated or r.truncated: return frames[min(arrival + 15, len(frames) - 1)], xy, r.success

panels = [("expert", 1000), ("BC", 1000), ("expert", 1005), ("BC", 1005), ("BC", 1010)]
S, GAP, HEAD = 3, 24, 58
font = ImageFont.load_default(size=22); small = ImageFont.load_default(size=17)
W = 128 * S * 5 + GAP * 2
canvas = Image.new("RGB", (W, 128 * S + HEAD), "white"); draw = ImageDraw.Draw(canvas)
x = 0
for i, (name, seed) in enumerate(panels):
    if i in (2, 4): x += GAP
    frame, xy, ok = wrist_at_arrival(pols[name], seed)
    canvas.paste(Image.fromarray(np.kron(frame, np.ones((S, S, 1), dtype=np.uint8))), (x, HEAD))
    color = "green" if ok else "red"
    draw.text((x + 8, 4), f"{name} · seed {seed}", fill="black", font=font)
    draw.text((x + 8, 32), f"{xy:.2f} cm off · {'success' if ok else 'FAIL'}", fill=color, font=small)
    print(name, seed, f"{xy:.2f} cm", ok)
    x += 128 * S
canvas.save("scratch/noisy_expert/wrist_at_arrival.png")
env.close()
