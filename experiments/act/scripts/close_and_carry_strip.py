"""Front + wrist frames around the close and the carry, to check a failure by eye.
    python experiments/act/scripts/close_and_carry_strip.py <checkpoint> <seed> <out.png>"""
import os, sys, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.policies.act.policy import ACTPolicy
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

TIMES = [70, 90, 110, 140, 180, 230, 260]
env = PickPlaceCube(); policy = ACTPolicy.load(sys.argv[1]); seed = int(sys.argv[2])
obs = env.reset(seed=seed); policy.reset(); z0 = env.cube_pos[2]; panels = []; t = 0
while t <= max(TIMES):
    if t in TIMES:
        front = env.render()                                          # 256x256
        wrist = np.kron(obs.images["wrist"], np.ones((2, 2, 1), dtype=np.uint8))   # 128 -> 256
        panels.append((t, np.concatenate([front, wrist], axis=0), obs.state[7], (env.cube_pos[2] - z0) * 100))
    obs = env.step(policy(obs)).observation; t += 1
env.close()

font = ImageFont.load_default(size=18)
canvas = Image.new("RGB", (256 * len(panels), 512 + 50), "white"); draw = ImageDraw.Draw(canvas)
for i, (t, img, opening, rise) in enumerate(panels):
    canvas.paste(Image.fromarray(img), (256 * i, 50))
    draw.text((256 * i + 6, 2), f"t={t}  opening {opening:.2f}", fill="black", font=font)
    draw.text((256 * i + 6, 24), f"cube rise {rise:+.1f} cm", fill="red" if rise < 3 else "green", font=font)
canvas.save(sys.argv[3]); print("saved", sys.argv[3])
