"""T7 scenes: what does reset draw, and does it sit still? N seeds.

Per seed: the instruction, each object (kind, size, yaw), and how far each one
moved over 1 s of the arm holding still -- a ball rolling or a bottle tipping
at spawn would be a scene bug, not a picking failure. Saves a contact sheet of
the top and front views.
"""

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from dream_robot.core.registry import TASKS

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=12)
p.add_argument("--settle", type=int, default=30, help="steps holding still (30 = 1 s)")
p.add_argument("--out", default="scratch/agent/t7_scenes.png")
args = p.parse_args()

env = TASKS["robosuite/assorted_pick_place"]()
tiles, kinds, worst = [], {}, []
for seed in range(1000, 1000 + args.seeds):
    obs = env.reset(seed=seed)
    top = env.rgbd("top").rgb
    front = np.asarray(Image.fromarray(env.render()).resize(top.shape[1::-1]))
    start = env.object_positions
    hold = np.concatenate([obs.state[env.embodiment.arm_slice], [1.0]])   # arm still, gripper open
    for _ in range(args.settle):
        env.step(hold)
    moved = np.linalg.norm(env.object_positions - start, axis=1)

    print(f"seed {seed}: {env.prompt}")
    for i, (o, d) in enumerate(zip(env.objects, moved)):
        mark = "*" if i == env.instruction.target else " "
        print(f"  {mark} {o.name:14s} size {o.size}  yaw {np.rad2deg(o.yaw):6.1f}  moved {d*1000:5.1f} mm")
        kinds[o.kind] = kinds.get(o.kind, 0) + 1
        worst.append((d, seed, o.name))

    tile = Image.fromarray(np.concatenate([top, front], axis=1))
    ImageDraw.Draw(tile).text((4, 4), f"{seed}: {env.prompt}", fill=(255, 255, 255))
    tiles.append(np.asarray(tile))

worst.sort(reverse=True)
print("kinds drawn:", dict(sorted(kinds.items())))
print("most movement:", [(f"{d*1000:.1f} mm", s, n) for d, s, n in worst[:5]])

rows = [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, len(tiles) - 1, 2)]
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
iio.imwrite(out, np.concatenate(rows, axis=0))
print("wrote", out)
