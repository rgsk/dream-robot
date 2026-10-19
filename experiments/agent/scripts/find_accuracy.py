"""How far is find()'s point from where the cube really is? T5, both cubes, N seeds.

Ground truth is read here only to score -- find never sees it. Truth for the
top-face centre is the cube's body position plus its half-height on z.
"""

import argparse

import imageio.v3 as iio
import numpy as np

from dream_robot.agent.perception import colour_mask, find, overlay
from dream_robot.core.registry import TASKS

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=20)
p.add_argument("--hw", type=int, default=256)
p.add_argument("--out", default="scratch/agent/find")
args = p.parse_args()

env = TASKS["robosuite/instructed_pick_place"]()
m = env._env.sim.model._model
errs, misses = [], 0
for seed in range(1000, 1000 + args.seeds):
    env.reset(seed=seed)
    ins = env.instruction
    frame = env.rgbd("top", (args.hw, args.hw))
    for color, pos, slot in (
        (ins.color, env.cube_pos, ins.target_slot),
        (ins.distractor_color, env.distractor_pos, 1 - ins.target_slot),
    ):
        body = env._env.cube_body_ids[slot]
        half = m.geom_size[m.geom_bodyid == body][0][2]
        truth = pos + [0, 0, half]
        found = find(frame, colour_mask(frame.rgb, color))
        if len(found) != 1:
            misses += 1
            print(f"seed {seed} {color:6s}: {len(found)} matches, wanted 1")
            continue
        (f,) = found
        e = (f.point - truth) * 100
        errs.append(e)
        cm = f"x {e[0]:+.2f} y {e[1]:+.2f} z {e[2]:+.2f}"
        print(f"seed {seed} {color:6s}: err cm {cm}  ({f.pixels} px)")
        if seed < 1003:
            iio.imwrite(f"{args.out}_{seed}_{color}.png", overlay(frame, found))

errs = np.array(errs)
xy = np.linalg.norm(errs[:, :2], axis=1)
print(f"\n{len(errs)} found, {misses} not exactly one match, at {args.hw}x{args.hw}")
print(f"xy error cm: median {np.median(xy):.2f}, max {xy.max():.2f}")
print(f"z error cm: median {np.median(errs[:, 2]):+.2f}, max |z| {np.abs(errs[:, 2]).max():.2f}")
