"""find_objects vs ground truth on T7, N seeds.

Truth is read here only to score. Per seed: are the objects and the two bins all
found, each as the right kind? Per primitive (whose true size is known): xy error
of the footprint centre, height, width (the narrow side, what the grasp closes
on) and long-axis error for elongated ones. Meshes: found values printed per kind.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from dream_robot.agent.tabletop import find_objects, overlay
from dream_robot.core.registry import TASKS

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=30)
p.add_argument("--out", default="scratch/agent/find_objects")
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)


def truth(o, yaw):
    """(height, width, long-axis yaw or None) for a primitive."""
    s = o.size
    if o.kind in ("box", "bar"):
        hx, hy = s["half_x"], s["half_y"]
        axis = yaw if hx >= hy else yaw + np.pi / 2
        elongated = max(hx, hy) / min(hx, hy) > 1.3
        return 2 * s["half_z"], 2 * min(hx, hy), axis if elongated else None
    if o.kind == "cylinder":
        return 2 * s["half_height"], 2 * s["radius"], None
    return 2 * s["radius"], 2 * s["radius"], None


def axis_err(a, b):
    d = (a - b) % np.pi
    return np.rad2deg(min(d, np.pi - d))


env = TASKS["robosuite/assorted_pick_place"]()
counts_ok, rows, mesh, shapes = 0, defaultdict(list), defaultdict(list), defaultdict(list)
for seed in range(1000, 1000 + args.seeds):
    env.reset(seed=seed)
    frame = env.rgbd("top")
    things = find_objects(frame, env.table_xy, env.robot_mask("top"))
    objs = [t for t in things if t.kind == "object"]
    bins = [t for t in things if t.kind == "container"]
    ok = len(objs) == len(env.objects) and len(bins) == 2
    counts_ok += ok
    if not ok or seed < 1004:
        iio.imwrite(out / f"{seed}.png", overlay(frame, things))
    if not ok:
        print(f"seed {seed}: {len(objs)} objects (want {len(env.objects)}), {len(bins)} containers")
        continue
    pos = env.object_positions
    for o, p_ in zip(env.objects, pos):
        t = min(objs, key=lambda t: np.linalg.norm(t.centre[:2] - p_[:2]))
        xy_err = np.linalg.norm(t.centre[:2] - p_[:2])
        shapes[o.kind].append((t.footprint, t.top))
        if o.colour is None:
            mesh[o.kind].append((t.height, t.width, t.length, xy_err))
            continue
        # yaw from the body's current rotation, about z
        xm = env._env.sim.data.body_xmat[env._env.object_body_ids[env.objects.index(o)]].reshape(3, 3)
        yaw = np.arctan2(xm[1, 0], xm[0, 0])
        h, w, axis = truth(o, yaw)
        rows[o.kind].append((xy_err, t.height - h, t.width - w,
                             axis_err(t.axis_yaw, axis) if axis is not None else np.nan,
                             t.colour == o.colour))

print(f"\ncounts right (objects + 2 containers): {counts_ok}/{args.seeds}")
print(f"{'kind':9s} {'n':>3s} {'xy mm med/max':>15s} {'height mm med/max|':>18s} {'width mm med/max|':>18s} {'axis deg med/max':>17s} colour")
for k, r in sorted(rows.items()):
    a = np.array(r, dtype=float)
    def mm(col):
        v = np.abs(a[:, col]) * 1000
        return f"{np.median(v):6.1f}/{v.max():5.1f}"
    ax = a[:, 3][~np.isnan(a[:, 3])]
    axs = f"{np.median(ax):6.1f}/{ax.max():5.1f}" if len(ax) else "     -"
    print(f"{k:9s} {len(a):3d} {mm(0):>15s} {mm(1):>18s} {mm(2):>18s} {axs:>17s} {int(a[:, 4].sum())}/{len(a)}")
print("\nmeshes (found height / width / length cm, xy-from-origin mm):")
for k, r in sorted(mesh.items()):
    a = np.array(r)
    print(f"  {k:7s} n={len(a)}  h {a[:,0].mean()*100:4.1f}±{a[:,0].std()*100:.1f}  w {a[:,1].mean()*100:4.1f}±{a[:,1].std()*100:.1f}"
          f"  l {a[:,2].mean()*100:4.1f}  xy {np.median(a[:,3])*1000:.1f}")
print("\nshape words per kind (footprint, top):")
for k, r in sorted(shapes.items()):
    c = defaultdict(int)
    for x in r: c[x] += 1
    print(f"  {k:9s}", dict(c))
