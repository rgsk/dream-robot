"""Can one ball size give a GRADED spill -- count tracking angle and duration?

    MUJOCO_GL=egl uv run python experiments/t6/scripts/graded_spill.py

Today the spill is nearly binary: 9 identical balls on a 12 cm tray share one
escape angle, and each has 6 cm of run-up, so they arrive at the rim fast and
all leave inside a second once the angle is passed. Nothing distinguishes them.

Mixed sizes were considered and rejected on physics: the big ones reach the low
edge first and dam the small ones behind them, so the release is a queue, not a
size-ordered curve.

What is left, with one size, is the rim and **how many balls are on the tray**.
Fill matters in both directions and is swept rather than assumed: a packed tray
gives every ball a neighbour instead of 6 cm of run-up, so escape sits closer to
the static condition and is more angle-sensitive, and the balls behind have to
queue -- gradation in time. But a packed tray also crowds balls over the rim
that a sparse one would hold, so it is more fragile at low rims. Which effect
wins is a measurement.

The grid is ball radius x rim/radius x fill, tilt ramped over 1 s as the arms do
it, sampled at 1, 2 and 5 s, and ranked by how graded the resulting curve is.
"""
import numpy as np
import mujoco

# The SCENE's tray, not a proxy: body half-size 0.07 with 0.01 walls, so the
# cavity is 0.06 half-width. The earlier sweep used a 0.005 wall and therefore a
# wider cavity, which fits a 6x6 lattice where the task fits 5x5 -- the winning
# config then did not transfer as measured.
HALF, WALL = 0.07, 0.01
INNER = HALF - WALL
G = 9.81
ANGLES = (15, 20, 25, 30, 35, 40, 45, 50, 60)
SAMPLES = (1.0, 2.0, 5.0)
RADII = (0.008, 0.010, 0.012, 0.015)
RATIOS = (0.45, 0.60, 0.75)
FILLS = (0.5, 0.75, 1.0)                # fraction of a single full layer


def slots(per_row, pitch, count):
    """``count`` lattice positions, taken from the CENTRE outwards.

    Row-major from slot 0 fills one corner first, so a half-full tray sits
    against one wall and spills the moment it tips that way -- a placement
    artefact that reads as a fill effect. Sorting by distance from the centre
    keeps every fill symmetric and comparable.
    """
    grid = [((c - (per_row - 1) / 2.0) * pitch, (r - (per_row - 1) / 2.0) * pitch)
            for r in range(per_row) for c in range(per_row)]
    grid.sort(key=lambda xy: xy[0] ** 2 + xy[1] ** 2)
    return grid[:count]


def build(radius, count, per_row, half_z):
    pitch = 2.1 * radius
    floor = -half_z + WALL
    bodies = []
    for slot, (lx, ly) in enumerate(slots(per_row, pitch, count)):
        bodies.append(
            f'<body name="m{slot}" pos="{lx:.4f} {ly:.4f} {floor + radius:.4f}">'
            f'<freejoint/><geom type="sphere" size="{radius}" mass="0.01" condim="4" '
            f'friction="0.4 0.005 0.0001"/></body>'
        )
    walls = [f'<geom type="box" pos="0 0 {-half_z + WALL/2:.4f}" size="{HALF} {HALF} {WALL/2}"/>']
    for x, y in ((0.0, -HALF + WALL / 2), (0.0, HALF - WALL / 2),
                 (-HALF + WALL / 2, 0.0), (HALF - WALL / 2, 0.0)):
        walls.append(f'<geom type="box" pos="{x:.4f} {y:.4f} 0" '
                     f'size="{WALL/2 if x else HALF} {HALF if x else WALL/2} {half_z}"/>')
    return f"""
<mujoco><option timestep="0.002"/><worldbody>
  <geom name="ground" type="plane" pos="0 0 -1.0" size="5 5 0.1"/>
  <body name="pot" pos="0 0 0">{''.join(walls)}</body>
  {''.join(bodies)}
</worldbody></mujoco>"""


def aboard(model, data, count, radius, half_z):
    pot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pot")
    rot = data.xmat[pot].reshape(3, 3)
    pos = data.xpos[pot]
    floor = -half_z + WALL
    n = 0
    for i in range(count):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"m{i}")
        local = rot.T @ (data.xpos[b] - pos)
        if np.all(np.abs(local[:2]) < INNER) and floor - radius < local[2] < floor + 6 * radius:
            n += 1
    return n


def curve(radius, ratio, fill):
    rim = ratio * radius
    half_z = (rim + WALL) / 2.0
    pitch = 2.1 * radius
    per_row = max(1, int((2 * (INNER - radius)) // pitch) + 1)
    count = max(4, int(round(fill * per_row * per_row)))
    # Every ball must start clear of the rim. A lattice corner that overhangs
    # the wall loses balls at rest, which then looks like an early spill -- the
    # bug that produced a flaky "24 of 25" in the scene.
    reach = max(max(abs(x), abs(y)) for x, y in slots(per_row, pitch, count))
    clearance = INNER - (reach + radius)
    if clearance < 0.002:
        return count, None, clearance
    rows = []
    for angle in ANGLES:
        model = mujoco.MjModel.from_xml_string(build(radius, count, per_row, half_z))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        dt, theta, t = model.opt.timestep, np.radians(angle), 0.0
        for _ in range(int(0.5 / dt)):
            mujoco.mj_step(model, data)
        samples = []
        for mark in SAMPLES:
            while t < 1.0 + mark:
                a = min(1.0, t / 1.0) * theta
                model.opt.gravity[:] = [0.0, -G * np.sin(a), -G * np.cos(a)]
                mujoco.mj_step(model, data)
                t += dt
            samples.append(aboard(model, data, count, radius, half_z))
        rows.append(samples)
    return count, rows, clearance


def score(rows, count):
    """How graded is this curve?

    Three things, in the order they matter:
      * it must start full -- a tray that is already shedding at 15 deg is
        useless, because the clean expert tilts up to 7 deg;
      * distinct intermediate levels between full and empty;
      * monotone: every step down, never back up.
    """
    at2s = [r[1] for r in rows]
    starts_full = at2s[0] == count
    levels = len({round(10 * v / count) for v in at2s if 0 < v < count})
    back = sum(1 for a, b in zip(at2s, at2s[1:]) if b > a)
    band = sum(1 for v in at2s if 0 < v < count)      # angles in the graded band
    return starts_full, levels, back, band


def main():
    print("tilt ramped over 1 s then held; balls still aboard at 2 s, as % of n\n")
    header = (f"{'r(mm)':>6} {'rim/r':>6} {'rim':>5} {'n':>4} | " +
              " ".join(f"{a:>4}" for a in ANGLES) + " | full? lvls back band")
    print(header)
    results = []
    for radius in RADII:
        for ratio in RATIOS:
            for fill in FILLS:
                count, rows, clearance = curve(radius, ratio, fill)
                if rows is None:
                    print(f"{radius*1000:>6.0f} {ratio:>6.2f} {ratio*radius*1000:>5.1f} "
                          f"{count:>4} | lattice overhangs the rim by "
                          f"{-clearance*1000:.1f} mm -- skipped")
                    continue
                starts_full, levels, back, band = score(rows, count)
                results.append((radius, ratio, count, rows, starts_full, levels, back, band,
                                clearance))
                cells = " ".join(f"{100 * r[1] // count:>4}" for r in rows)
                print(f"{radius*1000:>6.0f} {ratio:>6.2f} {ratio*radius*1000:>5.1f} {count:>4} | "
                      f"{cells} | {'yes' if starts_full else ' no':>5} {levels:>4} "
                      f"{back:>4} {band:>4}")

    print("\nranked: starts full, then most levels, then fewest reversals, then widest band")
    ranked = sorted(results, key=lambda r: (not r[4], -r[5], r[6], -r[7]))
    for radius, ratio, count, rows, starts_full, levels, back, band, clearance in ranked[:6]:
        at2s = [r[1] for r in rows]
        print(f"  r={radius*1000:.0f} mm  rim={ratio*radius*1000:.1f} mm ({ratio:.2f} r)  "
              f"n={count:<3} levels={levels} reversals={back} "
              f"clearance={clearance*1000:.1f}mm  curve: {' '.join(str(v) for v in at2s)}")
        print(f"      duration check at the mid angle: " +
              " ".join(f"{s:.0f}s:{r}" for s, r in zip(SAMPLES, rows[len(rows) // 2])))


if __name__ == "__main__":
    main()
