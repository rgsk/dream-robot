"""Is spilling a threshold in tilt, or does a small tilt spill if you wait?

    MUJOCO_GL=egl uv run python experiments/t6/scripts/spill_vs_time.py

Review question: does more tilt spill sooner and less tilt spill later, or is
there an angle below which nothing ever leaves?

**Why this rig tilts gradually.** scripts/t6_spill_curve.py builds the tray
already at the target angle and drops the marbles into it, so they slide the
tray's whole length and arrive at the lip with speed. That is what put the
calibration at 0.76 r and it is not what the task does -- the arms take about a
second to tilt the tray, and the marbles creep down and settle. Here the tray
stays level and GRAVITY is rotated instead, over a ramp, which is the same
motion without needing to move a static body. Then it holds, and counts.
"""
import importlib.util

import mujoco
import numpy as np

spec = importlib.util.spec_from_file_location("sc", "scripts/t6_spill_curve.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)

RADIUS, COUNT, U = 0.018, 9, 0.293
G = 9.81
SAMPLES = (1.0, 2.0, 5.0, 10.0, 30.0)
ANGLES = (5, 10, 15, 20, 25, 30, 45, 60)
RAMPS = {"1.0 s (as the arms tilt it)": 1.0}


def aboard(model, data, half_z):
    pot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pot")
    rot = data.xmat[pot].reshape(3, 3)
    pos = data.xpos[pot]
    floor = -half_z + sc.WALL
    n = 0
    for i in range(COUNT):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"m{i}")
        local = rot.T @ (data.xpos[bid] - pos)
        if (np.all(np.abs(local[:2]) < sc.INNER_XY)
                and floor - RADIUS < local[2] < floor + 6 * RADIUS):
            n += 1
    return n


def run(angle_deg, ramp, half_z):
    model = mujoco.MjModel.from_xml_string(sc.build(COUNT, RADIUS, 0.0, half_z))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    theta = np.radians(angle_deg)
    dt = model.opt.timestep
    # Settle level first, so nothing is counted as spilled that was simply
    # dropped in from a millimetre up.
    for _ in range(int(0.5 / dt)):
        mujoco.mj_step(model, data)
    out, t = [], 0.0
    for mark in SAMPLES:
        while t < ramp + mark:
            frac = min(1.0, t / ramp)
            a = frac * theta
            model.opt.gravity[:] = [0.0, -G * np.sin(a), -G * np.cos(a)]
            mujoco.mj_step(model, data)
            t += dt
        out.append(aboard(model, data, half_z))
    return out


def main():
    half_z = (U * RADIUS + sc.WALL) / 2.0
    need = int(np.ceil(0.75 * COUNT))
    print(f"tray lip {U * RADIUS * 1000:.1f} mm, {COUNT} marbles r={RADIUS*1000:.0f} mm; "
          f"the task fails below {need} aboard")
    for label, ramp in RAMPS.items():
        print(f"\ntilted over {label}; marbles aboard after holding:")
        print(f"{'tilt':>5} | " + "  ".join(f"{s:>4.0f}s" for s in SAMPLES))
        for angle in ANGLES:
            row = run(angle, ramp, half_z)
            print(f"{angle:>4}° | " + "  ".join(f"{v:>5}" for v in row))


if __name__ == "__main__":
    main()
