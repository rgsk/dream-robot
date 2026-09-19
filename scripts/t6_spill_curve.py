"""At what tilt does the pot actually spill its marbles?

The marbles exist so that tilting the pot *costs* something. That only works if
the angle they leave at is the angle we call a failure -- otherwise the video
shows a pot at 40 degrees holding everything while the run is logged as a
failure, and the number and the picture disagree.

    Spill angle, treating the marbles as a fluid
    --------------------------------------------
    Cavity: half-width a, inner depth d (floor to rim), filled to height h
    above the floor. Tilt by θ. The free surface stays horizontal, so in the
    pot's frame it is a line of slope tan θ through the fill level:

        z(x) = h + x·tan θ,        x ∈ [-a, +a]

    Contents reach the rim on the high side when z(+a) = d:

        h + a·tan θ = d      ⟹      θ_spill = atan( (d - h) / a )

    The spill angle is set by the FREEBOARD (d - h) against the half-width. For
    θ_spill = 45°, freeboard must equal half-width: d - h = a.

    robosuite's pot: a = 0.060, d = 0.130. So an empty-ish pot spills at
    atan(0.106/0.060) ≈ 60°, and hitting 45° needs the fill raised to
    h = d - a = 0.070 -- which is a pot filled most of the way up, not eight
    small marbles rolling around a deep box.

Measured on a STANDALONE model of the same cavity rather than inside robosuite:
the pot there is a free body resting on a table, and every way of holding it
still at an angle (rewriting its pose, pinning its velocity, faking its mass)
feeds the contact solver impulses it then discards, which walks marbles through
the 5 mm walls. A static body has none of that problem.

Run: MUJOCO_GL=glfw uv run python scripts/t6_spill_curve.py [count] [radius]
"""

from __future__ import annotations

import sys

import mujoco
import numpy as np

# robosuite's PotWithHandlesObject, collision geometry only: body half-size
# 0.07 and half-thickness 0.005, giving a 0.12 x 0.12 x 0.13 cavity.
HALF = 0.07
WALL = 0.005
INNER_XY = HALF - WALL          # 0.060
FLOOR_Z = -HALF + WALL          # -0.060
RIM_Z = HALF                    # +0.070

ANGLES_DEG = (0, 10, 20, 30, 35, 40, 45, 50, 55, 60, 70)
SETTLE_SECONDS = 2.5


def lattice(radius: float) -> tuple[float, int]:
    """Pitch and marbles-per-row for a lattice that fits inside the cavity.

    Packing across the full cavity width matters: a lattice sized to a 6 cm box
    with a generous pitch runs out of columns, stacks the surplus into layers
    that start ABOVE the rim, and those marbles are outside before the clock
    starts. The symptom is a count that is short by a constant at every angle.
    """
    pitch = 2.1 * radius
    usable = 2 * (INNER_XY - radius)
    return pitch, max(1, int(usable // pitch) + 1)


def build(count: int, radius: float, tilt_deg: float, half_z: float = HALF) -> str:
    """A static pot at ``tilt_deg``, marbles stacked on a lattice inside it."""
    floor_z, rim_z = -half_z + WALL, half_z
    pitch, per_row = lattice(radius)
    theta = np.radians(tilt_deg)
    rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta), -np.sin(theta)],
        [0.0, np.sin(theta), np.cos(theta)],
    ])
    marbles = []
    for slot in range(count):
        col, row = slot % per_row, (slot // per_row) % per_row
        layer = slot // (per_row * per_row)
        lx, ly = (np.array([col, row]) - (per_row - 1) / 2.0) * pitch
        lz = floor_z + radius + layer * pitch
        # A free joint has to sit on a top-level body, so the marbles cannot be
        # nested inside the tilted pot. Rotate their positions here instead.
        x, y, z = rot @ np.array([lx, ly, lz])
        marbles.append(
            f'<body name="m{slot}" pos="{x:.4f} {y:.4f} {z:.4f}">'
            f'<freejoint/><geom type="sphere" size="{radius}" mass="0.02" '
            f'condim="4" friction="0.4 0.005 0.0001" rgba="0.95 0.85 0.15 1"/></body>'
        )
    walls = [
        f'<geom type="box" pos="0 0 {-half_z + WALL / 2:.4f}" size="{HALF} {HALF} {WALL / 2}"/>',
        f'<geom type="box" pos="0 {-HALF + WALL / 2:.4f} 0" size="{HALF} {WALL / 2} {half_z}"/>',
        f'<geom type="box" pos="0 {HALF - WALL / 2:.4f} 0" size="{HALF} {WALL / 2} {half_z}"/>',
        f'<geom type="box" pos="{-HALF + WALL / 2:.4f} 0 0" size="{WALL / 2} {HALF} {half_z}"/>',
        f'<geom type="box" pos="{HALF - WALL / 2:.4f} 0 0" size="{WALL / 2} {HALF} {half_z}"/>',
    ]
    # The marbles are written in the pot's own frame, so the whole assembly --
    # pot body and the marbles inside it -- is tilted together by nesting them
    # in one rotated body. The marbles' free joints still let them move freely.
    return f"""
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 -1.0" size="5 5 0.1"/>
    <body name="pot" pos="0 0 0" euler="{tilt_deg} 0 0">
      {"".join(walls)}
    </body>
    {"".join(marbles)}
  </worldbody>
</mujoco>
"""


def spill(count: int, radius: float, tilt_deg: float, half_z: float = HALF,
          tray: bool = False) -> int:
    model = mujoco.MjModel.from_xml_string(build(count, radius, tilt_deg, half_z))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(SETTLE_SECONDS / model.opt.timestep)):
        mujoco.mj_step(model, data)

    pot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pot")
    rot = data.xmat[pot].reshape(3, 3)
    pos = data.xpos[pot]
    inside = 0
    for i in range(count):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"m{i}")
        local = rot.T @ (data.xpos[bid] - pos)
        on_footprint = np.all(np.abs(local[:2]) < INNER_XY)
        floor = -half_z + WALL
        # A tray's contents stand PROUD of its lip, so bounding them below the
        # rim -- fine for a deep pot -- counts a full tray as empty.
        held = (floor - radius < local[2] < floor + 6 * radius) if tray else (
            floor < local[2] < half_z
        )
        if on_footprint and held:
            inside += 1
    return inside


def tray_curve(count: int, radius: float, spill_deg: float) -> None:
    """The tray design: lip = r*(1 - cos θ). Confirms the angle it buys."""
    lip = radius * (1.0 - np.cos(np.radians(spill_deg)))
    half_z = (lip + WALL) / 2.0
    print(f"\ntray: {count} marbles r={radius}, lip = r*(1-cos {spill_deg:.0f}) "
          f"= {lip * 1000:.1f} mm  -> predicted spill {spill_deg:.0f} deg")
    for angle in (10, 20, 30, 35, 40, 45, 50, 55, 60, 70):
        n = spill(count, radius, angle, half_z, tray=True)
        print(f"{angle:>5}°  {n:>3}/{count}  {'#' * n}")


def scale_curve(u: float = 0.76) -> None:
    """Does the spill angle survive shrinking the marbles?

    The tray's lip is defined as a MULTIPLE of the marble radius
    (task.yaml: lip_over_radius), so shrinking the marbles shrinks the lip with
    them and the static escape condition

        tan θ = √(2rL − L²) / (r − L)     with L = u·r

    is scale free -- it depends only on u, not on r. Every row below should
    therefore spill at the same angle.

    It is not obvious that they will, for two reasons that both break scale
    invariance in the real thing:

      * gravity sets a timescale. A ball rolls the tray's half-width in
        t ≈ √(2a / (g·sinθ)) regardless of its size, so a SMALLER ball covers
        more of its own radii in that time and arrives at the lip carrying more
        speed relative to the lip it has to hop. That is the same effect that
        moved the calibration from 0.29 r to 0.76 r in the first place.
      * jamming. Nine 18 mm balls span the 12 cm cavity 3.3 diameters across,
        which is squarely in the arching regime; 36 or 81 smaller ones are not.
    """
    print(f"lip = {u} x marble radius in every row; one full layer each\n")
    angles = (10, 20, 30, 35, 40, 45, 50, 55, 60, 70)
    print(f"{'r (mm)':>7} {'n':>4} {'lip (mm)':>9} | " +
          " ".join(f"{a:>4}" for a in angles))
    for radius in (0.018, 0.012, 0.009, 0.006):
        lip = u * radius
        half_z = (lip + WALL) / 2.0
        _, per_row = lattice(radius)
        count = per_row * per_row
        kept = [spill(count, radius, a, half_z, tray=True) for a in angles]
        cells = " ".join(f"{100 * k // count:>4}" for k in kept)
        print(f"{radius * 1000:>7.0f} {count:>4} {lip * 1000:>9.1f} | {cells}")
    print("\ncells are % of the marbles still aboard; the task fails below 75%")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "scale":
        scale_curve()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "tray":
        tray_curve(9, 0.018, 45.0)
        return
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    radius = float(sys.argv[2]) if len(sys.argv) > 2 else 0.012
    pitch, per_row = lattice(radius)
    layers = int(np.ceil(count / (per_row * per_row)))
    fill = radius + (layers - 1) * pitch + radius
    predicted = np.degrees(np.arctan((RIM_Z - FLOOR_Z - fill) / INNER_XY))
    print(f"{count} marbles, r = {radius} m -> {per_row}x{per_row} per layer, "
          f"{layers} layer(s), fill h = {fill:.3f} m")
    print(f"fluid-model prediction: theta_spill = {predicted:.0f} deg\n")
    for angle in ANGLES_DEG:
        n = spill(count, radius, angle)
        print(f"{angle:>5}°  {n:>3}/{count}  {'#' * n}")


if __name__ == "__main__":
    main()
