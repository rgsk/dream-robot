"""Everything on the table, from depth: objects to pick and containers to place into.

A colour threshold asks "where is the red thing?" and needs a colour to ask. This
asks nothing: whatever rises above the table is something, and each something
is described well enough for a planner to name it ("#2 is the yellow bar") and
for the arm to act on it.

    depth -> world points          the camera is calibrated (core/camera.py)
    table plane                    the most common height in view, not a constant
    above it, on the table         foreground
    cluster on an xy grid          one cluster per thing; tall things that hide
                                   their neighbour in the image still separate in xy
    describe each cluster          footprint, height, shape, colour
    object or container?           a container's footprint is a ring with an empty
                                   middle; an object's is filled
    object -> grasp                where to close the fingers, along which axis
    container -> drop point        the centre of its opening

**Clusters are separated by clear table, so touching things are one cluster.**
T7 spaces objects 4 cm apart. Clutter needs a segmenter (SAM3 or similar) as a
different tool, not a smarter threshold here.

**Nothing here knows the scene's ground truth or its fixtures.** Two things are
given, as on a real cell: the table's extent (the workspace) and a mask of the
robot itself (its model and joint angles are known, so it is filtered out).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from dream_robot.agent.perception import COLOURS, _project, colour_mask
from dream_robot.core.camera import RGBD, backproject

CELL = 0.004            # xy grid, metres. Fine enough that a 4 cm gap is 10 cells
ABOVE_TABLE = 0.004     # a point this far over the table plane is foreground
MAX_HEIGHT = 0.25       # above this is the arm, not something on the table
MIN_POINTS = 30         # fewer is a speck of noise
TOP_BAND = 0.004        # points this close to the highest one form the top surface

# An object fills its footprint; a bin's walls fill a thin ring of it. Measured
# on T7, 30 seeds: objects 0.96-1.27, bins 0.32-0.36 (notes.md, step 7).
CONTAINER_FILL = 0.5

# The open Panda gripper spans 8 cm. Wider than this across the narrow side
# and the fingers close on air or land on top: flagged, not refused.
MAX_GRASP_WIDTH = 0.075
# Fingers go down this far below the top at most: half-height for short
# things (a ball's equator), no deeper for tall ones.
MAX_GRASP_DEPTH = 0.03


@dataclass(frozen=True)
class Thing:
    kind: str                  # "object" | "container"
    centre: np.ndarray         # xyz: footprint centre in xy, top in z
    length: float              # footprint, long side, metres
    width: float               # footprint, short side
    axis_yaw: float            # direction of the long side, radians in [-pi/2, pi/2)
    height: float              # top above the table
    footprint: str             # "round" | "rectangular"
    top: str                   # "flat" | "domed"
    colour: str                # a COLOURS name, "white" / "grey" / "black", or "mixed"
    rgb: tuple[int, int, int]  # median colour of the top surface
    pixels: np.ndarray         # (H, W) bool, this thing in the image
    table_z: float
    filled_area: float         # xy area the thing's own points cover

    @property
    def opening(self) -> tuple[float, float]:
        """Containers: inside length and width. The walls are a ring of area
        ~ 2 t (L + W) for wall thickness t, so t comes from the filled area."""
        t = self.filled_area / (2 * (self.length + self.width))
        return self.length - 2 * t, self.width - 2 * t

    @property
    def grasp(self) -> dict | None:
        """Objects only: where the finger pads close, and along which world direction."""
        if self.kind != "object":
            return None
        depth = min(self.height / 2, MAX_GRASP_DEPTH)
        point = self.centre - [0.0, 0.0, depth]
        close_along = _wrap_half_turn(self.axis_yaw + np.pi / 2)  # across the narrow side
        return {
            "point": _r(point),
            "yaw_deg": round(float(np.rad2deg(close_along)), 1),
            "above_table": round(float(point[2] - self.table_z), 4),
            "width": round(self.width, 4),
            "too_wide": bool(self.width > MAX_GRASP_WIDTH),
        }

    def as_dict(self, id_: int) -> dict:
        d = {
            "id": id_,
            "kind": self.kind,
            "centre": _r(self.centre),
            "size": {"length": round(self.length, 3), "width": round(self.width, 3),
                     "height": round(self.height, 3)},
            "footprint": self.footprint,
            "top": self.top,
            "colour": self.colour,
            "rgb": list(self.rgb),
        }
        if self.kind == "object":
            d["grasp"] = self.grasp
        else:
            # The centre of the opening, at rim height. the driver aims a place at this point.
            d["drop_point"] = _r(self.centre)
            length, width = self.opening
            d["opening"] = {
                "length": round(length, 3), "width": round(width, 3),
                "diagonal": round(float(np.hypot(length, width)), 3),
                # Direction of the long side (either side, for a square). Its diagonals
                # lie at about +-atan(width / length) from it -- 45 deg for a square.
                "axis_yaw_deg": round(float(np.rad2deg(self.axis_yaw)), 1),
            }
        return d


def find_objects(
    frame: RGBD, table_xy: tuple[float, float, float, float], robot: np.ndarray | None = None
) -> list[Thing]:
    """Every object and container on the table, left to right as seen in the image.

    ``table_xy`` = (x_min, x_max, y_min, y_max): the workspace, world frame.
    ``robot``: (H, W) bool, pixels of the arm itself, dropped (self-filtering).
    """
    pts = backproject(frame, np.ones(frame.depth.shape, dtype=bool))
    x0, x1, y0, y1 = table_xy
    on_table = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
    table_z = _table_height(pts[on_table, 2])
    fg = on_table & (pts[:, 2] > table_z + ABOVE_TABLE) & (pts[:, 2] < table_z + MAX_HEIGHT)
    if robot is not None:
        fg &= ~robot.reshape(-1)

    # Occupancy grid over the table, closed once so one thing's sparse points join up.
    ij = np.floor((pts[:, :2] - [x0, y0]) / CELL).astype(int)
    shape = (int(np.ceil((x1 - x0) / CELL)), int(np.ceil((y1 - y0) / CELL)))
    grid = np.zeros(shape, dtype=bool)
    grid[ij[fg, 0], ij[fg, 1]] = True
    labels, n = ndimage.label(ndimage.binary_closing(grid), structure=np.ones((3, 3)))

    point_label = np.zeros(len(pts), dtype=int)
    point_label[fg] = labels[ij[fg, 0], ij[fg, 1]]
    h, w = frame.depth.shape
    things = []
    for k in range(1, n + 1):
        sel = point_label == k
        if sel.sum() < MIN_POINTS:
            continue
        thing = _describe(frame, pts[sel], sel.reshape(h, w), labels == k, table_z, (x0, y0))
        if thing is not None:
            things.append(thing)
    return sorted(things, key=lambda t: _project(frame, t.centre)[0])


def overlay(frame: RGBD, things: list[Thing]) -> np.ndarray:
    """Each thing outlined (objects magenta, containers cyan) and numbered by its id in
    a dark box. Outlines, not tints: a planner must still see the true colours.
    Objects also get a yellow bar along the grasp's closing direction."""
    img = frame.rgb.copy()
    edge = {"object": (255, 0, 255), "container": (0, 220, 255)}
    for t in things:
        rim = t.pixels & ~ndimage.binary_erosion(t.pixels)
        img[rim] = edge[t.kind]
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default(size=14)
    for i, t in enumerate(things, start=1):
        if t.kind == "object":
            g = np.asarray(t.grasp["point"])
            a = np.deg2rad(t.grasp["yaw_deg"])
            half = np.array([np.cos(a), np.sin(a), 0.0]) * max(t.width / 2 + 0.01, 0.02)
            (ua, va), (ub, vb) = _project(frame, g - half), _project(frame, g + half)
            draw.line([(ua, va), (ub, vb)], fill=(255, 255, 0), width=2)
        u, v = _project(frame, t.centre)
        box = draw.textbbox((u + 8, v - 20), str(i), font=font)
        draw.rectangle([box[0] - 2, box[1] - 1, box[2] + 2, box[3] + 1],
                       fill=(0, 0, 0), outline=edge[t.kind])
        draw.text((u + 8, v - 20), str(i), fill=(255, 255, 255), font=font)
    return np.asarray(pil)


# --- internals ---------------------------------------------------------------


def _table_height(z: np.ndarray) -> float:
    """The most common height in view, to 2 mm, refined by the median near it."""
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.002, 0.002))
    mode = edges[hist.argmax()]
    return float(np.median(z[np.abs(z - mode) < 0.002]))


def _describe(frame, pts, pixels, cells, table_z, origin) -> Thing | None:
    xy = pts[:, :2]
    rect = _min_area_rect(xy)
    if rect is None:
        return None
    centre_xy, length, width, axis = rect
    top_z = float(np.percentile(pts[:, 2], 98))
    height = top_z - table_z

    hull_area = _hull_area(xy)
    occupied = cells.sum() * CELL**2
    kind = "container" if occupied / max(hull_area, 1e-9) < CONTAINER_FILL else "object"

    # Round footprints fill pi/4 of their bounding rectangle, rectangles all of it.
    footprint = "round" if hull_area / max(length * width, 1e-9) < 0.88 else "rectangular"
    top_pts = pts[:, 2] >= top_z - TOP_BAND
    top = "flat" if _hull_area(xy[top_pts]) / max(hull_area, 1e-9) > 0.5 else "domed"

    top_pixels = np.zeros_like(pixels)
    top_pixels[pixels] = top_pts
    rgb = frame.rgb[top_pixels if top_pixels.sum() >= 10 else pixels]
    colour = _colour_name(rgb)
    return Thing(
        kind=kind,
        centre=np.array([*centre_xy, top_z]),
        length=length,
        width=width,
        axis_yaw=axis,
        height=height,
        footprint=footprint,
        top=top,
        colour=colour,
        rgb=tuple(int(c) for c in np.median(rgb, axis=0)),
        pixels=pixels,
        table_z=table_z,
        filled_area=float(occupied),
    )


def _colour_name(rgb: np.ndarray) -> str:
    """A palette hue if most pixels have it; an achromatic name if most have no hue."""
    for name in COLOURS:
        if colour_mask(rgb[None], name).mean() > 0.6:
            return name
    x = rgb.astype(float) / 255.0
    mx, mn = x.max(axis=1), x.min(axis=1)
    if np.mean(mx - mn < 0.15) > 0.6:               # grey-ish: no hue to speak of
        v = float(np.median(mx))
        return "white" if v > 0.8 else "black" if v < 0.2 else "grey"
    return "mixed"


def _min_area_rect(xy: np.ndarray):
    """(centre, length, width, long-axis yaw) of the smallest rectangle around xy.

    Rotating calipers: the best rectangle has a side along some hull edge, so try
    each edge's direction and keep the smallest area.
    """
    try:
        hull = xy[ConvexHull(xy).vertices]
    except QhullError:
        return None
    best = None
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    for a in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (np.pi / 2), 6)):
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])  # world -> rect frame
        p = hull @ R.T
        lo, hi = p.min(axis=0), p.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, a, lo, hi, R)
    _, a, lo, hi, R = best
    centre = ((lo + hi) / 2) @ R               # back to world
    (dx, dy) = hi - lo
    axis = a if dx >= dy else a + np.pi / 2
    return centre, float(max(dx, dy)), float(min(dx, dy)), _wrap_half_turn(axis)


def _hull_area(xy: np.ndarray) -> float:
    try:
        return float(ConvexHull(xy).volume)    # in 2D, "volume" is the area
    except (QhullError, ValueError):
        return 0.0


def _wrap_half_turn(a: float) -> float:
    """An axis, not a direction: a and a + pi are the same. Into [-pi/2, pi/2)."""
    return float((a + np.pi / 2) % np.pi - np.pi / 2)


def _r(v) -> list[float]:
    return [round(float(c), 4) for c in v]
