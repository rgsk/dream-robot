"""Aim by pixel: the driver picks pixels in a camera image, depth turns them into 3D points.

The general alternative to ``tabletop.find_objects``: no assumption about a table,
objects or containers -- just "what surface is under this pixel". Works on any scene
a calibrated depth camera sees. Camera-only: the same image, depth and calibration a
real RGB-D camera gives.

The image a driver reads coordinates from is ``grid_view``: the frame with a labelled
pixel grid. ``mark`` draws the queried pixels back on it, so a miss is visible.

``object_at`` is the middle ground: the driver picks *which* thing by one pixel on it,
and code measures it, with ``find_objects``' own description (centre, size, grasp...).
What it assumes instead of a known table:

    support surface    the most common height among points within NEIGHBOURHOOD of the
                       pixel's point and not above it -- a table, a counter, a shelf
    the thing          points above that surface, near the pixel, clustered on an xy grid
                       as find_objects does; the cluster under the pixel
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scipy import ndimage

from dream_robot.agent import tabletop
from dream_robot.core.camera import RGBD, backproject

GRID_STEP = 64           # pixels between labelled grid lines
NEIGHBOURHOOD = 0.25     # metres in xy around the pointed point: the support is found here


def points(frame: RGBD, pixels: list[list[float]]) -> list[dict]:
    """World point under each (u, v) pixel: u right, v down, from the top-left corner."""
    h, w = frame.depth.shape
    out = []
    for u, v in pixels:
        if not (0 <= u < w and 0 <= v < h):
            out.append({"pixel": [u, v], "point": None, "error": f"outside the {w}x{h} image"})
            continue
        z = float(frame.depth[int(v), int(u)])
        fx, fy, cx, cy = frame.K[0, 0], frame.K[1, 1], frame.K[0, 2], frame.K[1, 2]
        cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])
        p = frame.cam_to_world[:3, :3] @ cam + frame.cam_to_world[:3, 3]
        out.append({"pixel": [u, v], "point": [round(float(c), 4) for c in p]})
    return out


def grid_view(frame: RGBD) -> Image.Image:
    """The frame with thin grid lines every GRID_STEP pixels, labelled along two edges."""
    img = Image.fromarray(frame.rgb).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default(size=12)
    w, h = img.size
    # Dark lines: the table is white. Labels on black boxes: readable on anything.
    for x in range(0, w, GRID_STEP):
        draw.line([(x, 0), (x, h)], fill=(0, 0, 0, 110), width=1)
    for y in range(0, h, GRID_STEP):
        draw.line([(0, y), (w, y)], fill=(0, 0, 0, 110), width=1)
    for x in range(GRID_STEP, w, GRID_STEP):
        _label(draw, (x + 2, 1), str(x), font)
    for y in range(GRID_STEP, h, GRID_STEP):
        _label(draw, (2, y + 1), str(y), font)
    return Image.alpha_composite(img, layer).convert("RGB")


def mark(img: Image.Image, pixels: list[list[float]]) -> np.ndarray:
    """A numbered crosshair at each queried pixel."""
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=13)
    for i, (u, v) in enumerate(pixels, start=1):
        draw.line([(u - 7, v), (u + 7, v)], fill=(255, 0, 255), width=2)
        draw.line([(u, v - 7), (u, v + 7)], fill=(255, 0, 255), width=2)
        draw.text((u + 6, v + 4), str(i), fill=(255, 0, 255), font=font)
    return np.asarray(img)


def _label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> None:
    box = draw.textbbox(xy, text, font=font)
    draw.rectangle([box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1], fill=(0, 0, 0, 200))
    draw.text(xy, text, fill=(255, 255, 0, 255), font=font)


def object_at(frame: RGBD, pixel: list[float], robot: np.ndarray | None = None
              ) -> tuple[tabletop.Thing | None, float, np.ndarray]:
    """(the thing under ``pixel``, or None if the pixel is on the support itself;
    the support's height; the pixel's own point). ``robot``: arm pixels, dropped."""
    h, w = frame.depth.shape
    u, v = int(pixel[0]), int(pixel[1])
    pts = backproject(frame, np.ones((h, w), dtype=bool))
    seed = pts[v * w + u]
    near = np.all(np.abs(pts[:, :2] - seed[:2]) < NEIGHBOURHOOD, axis=1)
    if robot is not None:
        near &= ~robot.reshape(-1)
    below = near & (pts[:, 2] <= seed[2] + 0.002)
    support_z = tabletop._table_height(pts[below, 2])
    if seed[2] < support_z + tabletop.ABOVE_TABLE:
        return None, support_z, seed

    fg = (near & (pts[:, 2] > support_z + tabletop.ABOVE_TABLE)
          & (pts[:, 2] < support_z + tabletop.MAX_HEIGHT))
    origin = seed[:2] - NEIGHBOURHOOD
    ij = np.floor((pts[:, :2] - origin) / tabletop.CELL).astype(int)
    n_cells = int(np.ceil(2 * NEIGHBOURHOOD / tabletop.CELL)) + 1
    grid = np.zeros((n_cells, n_cells), dtype=bool)
    grid[ij[fg, 0], ij[fg, 1]] = True
    labels, _ = ndimage.label(ndimage.binary_closing(grid), structure=np.ones((3, 3)))
    k = labels[tuple(ij[v * w + u])]
    if k == 0:                       # the pixel's own cell fell out (e.g. an edge pixel)
        return None, support_z, seed
    sel = np.zeros(len(pts), dtype=bool)
    sel[fg] = labels[ij[fg, 0], ij[fg, 1]] == k
    if sel.sum() < tabletop.MIN_POINTS:
        return None, support_z, seed
    thing = tabletop._describe(frame, pts[sel], sel.reshape(h, w), labels == k, support_z,
                               tuple(origin))
    return thing, support_z, seed
