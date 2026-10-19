"""Pixels that match -> 3D points. The step from an image to places the arm can reach.

Three stages; the first is per tool, the rest are shared:

    a mask of matching pixels           colour_mask(rgb, "blue"); SAM3 (phrase) later
    split mask into blobs               one blob per object          (never averaged across)
    backproject + top face -> point     where each blob is in 3D     (geometry, stays)

**Several matches are reported, never resolved.** Two blue cubes used to be one
mask and one median -- a point on whichever cube covered more pixels, or between
them, with ``found`` just as confident as for one cube (notes.md, step 3).
Perception says what it saw; choosing between candidates is the planner's job,
since it has the words and the judgement.

**Each mask source takes only the input it understands.** The colour mask takes
a colour, not a phrase: it knows four hues and nothing about shape or position,
so there is no sentence for it to misread. SAM3 will take a phrase, because it
can read one. ``find`` takes the mask and never sees either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from dream_robot.core.camera import RGBD, backproject

# Hue bands in degrees. Hue survives shading -- a cube's lit top and shadowed
# side differ in brightness, not in hue -- which is why this works at all.
HUES: dict[str, tuple[tuple[float, float], ...]] = {
    "red": ((0, 15), (345, 360)),
    "yellow": ((40, 70),),
    "green": ((90, 160),),
    "blue": ((200, 260),),
}
MIN_SATURATION = 0.5    # the table is white and the bins grey: both ~0
MIN_VALUE = 0.15        # pure shadow has no reliable hue
MIN_PIXELS = 20         # a blob smaller than this is noise, not an object

# Pixels within this of the highest visible point count as the top face. A
# camera above and in front also sees a side face; its points pull the centre
# towards the camera, so they are dropped.
TOP_FACE_BAND = 0.005   # metres


COLOURS: tuple[str, ...] = tuple(HUES)


def colour_mask(rgb: np.ndarray, colour: str) -> np.ndarray:
    """(H, W) bool: the pixels of ``colour``, one of COLOURS."""
    h, s, v = _hsv(rgb)
    in_band = np.zeros(h.shape, dtype=bool)
    for lo, hi in HUES[colour]:
        in_band |= (h >= lo) & (h < hi)
    return in_band & (s >= MIN_SATURATION) & (v >= MIN_VALUE)


@dataclass(frozen=True)
class Found:
    point: np.ndarray          # world xyz, centre of the visible top face
    pixels: int                # how much of the image this object covers
    mask: np.ndarray           # (H, W) bool, this object only


def find(frame: RGBD, mask: np.ndarray) -> list[Found]:
    """One object per connected blob of ``mask``, largest in the image first. Empty if none.

    Known limit: two matching objects that touch in the image are one blob.
    """
    labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
    found = []
    for i in range(1, n + 1):
        mask = labels == i
        if mask.sum() < MIN_PIXELS:
            continue
        points = backproject(frame, mask)
        top = points[points[:, 2] >= np.percentile(points[:, 2], 95) - TOP_FACE_BAND]
        found.append(Found(point=np.median(top, axis=0), pixels=int(mask.sum()), mask=mask))
    return sorted(found, key=lambda f: -f.pixels)


def overlay(frame: RGBD, found: list[Found]) -> np.ndarray:
    """The frame with each match tinted, a cross on its point, and -- when there is
    more than one -- its number, matching its position in the list (1-based)."""
    img = frame.rgb.copy()
    for f in found:
        img[f.mask] = (0.5 * img[f.mask] + [0, 127, 127]).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for i, f in enumerate(found, start=1):
        u, v = _project(frame, f.point)
        draw.line([(u - 6, v), (u + 6, v)], fill=(255, 0, 255))
        draw.line([(u, v - 6), (u, v + 6)], fill=(255, 0, 255))
        if len(found) > 1:
            draw.text((u + 8, v - 14), str(i), fill=(255, 0, 255))
    return np.asarray(pil)


def _project(frame: RGBD, point: np.ndarray) -> tuple[int, int]:
    world_to_cam = np.linalg.inv(frame.cam_to_world)
    x, y, z = world_to_cam[:3, :3] @ point + world_to_cam[:3, 3]
    return int(frame.K[0, 0] * x / z + frame.K[0, 2]), int(frame.K[1, 1] * y / z + frame.K[1, 2])


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hue in degrees [0, 360), saturation and value in [0, 1]."""
    x = rgb.astype(float) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(axis=-1), x.min(axis=-1)
    c = mx - mn
    safe = np.where(c == 0, 1.0, c)
    h = np.select(
        [mx == r, mx == g],
        [((g - b) / safe) % 6, (b - r) / safe + 2],
        (r - g) / safe + 4,
    ) * 60.0
    s = np.where(mx == 0, 0.0, c / np.where(mx == 0, 1.0, mx))
    return np.where(c == 0, 0.0, h), s, mx
