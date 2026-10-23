"""Colour names for pixels, and projecting a world point into a camera image.

What is left of the colour-threshold finder (find_by_colour, retired 2026-10-20):
find_objects uses ``colour_mask`` to name an object's colour, and ``_project`` to
draw on the camera image. Hue survives shading, which is why a threshold names
flat-coloured objects reliably; textured ones come out "mixed".
"""

from __future__ import annotations

import numpy as np

from dream_robot.core.camera import RGBD

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



COLOURS: tuple[str, ...] = tuple(HUES)


def colour_mask(rgb: np.ndarray, colour: str) -> np.ndarray:
    """(H, W) bool: the pixels of ``colour``, one of COLOURS."""
    h, s, v = _hsv(rgb)
    in_band = np.zeros(h.shape, dtype=bool)
    for lo, hi in HUES[colour]:
        in_band |= (h >= lo) & (h < hi)
    return in_band & (s >= MIN_SATURATION) & (v >= MIN_VALUE)


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
