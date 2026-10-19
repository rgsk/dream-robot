"""A calibrated depth camera frame, and pixels -> 3D points.

What a real RGB-D camera gives you: an image, a depth per pixel, the lens
(intrinsics K) and where the camera sits (extrinsics). None of it is ground
truth about objects -- a real robot has all four -- so the executor may use it.
It never enters a dataset: ``observation.*`` stays images + joint state.

Numpy only, like the rest of core.

Conventions, OpenCV's: pixel (u, v) with u right and v **down**, row 0 at the
top of the image; camera frame x right, y down, z forward along the lens.

    pixel -> camera:   x = (u - cx) * z / f,   y = (v - cy) * z / f,   z = depth
    camera -> world:   p_world = R · p_cam + t        (cam_to_world = [R | t])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBD:
    rgb: np.ndarray            # uint8 (H, W, 3), row 0 = top of the image
    depth: np.ndarray          # float (H, W), metres along the optical axis
    K: np.ndarray              # (3, 3) intrinsics
    cam_to_world: np.ndarray   # (4, 4), OpenCV camera frame -> world


def backproject(frame: RGBD, mask: np.ndarray) -> np.ndarray:
    """World points (N, 3) for the pixels where ``mask`` is True."""
    v, u = np.nonzero(mask)
    z = frame.depth[v, u]
    fx, fy = frame.K[0, 0], frame.K[1, 1]
    cx, cy = frame.K[0, 2], frame.K[1, 2]
    # +0.5: a pixel's centre, not its corner, is where its ray passes through.
    cam = np.stack([(u + 0.5 - cx) * z / fx, (v + 0.5 - cy) * z / fy, z], axis=1)
    R, t = frame.cam_to_world[:3, :3], frame.cam_to_world[:3, 3]
    return cam @ R.T + t
