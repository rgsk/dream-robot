"""Turning frames into a watchable file. Sim-agnostic, like the rest of core.

ROADMAP rule 5: every run writes results.json *and* a video. A video that only
exists because someone ran an ad-hoc script is not an artifact of the project --
it is a screenshot. This module is what makes the video regenerable.

Imports numpy and imageio. Never a simulator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def upscale_nearest(frame: np.ndarray, factor: int) -> np.ndarray:
    """Enlarge by integer row/column repetition.

    Nearest-neighbour on purpose. These panels exist to show what the policy
    actually sees, and any smoothing filter would render a 128x128 observation
    as something better-looking than the thing the network is handed. The
    blockiness is the information.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return frame
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def observation_panel(
    wide: np.ndarray,
    images: Mapping[str, np.ndarray],
    *,
    order: Sequence[str] = ("top", "wrist"),
) -> np.ndarray:
    """A wide view on the left, the policy's own cameras stacked on the right.

    The split is the point: everything on the left is for a human, everything on
    the right is the complete input a policy receives. Reading them side by side
    is how you notice that a camera has the gripper out of frame during the part
    of the episode that matters.

    Camera panels are scaled to share the wide view's height, so the right-hand
    column is honest about relative resolution rather than padded to look equal.
    """
    wide = np.asarray(wide)
    if wide.dtype != np.uint8 or wide.ndim != 3 or wide.shape[2] != 3:
        raise ValueError(f"wide frame must be uint8 (H, W, 3), got {wide.dtype} {wide.shape}")

    present = [name for name in order if name in images]
    if not present:
        raise ValueError(f"none of {list(order)} in images {sorted(images)}")

    target_h = wide.shape[0] // len(present)
    column = []
    for name in present:
        img = np.asarray(images[name])
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"camera {name!r} must be uint8 (H, W, 3), got {img.shape}")
        if target_h % img.shape[0]:
            raise ValueError(
                f"camera {name!r} is {img.shape[0]}px tall, which does not divide the "
                f"{target_h}px panel height. Integer scaling only -- see upscale_nearest."
            )
        column.append(upscale_nearest(img, target_h // img.shape[0]))

    stacked = np.concatenate(column, axis=0)
    if stacked.shape[0] != wide.shape[0]:  # odd heights leave a row or two over
        stacked = stacked[: wide.shape[0]]
    return np.concatenate([wide, stacked], axis=1)


def write_video(frames: Sequence[np.ndarray], path: Path, *, fps: float) -> Path:
    """Encode frames to mp4 at the environment's own control rate.

    ``fps`` must be the task's control_hz. A video written at any other rate
    shows the robot moving at a speed it never moved, which quietly
    misrepresents the cycle time that the results matrix reports.
    """
    if not frames:
        raise ValueError("no frames to write")
    stack = np.stack([np.asarray(f) for f in frames])
    if stack.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {stack.dtype}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, stack, fps=fps, codec="libx264")
    return path


def camera_strip(
    images: Mapping[str, np.ndarray],
    *,
    order: Sequence[str] = ("top", "wrist"),
    scale: int = 3,
) -> np.ndarray:
    """The policy's cameras side by side, and nothing else.

    ``observation_panel`` puts a wide human-legible view next to these. This one
    deliberately cannot: it is what a frame read back out of a recorded dataset
    contains, and a dataset holds no render camera. Reconstructing a video from
    a dataset with this function shows the actual training input at its actual
    resolution, which is the only way to notice that the wrist camera has been
    staring at a finger for the whole episode.
    """
    present = [name for name in order if name in images]
    if not present:
        raise ValueError(f"none of {list(order)} in images {sorted(images)}")
    return np.concatenate(
        [upscale_nearest(np.asarray(images[name]), scale) for name in present], axis=1
    )


class VideoWriter:
    """Append frames to an mp4 as they are produced, holding none of them.

    ``write_video`` wants every frame in a list, which is fine for a demo and
    wrong for evaluation: a failed episode runs the full horizon, and a 600-step
    panel at 256x384 is ~177 MB. Deciding *after* an episode whether it was
    worth filming means buffering one episode at most, then streaming it out.

    Used as a context manager. Creates the file lazily, on the first frame, so a
    writer that is never fed leaves no empty video behind -- which is what makes
    "up to two failures" produce no failures file on a policy that never fails.
    """

    def __init__(self, path: Path, *, fps: float):
        self.path = Path(path)
        self._fps = fps
        self._handle = None
        self.frames = 0

    def __enter__(self) -> VideoWriter:
        return self

    def append(self, frames: Sequence[np.ndarray]) -> None:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                raise ValueError(f"frames must be uint8, got {arr.dtype}")
            if self._handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = iio.imopen(self.path, "w", plugin="pyav")
                self._handle.init_video_stream("libx264", fps=self._fps)
            self._handle.write_frame(arr)
            self.frames += 1

    def close(self) -> Path | None:
        if self._handle is None:
            return None
        self._handle.close()
        self._handle = None
        return self.path

    def __exit__(self, *exc) -> None:
        self.close()
