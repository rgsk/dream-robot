from collections.abc import Mapping, Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def upscale_nearest(frame: np.ndarray, factor: int) -> np.ndarray:
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
    if wide.dtype != np.uint8 or wide.ndim != 3 or wide.shape[2] != 3:
        raise ValueError(f"wide frame must be uint8 (H, W, 3), got {wide.dtype} {wide.shape}")
    present = [name for name in order if name in images]
    if not present:
        raise ValueError(f"none of {list(order)} in images {sorted(images)}")
    if wide.shape[0] % len(present):
        raise ValueError(
            f"render height {wide.shape[0]}px is not divisible by {len(present)} cameras "
            f"{present}; each panel would be {wide.shape[0] / len(present)}px. "
            f"Set render.height in task.yaml to a multiple of {len(present)}."
        )

    target_h = wide.shape[0] // len(present)
    column = []
    for name in present:
        image = images[name]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"camera {name!r} must be uint8 (H, W, 3), got {image.dtype} {image.shape}"
            )
        if target_h % image.shape[0]:
            raise ValueError(
                f"camera {name} is {image.shape[0]}px tall, which does not divide the "
                f"{target_h}px panel height. Integer scaling only -- see upscale_nearest."
            )
        column.append(upscale_nearest(image, target_h // image.shape[0]))

    stacked = np.concatenate(column, axis=0)
    panel = np.concatenate([wide, stacked], axis=1)
    return panel

def write_video(frames: Sequence[np.ndarray], path: Path, *, fps: float) -> Path:
    if not frames:
        raise ValueError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack(frames)
    if stack.dtype != np.uint8:
        raise ValueError(f"frames must be uint8, got {stack.dtype}")
    iio.imwrite(path, stack, fps=fps, codec="libx264")
    return path
