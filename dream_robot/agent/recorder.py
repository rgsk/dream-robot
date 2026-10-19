"""Camera clips of each skill: what a person watching the robot would have seen.

A driver judges outcomes by looking, never from the sim's state. One image after a
skill shows the end, not the moment something went wrong -- a box tipped by the
arm on the way, a bar that slid onto a rim. So every skill records its cameras,
and the driver gets a grid of frames now and can ask for more later
(``review_clip``): a time window, a camera, a region to enlarge. That is how an
image-only model watches video.

Frames are kept as JPEG bytes (a 5 s motion is ~50 frames x 3 cameras; raw that is
~100 MB) for the current episode only. When ``out_dir`` is set each clip is also
written as one MP4 per camera, for a human.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CAMERAS = ("scene", "top", "wrist")
FRAME_HW = (512, 512)     # sharp enough to see a 1-2 cm lift at the bin
STRIDE = 3                # record every 3rd control step: 10 fps at 30 Hz


@dataclass
class Clip:
    id: str
    skill: str
    fps: float
    times: list[float] = field(default_factory=list)              # seconds from clip start
    jpegs: dict[str, list[bytes]] = field(default_factory=lambda: {c: [] for c in CAMERAS})
    videos: dict[str, str] = field(default_factory=dict)

    @property
    def seconds(self) -> float:
        return self.times[-1] if self.times else 0.0

    def frame(self, camera: str, i: int) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(self.jpegs[camera][i])).convert("RGB"))

    def summary(self) -> dict:
        return {"clip": self.id, "skill": self.skill, "seconds": round(self.seconds, 2),
                "frames": len(self.times), "cameras": list(CAMERAS),
                **({"videos": self.videos} if self.videos else {})}


class Recorder:
    """Records the cameras while a skill runs. ``start`` -> ``tick`` per step -> ``stop``."""

    def __init__(self, grab, control_hz: float, out_dir: str | None = None):
        self._grab = grab                    # (camera, hw) -> RGB frame
        self._hz = control_hz
        self._out = Path(out_dir) if out_dir else None
        self.clips: dict[str, Clip] = {}
        self._clip: Clip | None = None
        self._step = 0
        self._episode = "warmup"         # before the first reset

    def new_episode(self, name: str) -> None:
        self.clips.clear()
        self._episode = name

    def start(self, skill: str) -> None:
        cid = f"{self._episode}-{len(self.clips) + 1}-{skill}"
        self._clip = Clip(cid, skill, fps=self._hz / STRIDE)
        self._step = 0
        self._capture()

    def tick(self) -> None:
        if self._clip is None:
            return
        self._step += 1
        if self._step % STRIDE == 0:
            self._capture()

    def stop(self) -> Clip:
        clip, self._clip = self._clip, None
        if self._step % STRIDE:                       # always end on the final state
            self._clip = clip
            self._capture()
            self._clip = None
        if self._out is not None:
            self._out.mkdir(parents=True, exist_ok=True)
            for cam in CAMERAS:
                path = self._out / f"{clip.id}-{cam}.mp4"
                frames = np.stack([clip.frame(cam, i) for i in range(len(clip.times))])
                iio.imwrite(path, frames, fps=clip.fps, codec="libx264", macro_block_size=8)
                clip.videos[cam] = str(path)
        self.clips[clip.id] = clip
        return clip

    def _capture(self) -> None:
        clip = self._clip
        clip.times.append(self._step / self._hz)
        for cam in CAMERAS:
            buf = io.BytesIO()
            Image.fromarray(self._grab(cam, FRAME_HW)).save(buf, format="JPEG", quality=90)
            clip.jpegs[cam].append(buf.getvalue())


def grid(clip: Clip, camera: str, *, start: float = 0.0, end: float | None = None,
         n: int = 8, crop: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """``n`` frames evenly spread over [start, end] seconds (the last frame always
    included), each stamped with its time, 4 to a row. ``crop`` = (x0, y0, x1, y1) as
    fractions of the frame: that region, enlarged to fill each tile."""
    end = clip.seconds if end is None else end
    idx = [i for i, t in enumerate(clip.times) if start - 1e-6 <= t <= end + 1e-6]
    if not idx:
        raise ValueError(f"no frames between {start} and {end} s (clip is {clip.seconds:.2f} s)")
    spread = max(n - 1, 1)
    picks = sorted({idx[round(k * (len(idx) - 1) / spread)] for k in range(min(n, len(idx)))})
    tile = 512 if len(picks) <= 2 else 384 if len(picks) <= 4 else 256
    font = ImageFont.load_default(size=16)
    tiles = []
    for i in picks:
        img = Image.fromarray(clip.frame(camera, i))
        if crop is not None:
            w, h = img.size
            x0, y0, x1, y1 = crop
            img = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
        img = img.resize((tile, tile))
        draw = ImageDraw.Draw(img)
        label = f"{camera} t={clip.times[i]:.1f}s"
        box = draw.textbbox((4, 4), label, font=font)
        draw.rectangle([box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2], fill=(0, 0, 0))
        draw.text((4, 4), label, fill=(255, 255, 255), font=font)
        tiles.append(np.asarray(img))
    cols = min(4, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + cols], axis=1) for r in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=0)
