"""(frames at t, frames at t + k, state at t, actions t .. t + k - 1) samples.

ACT's chunk dataset with one more read per sample: the frame the chunk leads to.
The episode split, padding and stats are ACT's, so the IDM trains on the same
frames and holds out the same episodes as every other policy on T5.

Near an episode's end, t + k runs past the last frame: the future is the last
frame and the missing actions are padded (masked out of the loss), so the
sample still says "get from here to how the episode ended".

**Frames are decoded once, not every epoch.** LeRobot stores them as AV1 video;
decoding four frames per sample kept four CPU workers at 100% and made a 1M
parameter model take ~30 s an epoch. ``decoded_frames`` writes every frame of
every camera to ``<root>/decoded/<camera>.npy`` (uint8, ~1.6 GB per camera on
T5) the first time, and memory-maps it after.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dream_robot.core.dataset import image_feature, open_dataset
from dream_robot.policies.act.data import ChunkDataset
from dream_robot.policies.act.data import compute_stats as compute_stats  # re-export


class _Frames(Dataset):
    def __init__(self, repo_id, root, camera):
        self._dataset, self._key = open_dataset(repo_id, root), image_feature(camera)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, i):
        return (self._dataset[i][self._key] * 255).round().to(torch.uint8)


def decoded_frames(repo_id: str, root: Path, cameras, *, workers: int = 8) -> dict[str, Path]:
    """``{camera: path}`` of uint8 arrays (frames, 3, H, W), by table row. Built once."""
    out = {}
    for camera in cameras:
        path = Path(root) / "decoded" / f"{camera}.npy"
        if not path.exists():
            frames = _Frames(repo_id, root, camera)
            print(f"decoding {len(frames)} {camera} frames once -> {path}", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".partial.npy")
            first = frames[0]
            arr = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.uint8, shape=(len(frames), *first.shape)
            )
            row = 0
            for batch in DataLoader(frames, batch_size=256, num_workers=workers):
                arr[row : row + len(batch)] = batch.numpy()
                row += len(batch)
            arr.flush()
            del arr
            tmp.rename(path)
        out[camera] = path
    return out


class PairDataset(ChunkDataset):
    """``chunk`` is k: the future frame is k rows on, and k actions lead to it."""

    def __init__(self, repo_id, root, cameras, episodes, chunk):
        super().__init__(repo_id, root, cameras, episodes, chunk)
        self._paths = decoded_frames(repo_id, root, cameras)
        self._frames: dict | None = None

    def __getstate__(self):
        # A memory map pickles as a full copy of its data: 3.3 GB into every
        # loader worker. Send the paths; each process maps the files itself.
        state = self.__dict__.copy()
        state["_frames"] = None
        return state

    def _images(self, row: int) -> dict[str, torch.Tensor]:
        if self._frames is None:
            self._frames = {c: np.load(p, mmap_mode="r") for c, p in self._paths.items()}
        return {
            name: torch.from_numpy(np.array(self._frames[name][row])).float() / 255.0
            for name in self._cameras
        }

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        actions, pad = self.chunk_at(row)
        ahead = min(row + self._chunk, int(self._last[row]))
        return (
            self._images(row),
            torch.from_numpy(self.states[row]),
            torch.from_numpy(actions),
            torch.from_numpy(pad),
            self._images(ahead),
        )
