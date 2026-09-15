"""Reading a recorded dataset as (images, state, next ``chunk`` actions) samples.

The episode split, augmentation and loader are BC's, imported rather than copied,
so the two policies are trained on identical frames with identical held-out
episodes and the matrix compares architectures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dream_robot.core.dataset import ACTION_FEATURE, STATE_FEATURE, image_feature, open_dataset
from dream_robot.policies.bc.data import IMAGE_STAT_SAMPLES
from dream_robot.policies.bc.model import NormalizationStats


class ChunkDataset(Dataset):
    """One sample per frame: its images, state, and the next ``chunk`` actions.

    **A chunk never crosses into the next episode.** Near an episode's end the
    chunk is filled out by repeating the final action, and ``pad`` marks those
    steps so the loss and the CVAE encoder ignore them. Repeating the last
    action rather than zeros keeps the padded values in range if anything does
    read them.

    States and actions are read once into memory from the table (fast); only
    images are decoded per sample.
    """

    def __init__(
        self,
        repo_id: str,
        root: Path,
        cameras: tuple[str, ...],
        episodes: list[int],
        chunk: int,
    ):
        if chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {chunk}")
        self._dataset = open_dataset(repo_id, root)
        self._cameras = cameras
        self._chunk = chunk
        table = self._dataset.hf_dataset
        episode_index = np.asarray(table["episode_index"])
        self.states = np.asarray(table[STATE_FEATURE], dtype=np.float32)
        self.actions = np.asarray(table[ACTION_FEATURE], dtype=np.float32)

        self._last = np.empty(len(episode_index), dtype=np.int64)
        for episode in np.unique(episode_index):
            rows = np.flatnonzero(episode_index == episode)
            if rows[-1] - rows[0] + 1 != rows.size:
                raise ValueError(
                    f"episode {episode} is not contiguous; chunks would cross episodes"
                )
            self._last[rows] = rows[-1]

        self.indices = np.flatnonzero(np.isin(episode_index, episodes))
        if self.indices.size == 0:
            raise ValueError(f"episodes {episodes} contain no frames")

    def __len__(self) -> int:
        return int(self.indices.size)

    def chunk_at(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """``(actions (chunk, dim), pad (chunk,))`` starting at table row ``row``."""
        stop = min(row + self._chunk, int(self._last[row]) + 1)
        actions = self.actions[row:stop]
        pad = np.zeros(self._chunk, dtype=bool)
        pad[len(actions) :] = True
        if len(actions) < self._chunk:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self._chunk - len(actions), axis=0)]
            )
        return actions, pad

    def __getitem__(self, i: int):
        row = int(self.indices[i])
        actions, pad = self.chunk_at(row)
        item = self._dataset[row]
        images = {name: item[image_feature(name)] for name in self._cameras}
        return (
            images,
            torch.from_numpy(self.states[row]),
            torch.from_numpy(actions),
            torch.from_numpy(pad),
        )


def compute_stats(
    dataset: ChunkDataset, cameras: tuple[str, ...], *, samples: int = IMAGE_STAT_SAMPLES
) -> NormalizationStats:
    """Normalisation constants from the training episodes only.

    State and action statistics come straight from the in-memory table rather
    than by iterating the dataset, which would decode every image of every frame
    just to read two small vectors. Image statistics are measured on sampled
    frames (LeRobot's reported image std is a different quantity -- see
    ``policies/bc/data.py``).
    """
    states = dataset.states[dataset.indices]
    actions = dataset.actions[dataset.indices]
    step = max(1, len(dataset) // samples)
    frames = torch.stack([dataset[i][0][cameras[0]] for i in range(0, len(dataset), step)])
    return NormalizationStats(
        state_mean=states.mean(0),
        state_std=states.std(0),
        action_mean=actions.mean(0),
        action_std=actions.std(0),
        image_mean=frames.mean(dim=(0, 2, 3)).numpy(),
        image_std=frames.std(dim=(0, 2, 3)).numpy(),
    )
