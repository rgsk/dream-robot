"""Reading a recorded dataset for training. Nothing here knows which simulator wrote it.

Two things live here because both are easy to get quietly wrong: how the train
and validation sets are split, and where the normalisation constants come from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dream_robot.core.dataset import (
    ACTION_FEATURE,
    STATE_FEATURE,
    embodiment_from_names,
    image_feature,
    open_dataset,
)
from dream_robot.core.schema import Embodiment
from dream_robot.policies.bc.model import NormalizationStats

#: Frames sampled to measure per-pixel image statistics. A few hundred is plenty
#: for a per-channel mean and std, and it keeps startup to a couple of seconds.
IMAGE_STAT_SAMPLES = 256


@dataclass(frozen=True)
class DatasetSpec:
    """What the dataset tells a policy about its own shape. ROADMAP rule 2.

    Read once, at the top of training, and copied into the checkpoint. Nothing
    downstream re-opens the dataset to find out how big an action is.
    """

    repo_id: str
    root: Path
    fps: float
    state_dim: int
    action_dim: int
    embodiment: Embodiment
    cameras: tuple[str, ...]
    image_hw: tuple[int, int]
    prompt: str
    episodes: int
    frames: int


def read_spec(repo_id: str, root: Path) -> DatasetSpec:
    meta = open_dataset(repo_id, root).meta
    cameras = tuple(key.rsplit(".", 1)[-1] for key in meta.camera_keys)
    if not cameras:
        raise ValueError(
            f"{repo_id} has no cameras. State-only BC is a pipeline sanity check "
            "and never a reported result (spec.md)."
        )
    _, height, width = meta.features[image_feature(cameras[0])]["shape"]
    tasks = list(meta.tasks.index) if hasattr(meta.tasks, "index") else list(meta.tasks)
    return DatasetSpec(
        repo_id=repo_id,
        root=Path(root),
        fps=float(meta.fps),
        state_dim=int(meta.features[STATE_FEATURE]["shape"][0]),
        action_dim=int(meta.features[ACTION_FEATURE]["shape"][0]),
        embodiment=embodiment_from_names(meta.features[ACTION_FEATURE]["names"]),
        cameras=cameras,
        image_hw=(int(height), int(width)),
        prompt=str(tasks[0]) if tasks else "",
        episodes=int(meta.total_episodes),
        frames=int(meta.total_frames),
    )


def split_episodes(
    count: int, *, validation: int, seed: int = 0
) -> tuple[list[int], list[int]]:
    """Hold out whole episodes, never individual frames.

    **This is the one that silently inflates every number.** Consecutive frames
    of a 30 Hz episode are nearly identical -- the arm moves under a millimetre
    between them -- so a random frame-level split puts a frame's own neighbours
    in the other set. Validation loss then measures interpolation between
    adjacent frames of a trajectory the model has already memorised, reports a
    tiny number, and predicts nothing about a rollout on an unseen scene.

    Splitting by episode makes the held-out set a set of cube positions the
    model has never seen, which is the thing the eval harness will actually
    test.
    """
    if validation < 1:
        raise ValueError(f"need at least one validation episode, got {validation}")
    if validation >= count:
        raise ValueError(
            f"cannot hold out {validation} of {count} episodes; nothing would be left"
        )
    order = np.random.default_rng(seed).permutation(count)
    return sorted(order[validation:].tolist()), sorted(order[:validation].tolist())


class FrameDataset(Dataset):
    """One (images, state, action) triple per frame of the chosen episodes.

    Backed by LeRobot's own loader, which decodes video on demand and hands back
    channel-first float32 in [0, 1] -- the model's input format already, so no
    conversion sits between the dataset and training where it could disagree
    with the one used at rollout.
    """

    def __init__(
        self,
        repo_id: str,
        root: Path,
        cameras: tuple[str, ...],
        episodes: list[int],
    ):
        self._dataset = open_dataset(repo_id, root)
        self._cameras = cameras
        episode_index = np.asarray(self._dataset.hf_dataset["episode_index"])
        self._indices = np.flatnonzero(np.isin(episode_index, episodes))
        if self._indices.size == 0:
            raise ValueError(f"episodes {episodes} contain no frames")

    def __len__(self) -> int:
        return int(self._indices.size)

    def __getitem__(self, i: int):
        item = self._dataset[int(self._indices[i])]
        images = {name: item[image_feature(name)] for name in self._cameras}
        return images, item[STATE_FEATURE], item[ACTION_FEATURE]


def random_shift(images: torch.Tensor, pad: int) -> torch.Tensor:
    """Translate each image by up to ``pad`` pixels, edge-padded.

    The one augmentation worth having here. 20 training episodes see the cube in
    20 places, and without it the encoder can key on absolute pixel position and
    still fit every frame. Shifting breaks that: a feature has to be found
    wherever it is, which is what the spatial softmax is for.

    Applied per batch rather than per sample -- one shift for the whole batch is
    a weaker augmentation but costs one ``roll`` instead of N crops, and at 30 Hz
    the batch is a random draw across episodes anyway.
    """
    if pad < 1:
        return images
    dy, dx = np.random.randint(-pad, pad + 1, size=2)
    return torch.roll(images, shifts=(int(dy), int(dx)), dims=(-2, -1))


def compute_stats(
    dataset: FrameDataset, cameras: tuple[str, ...], *, samples: int = IMAGE_STAT_SAMPLES
) -> NormalizationStats:
    """Normalisation constants, measured on the training split only.

    State and action constants are measured here rather than taken from
    ``meta.stats`` for one reason: LeRobot's stats cover the whole dataset, and
    fitting normalisation on data that includes the validation episodes is a
    small leak that gets copied into every policy that follows this one.

    **Image stats must be measured, not read.** LeRobot's ``meta.stats`` reports
    an image ``std`` of ~0.003 for this dataset where the true per-pixel std is
    ~0.28 -- it is a spread of aggregated per-image statistics, not of pixels.
    Normalising by it would scale the input by about 100x. The mean it reports is
    right; only the std is a different quantity than the name suggests.
    """
    states, actions = [], []
    for _, state, action in dataset:
        states.append(np.asarray(state))
        actions.append(np.asarray(action))
    states, actions = np.stack(states), np.stack(actions)

    step = max(1, len(dataset) // samples)
    frames = torch.stack([
        dataset[i][0][cameras[0]] for i in range(0, len(dataset), step)
    ])
    return NormalizationStats(
        state_mean=states.mean(0),
        state_std=states.std(0),
        action_mean=actions.mean(0),
        action_std=actions.std(0),
        image_mean=frames.mean(dim=(0, 2, 3)).numpy(),
        image_std=frames.std(dim=(0, 2, 3)).numpy(),
    )


def make_loader(
    dataset: FrameDataset, *, batch_size: int, shuffle: bool, workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=workers > 0,
    )
