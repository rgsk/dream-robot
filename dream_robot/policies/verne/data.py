"""T1 demos on the GPU, and random training samples from them."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

REPO = Path(__file__).resolve().parents[3]
CHUNK = 16  # actions per sample


def open_demos(name: str):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(f'dream_robot/{name}', root=REPO / 'data/robosuite' / name)


def load_table(name: str) -> tuple[NDArray, NDArray, NDArray]:
    # out: state [N, 8], action [N, 8], episode [N]; no video decode
    ds = open_demos(name)
    return (
        np.asarray(ds.hf_dataset['observation.state'], dtype=np.float32),
        np.asarray(ds.hf_dataset['action'], dtype=np.float32),
        np.asarray(ds.hf_dataset['episode_index'], dtype=np.int64),
    )


def load_top(name: str) -> NDArray[np.uint8]:
    # out: [N, 3, 128, 128] uint8, every top-camera frame decoded once (a few minutes)
    ds = open_demos(name)
    top = np.zeros((len(ds), 3, 128, 128), dtype=np.uint8)
    for i in range(len(ds)):
        top[i] = (ds[i]['observation.images.top'] * 255).round().to(torch.uint8).numpy()
    return top


@dataclass(frozen=True, kw_only=True)
class Demos:
    top: Tensor  # [N, 3, 128, 128] uint8
    state: Tensor  # [N, 8] joints + gripper, raw
    action: Tensor  # [N, 8] raw
    episode: NDArray[np.int64]  # [N]
    s_mean: Tensor  # [8], stats from the training demos only
    s_std: Tensor
    a_mean: Tensor
    a_std: Tensor


def to_gpu(top, state, action, episode, train_ids, dev: str = 'cuda') -> Demos:
    s, a = torch.tensor(state, device=dev), torch.tensor(action, device=dev)
    lab = torch.tensor(np.isin(episode, train_ids), device=dev)
    return Demos(
        top=torch.tensor(top, device=dev),
        state=s,
        action=a,
        episode=episode,
        s_mean=s[lab].mean(0),
        s_std=s[lab].std(0),
        a_mean=a[lab].mean(0),
        a_std=a[lab].std(0),
    )


@dataclass(frozen=True, kw_only=True)
class Batch:
    frame: Tensor  # [B, 3, 128, 128] in [-1, 1], the top camera now
    state: Tensor  # [B, 8] std units
    acts: Tensor  # [B, CHUNK, 8] std units, the next CHUNK actions


def get_batch(d: Demos, ids: NDArray, B: int, rng: np.random.Generator) -> Batch:
    # B samples: a random demo from `ids`, a random step of it; actions past the demo's end
    # repeat its last one, so the end of every demo is trained on too
    first = np.searchsorted(d.episode, ids)
    lens = np.searchsorted(d.episode, ids, side='right') - first
    pick = rng.integers(0, len(ids), B)
    now = first[pick] + rng.integers(0, lens[pick])  # [B]
    last = first[pick] + lens[pick] - 1
    rows = torch.tensor(np.minimum(now[:, None] + np.arange(CHUNK), last[:, None]))
    return Batch(
        frame=d.top[now].float() / 127.5 - 1,
        state=(d.state[now] - d.s_mean) / d.s_std,
        acts=(d.action[rows.to(d.action.device)] - d.a_mean) / d.a_std,
    )
