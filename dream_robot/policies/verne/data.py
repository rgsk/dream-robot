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


def load_cameras(name: str) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    # out: top, wrist, each [N, 3, 128, 128] uint8 by table row. Decoded once per dataset into
    # data/robosuite/<name>/decoded/<camera>.npy (a few minutes; the layout idm's decoded_frames
    # uses), then read from there in seconds by every notebook and step
    folder = REPO / 'data/robosuite' / name / 'decoded'
    paths = {c: folder / f'{c}.npy' for c in ('top', 'wrist')}
    if not all(path.exists() for path in paths.values()):
        ds = open_demos(name)
        print(f'decoding {len(ds)} top + wrist frames once -> {folder}', flush=True)
        out = {c: np.zeros((len(ds), 3, 128, 128), dtype=np.uint8) for c in paths}
        for i in range(len(ds)):  # a dataset item decodes all its cameras: both in one pass
            item = ds[i]
            for c in paths:
                out[c][i] = (item[f'observation.images.{c}'] * 255).round().to(torch.uint8).numpy()
        folder.mkdir(exist_ok=True)
        for c, path in paths.items():
            np.save(path.with_suffix('.partial.npy'), out[c])
            path.with_suffix('.partial.npy').rename(path)  # never a half-written decoded/<c>.npy
    return np.load(paths['top']), np.load(paths['wrist'])


def canvas(top, wrist):
    # [..., 3, 128, 128] each -> [..., 3, 128, 256]: the two cameras side by side, top on the
    # left (numpy or torch)
    return (
        np.concatenate([top, wrist], -1)
        if isinstance(top, np.ndarray)
        else torch.cat([top, wrist], -1)
    )


@dataclass(frozen=True, kw_only=True)
class Demos:
    latent: Tensor  # [N, Z, 16, 32] float16, the [top | wrist] canvas through the VAE
    state: Tensor  # [N, 8] joints + gripper, raw
    action: Tensor  # [N, 8] raw
    episode: NDArray[np.int64]  # [N]
    s_mean: Tensor  # [8], stats from the training demos only
    s_std: Tensor
    a_mean: Tensor
    a_std: Tensor


def to_gpu(latent, state, action, episode, train_ids, dev: str = 'cuda') -> Demos:
    s, a = torch.tensor(state, device=dev), torch.tensor(action, device=dev)
    lab = torch.tensor(np.isin(episode, train_ids), device=dev)
    return Demos(
        latent=torch.tensor(latent, device=dev),
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
    frame: Tensor  # [B, Z, 16, 32], the canvas's latent now
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
        frame=d.latent[now].float(),
        state=(d.state[now] - d.s_mean) / d.s_std,
        acts=(d.action[rows.to(d.action.device)] - d.a_mean) / d.a_std,
    )


def frame_of(obs, vae) -> Tensor:
    # a sim observation -> the model's frame [1, Z, 16, 32]: the canvas [top | wrist] now through
    # the frozen VAE, as the training latents were made
    top, wrist = (torch.from_numpy(obs.images[k]).permute(2, 0, 1) for k in ('top', 'wrist'))
    with torch.no_grad():
        return vae.encode(canvas(top, wrist)[None].cuda().float() / 127.5 - 1)
