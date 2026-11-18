"""Demos (T1, T1-wide, T5) on the GPU, and random training samples from them."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

REPO = Path(__file__).resolve().parents[3]
CHUNK = 16  # actions per sample
# the instruction as a one-hot: the cube's place (step 13: near = slot 0, x -0.07, nearer the robot
# base at x -0.56; far = slot 1, x +0.07), T5's colour words, then its side words. T1's prompt ('put
# the cube in the bin') has none of them, so every T1 frame reads all zeros
WORDS = ('near', 'far', 'red', 'green', 'blue', 'yellow', 'left', 'right')
PLACES, COLOURS = slice(0, 2), slice(2, 6)  # PLACES[slot] = the slot's word
# how a T5 prompt names its cube (step 13): by colour (the recorded prompt), by place, or both
KINDS = ('colour', 'position', 'both')


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


def one_hot(prompt: str) -> NDArray[np.float32]:
    # 'put the green cube in the left box' -> [6]: 1 at 'green' and 'left'
    words = prompt.lower().replace('.', ' ').split()
    return np.array([w in words for w in WORDS], dtype=np.float32)


def load_instructions(name: str) -> NDArray[np.float32]:
    # out: [N, 6] by table row, each frame's prompt as one_hot
    ds = open_demos(name)
    prompts = {i: t for t, i in ds.meta.tasks['task_index'].items()}
    table = np.stack([one_hot(prompts[i]) for i in range(len(prompts))])
    return table[np.asarray(ds.hf_dataset['task_index'], dtype=np.int64)]


def load_slots(name: str) -> NDArray[np.int64]:
    # out: [N] by table row, the named cube's slot (0 near, 1 far) from the recorder's
    # instructions.json (its kept attempts, in episode order); -1 for datasets without one (T1)
    ds = open_demos(name)
    episode = np.asarray(ds.hf_dataset['episode_index'], dtype=np.int64)
    path = REPO / 'data/robosuite' / name / 'instructions.json'
    if not path.exists():
        return np.full(len(episode), -1)
    kept = [a for a in json.loads(path.read_text())['attempts'] if a['kept']]
    return np.array([a['target_slot'] for a in kept])[episode]


def reword(ins, slot, kind):
    # one-hot [..., 8] of the recorded prompt, slot [...] (0 near, 1 far), kind (KINDS index, [...])
    # -> the one-hot of the same instruction named by colour (unchanged), place (colour words off,
    # the slot's word on) or both. Numpy or torch; rows with slot -1 (T1) stay as they are
    lib = torch if isinstance(ins, Tensor) else np
    out = ins.clone() if lib is torch else ins.copy()
    t5 = slot >= 0
    pos, both = t5 & (kind == 1), t5 & (kind == 2)
    out[..., COLOURS] = out[..., COLOURS] * ~pos[..., None]
    named = pos | both
    place = lib.zeros_like(out[..., PLACES])
    if lib is torch:
        place.scatter_(-1, slot.clamp(min=0)[..., None], 1.0)
    else:
        np.put_along_axis(place, np.maximum(slot, 0)[..., None], 1.0, -1)
    out[..., PLACES] = place * named[..., None]
    return out


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
    instruction: Tensor  # [N, 8] one-hot (WORDS), the recorded prompt
    slot: Tensor  # [N] the named cube's slot (0 near, 1 far), -1 without one (T1)
    action: Tensor  # [N, 8] raw
    episode: NDArray[np.int64]  # [N]
    s_mean: Tensor  # [8], stats from the training demos only
    s_std: Tensor
    a_mean: Tensor
    a_std: Tensor


def to_gpu(
    latent, state, instruction, slot, action, episode, train_ids, dev: str = 'cuda'
) -> Demos:
    s, a = torch.tensor(state, device=dev), torch.tensor(action, device=dev)
    lab = torch.tensor(np.isin(episode, train_ids), device=dev)
    return Demos(
        latent=torch.tensor(latent, device=dev),
        state=s,
        instruction=torch.tensor(instruction, device=dev),
        slot=torch.tensor(slot, device=dev),
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
    state: Tensor  # [B, 16] std units: [previous step, now] (a demo's first step: previous = now)
    instruction: Tensor  # [B, 8]
    acts: Tensor  # [B, CHUNK, 8] std units, the next CHUNK actions


def get_batch(d: Demos, ids: NDArray, B: int, rng: np.random.Generator, mix: bool = True) -> Batch:
    # B samples: a random demo from `ids`, a random step of it; actions past the demo's end
    # repeat its last one, so the end of every demo is trained on too. mix: each T5 sample's prompt
    # names the cube by colour, place or both (KINDS, uniform); False: the recorded prompt
    first = np.searchsorted(d.episode, ids)
    lens = np.searchsorted(d.episode, ids, side='right') - first
    pick = rng.integers(0, len(ids), B)
    now = first[pick] + rng.integers(0, lens[pick])  # [B]
    prev = np.maximum(now - 1, first[pick])
    last = first[pick] + lens[pick] - 1
    rows = torch.tensor(np.minimum(now[:, None] + np.arange(CHUNK), last[:, None]))
    return Batch(
        frame=d.latent[now].float(),
        state=(torch.cat([d.state[prev], d.state[now]], 1) - d.s_mean.repeat(2))
        / d.s_std.repeat(2),
        instruction=reword(
            d.instruction[now],
            d.slot[now],
            torch.tensor(rng.integers(0, 3 if mix else 1, B), device=d.slot.device),
        ),
        acts=(d.action[rows.to(d.action.device)] - d.a_mean) / d.a_std,
    )


def frame_of(obs, vae) -> Tensor:
    # a sim observation -> the model's frame [1, Z, 16, 32]: the canvas [top | wrist] now through
    # the frozen VAE, as the training latents were made
    top, wrist = (torch.from_numpy(obs.images[k]).permute(2, 0, 1) for k in ('top', 'wrist'))
    with torch.no_grad():
        return vae.encode(canvas(top, wrist)[None].cuda().float() / 127.5 - 1)
