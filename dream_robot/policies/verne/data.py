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
# the instruction as a one-hot: T5's colour words, then its side words. T1's prompt ('put the cube
# in the bin') has none of them, so every T1 frame reads all zeros
WORDS = ('red', 'green', 'blue', 'yellow', 'left', 'right')


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
    # out: [N] by table row, the named cube's slot (0/1, T5's instructions.json: its kept attempts are
    # the episodes in order) on the frames before the episode's first gripper close (action[7] 1 ->
    # 0); -1 after it (the choice is made, the arm's pose gives it away) and where the dataset has
    # no instructions.json (T1, DAgger rounds)
    path = REPO / 'data/robosuite' / name / 'instructions.json'
    _, action, episode = load_table(name)
    if not path.exists():
        return np.full(len(episode), -1, dtype=np.int64)
    kept = [a for a in json.loads(path.read_text())['attempts'] if a['kept']]
    assert len(kept) == episode.max() + 1, (len(kept), episode.max() + 1)
    slot = np.array([a['target_slot'] for a in kept], dtype=np.int64)[episode]
    first = np.searchsorted(episode, episode)  # each row's episode start
    closed = np.maximum.accumulate(np.where(action[:, 7] < 0.5, np.arange(len(episode)), -1))
    slot[closed >= first] = -1  # a close at or before this row, in this episode
    return slot


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
    instruction: Tensor  # [N, 6] one-hot (WORDS)
    action: Tensor  # [N, 8] raw
    slot: Tensor  # [N] int64, the named cube's slot 0/1 (T5), -1 = no label
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
    state: Tensor  # [B, 8] std units
    instruction: Tensor  # [B, 6]
    slot: Tensor  # [B] 0/1, -1 = no label
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
        instruction=d.instruction[now],
        slot=d.slot[now],
        acts=(d.action[rows.to(d.action.device)] - d.a_mean) / d.a_std,
    )


def first_frames(d: Demos, ids: NDArray) -> Batch:
    # each episode's first frame (the home pose: only the picture and the word say which cube);
    # acts are not used
    now = np.searchsorted(d.episode, ids)
    return Batch(
        frame=d.latent[now].float(),
        state=(d.state[now] - d.s_mean) / d.s_std,
        instruction=d.instruction[now],
        slot=d.slot[now],
        acts=torch.empty(0),
    )


# step 09: T5 scenes without a demo, for the slot head only: an env reset (home pose), its first frame through
# the VAE and the named cube's slot. Seeds clear of the demos' (0-164) and the rollouts' (1000-1049)
SCENE_SEEDS = {'train': range(5000, 6000), 'val': range(9000, 9200)}


@dataclass(frozen=True, kw_only=True)
class Scenes:
    latent: Tensor  # [n, Z, 16, 32]
    state: Tensor  # [n, 8] raw
    instruction: Tensor  # [n, 6]
    slot: Tensor  # [n] 0/1


def load_scenes(part: str, vae_weights: Path, dev: str = 'cuda') -> Scenes:
    # part 'train' / 'val' (SCENE_SEEDS). Rendered once (a few minutes) into
    # data/robosuite/t5_scenes/<part>_<first 8 of the VAE weights' sha256>.npz, then read back
    import hashlib

    tag = hashlib.sha256(Path(vae_weights).read_bytes()).hexdigest()[:8]
    path = REPO / 'data/robosuite/t5_scenes' / f'{part}_{tag}.npz'
    if not path.exists():
        from . import vae as vae_mod
        from .rollouts import make_env

        enc, env = vae_mod.load(vae_weights), make_env('t5')
        out = {k: [] for k in ('latent', 'state', 'instruction', 'slot')}
        print(f'rendering {len(SCENE_SEEDS[part])} T5 scenes -> {path}', flush=True)
        for seed in SCENE_SEEDS[part]:
            obs = env.reset(seed=seed)
            out['latent'].append(frame_of(obs, enc)[0].half().cpu().numpy())
            out['state'].append(np.asarray(obs.state, dtype=np.float32))
            out['instruction'].append(one_hot(env.prompt))
            out['slot'].append(env.instruction.target_slot)
        env.close()
        path.parent.mkdir(exist_ok=True)
        np.savez(path.with_suffix('.partial.npz'), **{k: np.stack(v) for k, v in out.items()})
        path.with_suffix('.partial.npz').rename(path)
    z = np.load(path)
    return Scenes(**{k: torch.tensor(z[k], device=dev) for k in ('latent', 'state', 'instruction', 'slot')})


def scene_batch(d: Demos, sc: Scenes, ids) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # -> frame, state (std units, the demos' stats), instruction, slot for the scene rows `ids`
    return sc.latent[ids].float(), (sc.state[ids] - d.s_mean) / d.s_std, sc.instruction[ids], sc.slot[ids]


def frame_of(obs, vae) -> Tensor:
    # a sim observation -> the model's frame [1, Z, 16, 32]: the canvas [top | wrist] now through
    # the frozen VAE, as the training latents were made
    top, wrist = (torch.from_numpy(obs.images[k]).permute(2, 0, 1) for k in ('top', 'wrist'))
    with torch.no_grad():
        return vae.encode(canvas(top, wrist)[None].cuda().float() / 127.5 - 1)
