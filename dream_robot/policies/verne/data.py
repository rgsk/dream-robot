"""Demos (T1, T1-wide, T5) on the GPU, and random training samples from them."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

REPO = Path(__file__).resolve().parents[3]
CHUNK = 16  # actions per sample
# step 20 (roadmap box 06): an action = the hand's move relative to the hand at the chunk's start (3 position,
# 6 rotation; hand.py), the jaw, progress 0..1
ACTION = 11
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


def as_frame(t: Tensor) -> Tensor:
    # stored frames -> the model's input: latents (float16) as floats; raw canvases (uint8, vae.INPUT 'pixels')
    # scaled to [-1, 1], as the VAE saw them
    return t.float() / 127.5 - 1 if t.dtype == torch.uint8 else t.float()


@dataclass(frozen=True, kw_only=True)
class Demos:
    latent: Tensor  # [N, Z, 16, 32] float16, the [top | wrist] canvas through the VAE (pixels: [N, 3, 128, 256] uint8)
    state: Tensor  # [N, 8] joints + gripper, raw
    instruction: Tensor  # [N, 8] one-hot (WORDS), the recorded prompt
    slot: Tensor  # [N] the named cube's slot (0 near, 1 far), -1 without one (T1)
    action: Tensor  # [N, 8] raw, the recorded joint targets + gripper command
    episode: NDArray[np.int64]  # [N]
    hand_pos: Tensor  # [N, 3], hand_rot [N, 3, 3]: the hand at each state (forward kinematics)
    hand_rot: Tensor
    goal_pos: Tensor  # [N, 3], goal_rot [N, 3, 3]: the hand at each recorded joint target
    goal_rot: Tensor
    progress: Tensor  # [N] step / (episode length − 1)
    s_mean: Tensor  # [8], stats from the training demos only
    s_std: Tensor
    a_mean: Tensor  # [ACTION], over the training rows' chunks
    a_std: Tensor


def to_gpu(
    latent, state, instruction, slot, action, episode, train_ids, dev: str = 'cuda'
) -> Demos:
    from . import hand

    s, a = torch.tensor(state, device=dev), torch.tensor(action, device=dev)
    lab = np.isin(episode, train_ids)
    kin = (
        hand.t5_kinematics()
    )  # the Panda arm; relative moves don't depend on where its base stands
    hp, hr = kin.pose(state[:, :7])
    gp, gr = kin.pose(action[:, :7])
    first = np.searchsorted(episode, episode)
    length = np.searchsorted(episode, episode, side='right') - first
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)  # noqa: E731
    d = Demos(
        latent=torch.tensor(latent, device=dev),
        state=s,
        instruction=torch.tensor(instruction, device=dev),
        slot=torch.tensor(slot, device=dev),
        action=a,
        episode=episode,
        hand_pos=t(hp),
        hand_rot=t(hr),
        goal_pos=t(gp),
        goal_rot=t(gr),
        progress=t((np.arange(len(episode)) - first) / np.maximum(length - 1, 1)),
        s_mean=s[lab].mean(0),
        s_std=s[lab].std(0),
        a_mean=torch.zeros(ACTION, device=dev),
        a_std=torch.ones(ACTION, device=dev),
    )
    # action stats over every training row's chunk
    now = np.flatnonzero(lab)
    raw = targets(d, now, chunk_rows(d, now))
    return replace(d, a_mean=raw.mean((0, 1)), a_std=raw.std((0, 1)))


def chunk_rows(d: Demos, now: NDArray) -> NDArray:
    # [B] rows -> [B, CHUNK] the next CHUNK rows, held at the episode's last
    last = np.searchsorted(d.episode, d.episode[now], side='right') - 1
    return np.minimum(now[:, None] + np.arange(CHUNK), last[:, None])


def targets(d: Demos, now: NDArray, rows: NDArray) -> Tensor:
    # raw action targets [B, CHUNK, ACTION]: the hand at each recorded joint target in rows, relative to the hand
    # at the chunk's start (state at now), then the recorded gripper command and the progress
    from . import hand

    n, r = torch.as_tensor(now, device=d.state.device), torch.as_tensor(rows, device=d.state.device)
    rel = hand.relative_t(
        d.hand_pos[n][:, None], d.hand_rot[n][:, None], d.goal_pos[r], d.goal_rot[r]
    )
    return torch.cat([rel, d.action[r][..., 7:8].float(), d.progress[r][..., None]], -1)


@dataclass(frozen=True, kw_only=True)
class Batch:
    frame: Tensor  # [B, Z, 16, 32], the canvas's latent now
    state: Tensor  # [B, 16] std units: [previous step, now] (a demo's first step: previous = now)
    instruction: Tensor  # [B, 8]
    acts: Tensor  # [B, CHUNK, ACTION] std units, the next CHUNK hand moves (step 20)
    now: NDArray  # [B] the samples' table rows (step 19i: their clips, for paired co-training)


def get_batch(
    d: Demos,
    ids: NDArray,
    B: int,
    rng: np.random.Generator,
    early: float = 0.0,
    decision: NDArray | None = None,
) -> Batch:
    # B samples: a random demo from `ids`, a random step of it; actions past the demo's end
    # repeat its last one, so the end of every demo is trained on too. The prompt is the recorded one
    # (step 16: colour + side only; steps 13-15 mixed in near/far prompts, train and val alike)
    first = np.searchsorted(d.episode, ids)
    lens = np.searchsorted(d.episode, ids, side='right') - first
    # early (step 19k): the share of samples from the decision window (decision = clips.decision_steps)
    pick = rng.integers(0, len(ids), B)
    span = lens[pick]
    if early > 0:
        span = np.where(rng.random(B) < early, decision[pick], span)
    now = first[pick] + rng.integers(0, span)  # [B]
    prev = np.maximum(now - 1, first[pick])
    return Batch(
        frame=as_frame(d.latent[now]),
        state=(torch.cat([d.state[prev], d.state[now]], 1) - d.s_mean.repeat(2))
        / d.s_std.repeat(2),
        instruction=d.instruction[now],
        acts=(targets(d, now, chunk_rows(d, now)) - d.a_mean) / d.a_std,
        now=now,
    )


# Step 19 clips: a present frame and FUTURE frames STRIDE control steps apart (frames past the episode's end
# repeat its last). 7 x 32 = 224 steps covers most of a T5 episode (192-324 steps, median 251).
FUTURE, STRIDE = 7, 32


def clip_rows(episode: NDArray, now: NDArray) -> NDArray[np.int64]:
    # episode [N] (each episode's rows contiguous), present rows now [B] -> rows [B, 1 + FUTURE]: the present,
    # then now + STRIDE·k, held at the episode's last row
    last = np.searchsorted(episode, episode[now], side='right') - 1
    return np.minimum(now[:, None] + STRIDE * np.arange(1 + FUTURE), last[:, None])


# T5 reset scenes (no demo, home pose) for the video side's pretraining (steps 14-15) and the binding probe
SCENE_SEEDS = {'train': range(5000, 6000), 'val': range(9000, 9200)}


@dataclass(frozen=True, kw_only=True)
class Scenes:
    latent: Tensor  # [N, Z, 16, 32] float16, the first frame's canvas through the VAE
    state: Tensor  # [N, 8] raw, the home pose
    instruction: Tensor  # [N, 8] the reset's prompt ('red left'), WORDS one-hot
    slot: Tensor  # [N] its named cube's slot (0 near, 1 far)
    colours: Tensor  # [N, 2] each slot's colour, an index into the colour words (0 red .. 3 yellow)


def load_scenes(part: str, vae_weights, dev: str = 'cuda', frames: str | None = None) -> Scenes:
    # part 'train' / 'val' (SCENE_SEEDS). Rendered once (a few minutes) into
    # data/robosuite/t5_scenes/<part>_<first 8 of the VAE weights' sha256>_colours.npz, then read back.
    # frames: 'latent' / 'pixels' (default vae.INPUT); pixels: the raw canvas in [-1, 1] (float16) instead of the
    # latent, cached as <part>_pixels_colours.npz
    import hashlib

    from . import vae as vae_mod

    frames = frames or vae_mod.INPUT
    tag = 'pixels' if frames == 'pixels' else hashlib.sha256(Path(vae_weights).read_bytes()).hexdigest()[:8]
    path = REPO / 'data/robosuite/t5_scenes' / f'{part}_{tag}_colours.npz'
    if not path.exists():
        from .rollouts import make_env

        enc = vae_mod.Pixels() if frames == 'pixels' else vae_mod.load_vae(vae_weights)
        env = make_env('t5')
        colour_words = WORDS[COLOURS]
        out = {k: [] for k in ('latent', 'state', 'instruction', 'slot', 'colours')}
        print(f'rendering {len(SCENE_SEEDS[part])} T5 scenes -> {path}', flush=True)
        for seed in SCENE_SEEDS[part]:
            obs = env.reset(seed=seed)
            ins = env.instruction
            colours = [ins.distractor_color, ins.distractor_color]
            colours[ins.target_slot] = ins.color
            out['latent'].append(frame_of(obs, enc)[0].half().cpu().numpy())
            out['state'].append(np.asarray(obs.state, dtype=np.float32))
            out['instruction'].append(one_hot(env.prompt))
            out['slot'].append(ins.target_slot)
            out['colours'].append([colour_words.index(c) for c in colours])
        env.close()
        path.parent.mkdir(exist_ok=True)
        np.savez(path.with_suffix('.partial.npz'), **{k: np.stack(v) for k, v in out.items()})
        path.with_suffix('.partial.npz').rename(path)  # never a half-written file
    z = np.load(path)
    return Scenes(**{k: torch.tensor(z[k], device=dev) for k in z.files})


def load_erased(part: str, vae_weights, dev: str = 'cuda') -> Tensor:
    # always the VAE's latents (vae.load would hand pixel mode a pass-through): position labels and the locate
    # probe come from these, whatever the policy sees
    # [N, 2, Z, 16, 32] float16: each SCENE_SEEDS[part] scene's first frame with its near (0) / far (1)
    # cube hidden (alpha 0, both cameras re-rendered), through the VAE; rows as load_scenes. Rendered
    # once into data/robosuite/t5_scenes/<part>_<VAE tag>_erased.npy
    import hashlib

    tag = hashlib.sha256(Path(vae_weights).read_bytes()).hexdigest()[:8]
    path = REPO / 'data/robosuite/t5_scenes' / f'{part}_{tag}_erased.npy'
    if not path.exists():
        from . import vae as vae_mod
        from .rollouts import make_env

        enc, env, out = vae_mod.load_vae(vae_weights), make_env('t5'), []
        print(f'rendering {len(SCENE_SEEDS[part])} T5 scenes x 2 erased -> {path}', flush=True)
        for seed in SCENE_SEEDS[part]:
            env.reset(seed=seed)
            # after reset: robosuite rebuilds the model each reset
            model = env._env.sim.model._model
            pair = []
            for slot in (0, 1):
                model.geom_rgba[model.geom_bodyid == env._env.cube_body_ids[slot], 3] = 0.0
                env._raw = env._env._get_observations(force_update=True)
                pair.append(frame_of(env._observe(), enc)[0].half().cpu().numpy())
                env._paint()  # both cubes back, as reset painted them
            assert np.abs(pair[0] - pair[1]).max() > 0.5, f'seed {seed}: erasing changed nothing'
            out.append(np.stack(pair))
        env.close()
        np.save(path.with_suffix('.partial.npy'), np.stack(out))
        path.with_suffix('.partial.npy').rename(path)  # never a half-written file
    return torch.tensor(np.load(path), device=dev)


def colour_prompts(sc: Scenes, ids: Tensor, which: Tensor) -> Tensor:
    # [B, 8]: the colour word of slot `which` [B] (0 near, 1 far) in scenes `ids`, nothing else ('red')
    out = torch.zeros(len(ids), len(WORDS), device=ids.device)
    out[torch.arange(len(ids)), COLOURS.start + sc.colours[ids, which]] = 1.0
    return out


def frame_of(obs, vae) -> Tensor:
    # a sim observation -> the model's frame [1, Z, 16, 32]: the canvas [top | wrist] now through
    # the frozen VAE, as the training latents were made
    top, wrist = (torch.from_numpy(obs.images[k]).permute(2, 0, 1) for k in ('top', 'wrist'))
    with torch.no_grad():
        return vae.encode(canvas(top, wrist)[None].cuda().float() / 127.5 - 1)
