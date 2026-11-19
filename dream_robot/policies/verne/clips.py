"""Step 19, stage 1: the video side learns to continue a clip from its first frame and the caption (image-to-
video, Wan I2V-style), with no actions, labels or erase task. A clip: the present frame at a random step of an
episode (clean) and FUTURE frames STRIDE steps apart (data.clip_rows), noised with one shared level; the
caption is the episode's prompt ('red left'). Flow matching on the future, as the action side:

    x_τ = (1 − τ)·noise + τ·future,    loss = ‖clip_velocity(present, x_τ, τ) − (future − noise)‖²

The hope: to draw 'red left', the future frames must find the red cube; the cheapest place to do that once is
the present frame's tokens, which the arm reads (the present attends to itself only, so it gets no other
job). Checked by binding_probe (pretrain.py) on the present's patches, as steps 14-18.
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from frozendict import frozendict
from torch import Tensor

from . import lr as lr_mod
from .data import COLOURS, FUTURE, Demos, Scenes, clip_rows, colour_prompts
from .model import Verne
from .pretrain import scene_state


def decision_steps(d: Demos, ids: np.ndarray) -> np.ndarray:
    # per episode in `ids`: the end of its decision window, grasp // 3, the grasp being the first step the
    # gripper command closes (action[7]: 1 open, 0 closed; median step 87). Before it the arm has barely left
    # home and the picture doesn't yet show which cube it's going for (step 19j; checked on films)
    first = np.searchsorted(d.episode, ids)
    lens = np.searchsorted(d.episode, ids, side='right') - first
    closed = (d.action[:, 7] < 0.5).cpu().numpy()
    grasp = np.array([np.argmax(closed[f : f + n]) for f, n in zip(first, lens, strict=True)])
    return np.maximum(grasp // 3, 1)


def clip_batch(
    d: Demos,
    ids: np.ndarray,
    B: int,
    rng: np.random.Generator,
    early: float = 0.0,
    decision: np.ndarray | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # B clips from episodes `ids`, the present at a random step -> present [B, Z, 16, 32], future
    # [B, FUTURE, Z, 16, 32], state [B, 16] (previous, now; std units), the recorded prompt [B, 8].
    # early: the share of clips whose present falls in the decision window (decision = decision_steps(d, ids);
    # step 19j), the rest anywhere in the episode
    first = np.searchsorted(d.episode, ids)
    lens = np.searchsorted(d.episode, ids, side='right') - first
    pick = rng.integers(0, len(ids), B)
    span = lens[pick]
    if early > 0:
        span = np.where(rng.random(B) < early, decision[pick], span)
    now = first[pick] + rng.integers(0, span)
    prev = np.maximum(now - 1, first[pick])
    rows = torch.tensor(clip_rows(d.episode, now), device=d.latent.device)
    z = d.latent[rows].float()  # [B, 1 + FUTURE, Z, 16, 32]
    state = (torch.cat([d.state[prev], d.state[now]], 1) - d.s_mean.repeat(2)) / d.s_std.repeat(2)
    return z[:, 0], z[:, 1:], state, d.instruction[now]


def future_of(d: Demos, now: np.ndarray) -> Tensor:
    # the FUTURE frames of present rows `now` [B] -> [B, FUTURE, Z, 16, 32] (step 19i: the clips of a demo batch)
    rows = torch.tensor(clip_rows(d.episode, now)[:, 1:], device=d.latent.device)
    return d.latent[rows].float()


def velocity(net: Verne, present, x, tau, state, ins, future_by: str) -> Tensor:
    # future_by: 'all' (step 19: future tokens in the video transformer, reading words + state), 'present'
    # (19b: only the present reads them), 'heads' (19d: word-blind top / wrist heads read the present's keys
    # and values, as the arm does); 'change' (19e) has no velocity, see clip_loss
    if future_by == 'heads':
        return net.heads_velocity(present, x, tau, state, ins)
    return net.clip_velocity(present, x, tau, state, ins, future_by)


def clip_loss(
    net: Verne, present, future, state, ins, tau: Tensor, noise: Tensor, future_by: str = 'all'
) -> Tensor:
    # step 19e: plain regression of each patch's change, no noise; 19f: the top camera only
    if future_by in ('change', 'change_top'):
        pred, target = net.change(present, state, ins).float(), future - present[:, None]
        if future_by == 'change_top':
            pred, target = pred[..., :16], target[..., :16]
        return F.mse_loss(pred, target)
    x = (1 - tau[:, None, None, None, None]) * noise + tau[:, None, None, None, None] * future
    v = velocity(net, present, x, tau, state, ins, future_by)
    return F.mse_loss(v.float(), future - noise)


@torch.no_grad()
def val_numbers(net: Verne, val: list, future_by: str = 'all', seed: int = 0) -> dict:
    # fixed held-out clips, loss at τ 0.2 / 0.5 / 0.8 with the right caption and with another clip's
    # (rolled in the batch): 'shuffled' − 'right' = how much the caption is used
    net.eval()
    g = torch.Generator('cuda').manual_seed(seed)
    out = {}
    for tau in (0.2, 0.5, 0.8):
        right = wrong = 0.0
        for present, future, state, ins in val:
            t = torch.full((len(present),), tau, device=present.device)
            noise = torch.randn(future.shape, device=future.device, generator=g)
            right += clip_loss(net, present, future, state, ins, t, noise, future_by).item()
            wrong += clip_loss(
                net, present, future, state, ins.roll(1, 0), t, noise, future_by
            ).item()
        right, wrong = right / len(val), wrong / len(val)
        out[f'{tau}'] = {'right': right, 'shuffled': wrong}
    net.train()
    return out


def pretrain_clips(
    d: Demos,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    steps: int,
    B: int = 32,
    lr: float = 3e-4,
    seed: int = 0,
    ckpt: Path | None = None,
    log_every: int = 1000,
    probe=None,
    probe_every: int = 4000,
    future_by: str = 'all',
    early: float = 0.0,
) -> frozendict[str, Any]:
    # out: {'weights': a whole Verne on the cpu (its action side untouched), 'log': per log_every steps
    # {step, train, val numbers, and every probe_every steps 'probe': probe(net)}}
    torch.manual_seed(seed)
    rng, val_rng = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    net = Verne().to(d.latent.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    # val presents: anywhere in the episode, as before
    val = [clip_batch(d, val_ids, B, val_rng) for _ in range(4)]
    decision = decision_steps(d, train_ids)
    start, log = lr_mod.resume(ckpt, None, steps, 0.2, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, 0.2, 2000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        present, future, state, ins = clip_batch(d, train_ids, B, rng, early, decision)
        tau = torch.rand(B, device=present.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            noise = torch.randn_like(future)
            loss = clip_loss(net, present, future, state, ins, tau, noise, future_by)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % log_every == 0:
            v = val_numbers(net, val, future_by)
            entry = {'step': step, 'train': loss.item(), 'val': v}
            if probe is not None and step % probe_every == 0:
                entry['probe'] = probe(net)  # binding_probe: leaves the net in eval mode
                net.train()
            log.append(entry)
            print(
                f'step {step:6d}   train {loss.item():.4f}   val right / shuffled caption: '
                + '   '.join(f'τ {t} {x["right"]:.4f} / {x["shuffled"]:.4f}' for t, x in v.items()),
                flush=True,
            )
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)


@torch.no_grad()
def dream(
    net: Verne, present, state, ins, steps: int = 8, seed: int = 0, future_by: str = 'all'
) -> Tensor:
    # the future [B, FUTURE, Z, 16, 32] from noise, Euler on an even grid
    net.eval()
    g = torch.Generator('cuda').manual_seed(seed)
    x = torch.randn(len(present), FUTURE, *present.shape[1:], device=present.device, generator=g)
    taus = torch.linspace(0, 1, steps + 1, device=present.device)
    for i in range(steps):
        t = taus[i].expand(len(present))
        x = x + velocity(net, present, x, t, state, ins, future_by) * (taus[i + 1] - taus[i])
    return x


def decode(vae, z: Tensor) -> np.ndarray:
    # latents [..., Z, 16, 32] -> uint8 [..., 128, 256, 3]
    shape = z.shape[:-3]
    x = vae.dec(z.flatten(0, -4) * vae.z_std[:, None, None] + vae.z_mean[:, None, None]).clamp(
        -1, 1
    )
    img = ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return img.reshape(*shape, *img.shape[1:])


@torch.no_grad()
def binding_films(
    net: Verne, vae, sc: Scenes, s_mean, s_std, n: int = 3, future_by: str = 'all'
) -> np.ndarray:
    # the test that matters: a reset scene (both cubes still down) with each cube's colour word (the same
    # side word) -> the dreamed future. Per scene two rows, [present | dream] with the near cube's colour,
    # then the far cube's; the named cube should be the one that moves. uint8 [rows·128, 8·256, 3]
    dev = sc.latent.device
    rows = []
    for i in range(n):
        ids = torch.tensor([i, i], device=dev)
        which = torch.tensor([0, 1], device=dev)
        present = sc.latent[ids].float()
        state, ins = scene_state(sc, ids, s_mean, s_std), colour_prompts(sc, ids, which)
        fut = dream(net, present, state, ins, future_by=future_by)
        img = decode(vae, torch.cat([present[:, None], fut], 1))  # [2, 8, 128, 256, 3]
        rows += [np.concatenate(list(r), 1) for r in img]
    return np.concatenate(rows, 0)


@torch.no_grad()
def change_picture(net: Verne, vae, sc: Scenes, s_mean, s_std, n: int = 3) -> np.ndarray:
    # step 19e: per val reset scene (row) the scene, then the predicted change size per patch (summed over
    # future frames and channels, as a red overlay) with the near cube's colour word, then with the far
    # cube's. The named cube's patches should light up, the other's not. uint8 [n·128, 3·256, 3]
    net.eval()
    dev = sc.latent.device
    ids = torch.arange(n, device=dev)
    present = sc.latent[ids].float()
    scene = decode(vae, present).astype(np.float32)  # [n, 128, 256, 3]
    cols = [scene]
    for which in (0, 1):
        w = torch.full((n,), which, device=dev)
        ins = sc.instruction[ids].clone()
        ins[:, COLOURS] = 0
        ins[torch.arange(n), COLOURS.start + sc.colours[ids, which]] = 1
        size = net.change(present, scene_state(sc, ids, s_mean, s_std), ins).abs().sum((1, 2))
        heat = (size / size.amax((1, 2), keepdim=True)).cpu().numpy()  # [n, 16, 32] 0..1
        heat = heat.repeat(8, 1).repeat(8, 2)[..., None]  # latent cell = 8 x 8 px
        cols.append(scene * (1 - 0.6 * heat) + np.array([255, 0, 0]) * 0.6 * heat)
    grid = np.concatenate([np.concatenate(list(c), 0) for c in cols], 1)
    return grid.clip(0, 255).astype(np.uint8)
