"""Step 14, stage 1: the video side learns to paint the words, as Wan learned from captioned video.

A T5 reset scene's frame (no demo), noised to a random noise time tau, and a one-cube caption
('near red': that slot's place and colour) -> the clean frame (x-prediction). At low noise the
frame shows its colours; at high noise only the caption says which cube is which colour, so the
words must reach the right place.

Then binding_probe asks whether that also built the policy's lookup (colour -> place): frozen
weights, a clean frame and a colour prompt ('red left'), a linear probe on the patches entering
the last video layer (what the last action layer reads) -> near / far. An untrained model scores
~73% (32,768 features from 1000 scenes), step 13 71%.

    noisy = (1 − τ)·ε + τ·x,   ε ~ N(0, 1),  τ ~ U(0, 1)          (τ = 1 clean, as the action side's flow)
    loss  = ‖denoise(noisy, caption, τ) − x‖²
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from frozendict import frozendict
from torch import Tensor

from . import lr as lr_mod
from .data import PLACES, Scenes, captions
from .model import Verne

VAL_TAUS = (0.1, 0.5)  # high noise (colours gone: only the caption tells) and middle


def scene_state(sc: Scenes, ids: Tensor, s_mean: Tensor, s_std: Tensor) -> Tensor:
    # [B, 16] std units, the demos' stats; a reset scene: previous = now
    return (sc.state[ids].repeat(1, 2) - s_mean.repeat(2)) / s_std.repeat(2)


def pretrain_loss(
    net: Verne, frame: Tensor, state: Tensor, cap: Tensor, tau: Tensor, noise: Tensor
) -> Tensor:
    t = tau[:, None, None, None]
    return F.mse_loss(net.denoise((1 - t) * noise + t * frame, state, cap, tau), frame)


@torch.no_grad()
def val_losses(net: Verne, sc: Scenes, s_mean: Tensor, s_std: Tensor) -> dict[str, float]:
    # each val scene at each VAL_TAUS with fixed noise, captioned right ('near red' where near is
    # red) and
    # wrong (the same place with the other cube's colour); wrong - right = how much the caption is
    # used
    g = torch.Generator(device=sc.latent.device).manual_seed(0)
    ids = torch.arange(len(sc.slot), device=sc.latent.device)
    which = torch.randint(0, 2, (len(ids),), device=ids.device, generator=g)
    right = captions(sc, ids, which)
    wrong = right.clone()
    wrong[:, 2:6] = captions(sc, ids, 1 - which)[:, 2:6]  # the other cube's colour word, same place
    frame, state = sc.latent.float(), scene_state(sc, ids, s_mean, s_std)
    noise = torch.randn(frame.shape, device=frame.device, generator=g)
    out = {}
    for tau in VAL_TAUS:
        t = torch.full((len(ids),), tau, device=ids.device)
        for name, cap in (('right', right), ('wrong', wrong)):
            out[f'{name}_{tau}'] = pretrain_loss(net, frame, state, cap, t, noise).item()
    return out


def pretrain(
    sc: Scenes,
    val: Scenes,
    s_mean: Tensor,
    s_std: Tensor,
    steps: int,
    B: int = 64,
    lr: float = 3e-4,
    seed: int = 0,
    ckpt: Path | None = None,
) -> frozendict[str, Any]:
    # out: {'weights': state dict on the cpu (the whole Verne: its action side untouched), 'log':
    # per 1000
    # steps {step, train, right_0.1, wrong_0.1, right_0.5, wrong_0.5}}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = sc.latent.device
    net = Verne().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    start, log = lr_mod.resume(ckpt, None, steps, 0.2, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, 0.2, 2000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        ids = torch.tensor(rng.integers(0, len(sc.slot), B), device=dev)
        which = torch.tensor(rng.integers(0, 2, B), device=dev)  # which cube the caption names
        frame = sc.latent[ids].float()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = pretrain_loss(
                net,
                frame,
                scene_state(sc, ids, s_mean, s_std),
                captions(sc, ids, which),
                torch.rand(B, device=dev),
                torch.randn_like(frame),
            )
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 1000 == 0:
            v = val_losses(net, val, s_mean, s_std)
            log.append({'step': step, 'train': l.item()} | v)
            print(
                f'step {step:6d}   train {l.item():.3f}   tau 0.1 right {v["right_0.1"]:.3f} wrong '
                f'{v["wrong_0.1"]:.3f}   tau 0.5 right {v["right_0.5"]:.3f} wrong {v["wrong_0.5"]:.3f}',
                flush=True,
            )
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)


def binding_probe(
    net: Verne, sc: Scenes, val: Scenes, s_mean: Tensor, s_std: Tensor
) -> dict[str, dict[str, float]]:
    # -> {'colour' / 'both': {'train', 'val'} accuracy}. Prompts: the reset's own ('red left');
    # 'both' adds
    # the named cube's place word ('near red left': the ceiling, the answer is in the words)
    net.eval()

    def features(s: Scenes, kind: str) -> Tensor:
        ins = s.instruction.clone()
        if kind == 'both':
            ins[:, PLACES] = F.one_hot(s.slot, 2).float()
        captured = []
        hook = net.video[-1].norm1.register_forward_pre_hook(
            lambda m, a: captured.append(a[0].flatten(1).float())
        )
        with torch.no_grad():
            for i in range(0, len(s.slot), 100):
                ids = torch.arange(i, min(i + 100, len(s.slot)), device=s.latent.device)
                net.see(s.latent[ids].float(), scene_state(s, ids, s_mean, s_std), ins[ids])
        hook.remove()
        return torch.cat(captured)

    out = {}
    for kind in ('colour', 'both'):
        f = {'train': features(sc, kind), 'val': features(val, kind)}
        mu, sd = f['train'].mean(0), f['train'].std(0) + 1e-6
        torch.manual_seed(0)
        head = torch.nn.Linear(f['train'].shape[1], 2).to(mu.device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-1)
        for _ in range(2000):
            opt.zero_grad()
            F.cross_entropy(head((f['train'] - mu) / sd), sc.slot).backward()
            opt.step()
        with torch.no_grad():
            out[kind] = {
                p: (head((f[p] - mu) / sd).argmax(-1) == s.slot).float().mean().item()
                for p, s in (('train', sc), ('val', val))
            }
        print(
            f'probe, prompt {kind:6s}: train {out[kind]["train"]:.3f}   val {out[kind]["val"]:.3f}',
            flush=True,
        )
    return out


@torch.no_grad()
def denoise_picture(
    net: Verne, vae, val: Scenes, s_mean: Tensor, s_std: Tensor, n: int = 4
) -> np.ndarray:
    # uint8 [3, H, W]: per val scene (column) the clean canvas, the noised one at tau 0.1, and
    # one-shot
    # predictions captioned 'near <its colour>' and 'far <its colour>' (each true, about a
    # different cube)
    net.eval()
    dev = val.latent.device
    ids = torch.arange(n, device=dev)
    frame, state = val.latent[ids].float(), scene_state(val, ids, s_mean, s_std)
    g = torch.Generator(device=dev).manual_seed(0)
    tau = torch.full((n,), 0.1, device=dev)
    noisy = 0.9 * torch.randn(frame.shape, device=dev, generator=g) + 0.1 * frame
    rows = [frame, noisy]
    for which in (0, 1):
        cap = captions(val, ids, torch.full((n,), which, device=dev))
        rows.append(net.denoise(noisy, state, cap, tau))

    def decode(z: Tensor) -> Tensor:
        x = vae.dec(z * vae.z_std[:, None, None] + vae.z_mean[:, None, None]).clamp(-1, 1)
        return ((x + 1) * 127.5).round().byte()  # [n, 3, 128, 256]

    grid = torch.cat(
        [torch.cat(list(decode(r)), 2) for r in rows], 1
    )  # rows stacked, scenes side by side
    return grid.cpu().numpy()
