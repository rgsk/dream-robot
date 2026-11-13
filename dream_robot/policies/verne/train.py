"""The training loop: random batches, AdamW with a cosine learning rate (lr.py, as steps 02-06; the
VAE uses warmup-stable-decay), gradient clipping and checkpoints (lr.py),
a held-out loss (and the slot head's held-out accuracy) every 1000 steps."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from frozendict import frozendict
from numpy.typing import NDArray

from . import lr as lr_mod
import torch.nn.functional as F

from .data import Demos, Scenes, first_frames, get_batch, scene_batch
from .model import SLOT_WEIGHT, Verne, loss


def train(
    d: Demos,
    train_ids: NDArray,
    val_ids: NDArray,
    steps: int,
    B: int = 32,
    lr: float = 3e-4,
    decay: float = 0.2,  # only names where stable.pt is saved; the schedule is cosine
    seed: int = 0,
    ckpt: Path | None = None,
    extend_from: Path | None = None,
    scenes: tuple[Scenes, Scenes] | None = None,
) -> frozendict[str, Any]:
    # ckpt: a folder for step<N>.pt (every 2000) / stable.pt (resumes from the latest); extend_from:
    # a shorter run's
    # ckpt folder to continue from its stable.pt; scenes: (train, val) demo-less T5 scenes for the slot
    # head (step 09): each step also gives the head B of them, its loss the mean of demo rows and scenes
    # out: {'weights': state dict on the cpu, 'log': (step, train flow loss, val flow loss, val slot
    # accuracy, val first-frame slot accuracy (both None when no val row is labelled), val scene slot
    # accuracy (None without scenes)) every 1000 steps}
    torch.manual_seed(seed)
    rng, val_rng = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    net = Verne().to(d.latent.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    val = [get_batch(d, val_ids, 64, val_rng) for _ in range(4)]  # fixed held-out batches
    val_first = first_frames(d, val_ids)
    start, log = lr_mod.resume(ckpt, extend_from, steps, decay, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, decay, 2000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        b = get_batch(d, train_ids, B, rng)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l, ce, _ = loss(net, b.frame, b.state, b.instruction, b.slot, b.acts)
            if scenes:
                frame, st, ins, slot = scene_batch(d, scenes[0], rng.integers(0, len(scenes[0].slot), B))
                ce = (ce + F.cross_entropy(net.see(frame, st, ins)[1].float(), slot)) / 2
        opt.zero_grad()
        (l + SLOT_WEIGHT * ce).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = [loss(net, v.frame, v.state, v.instruction, v.slot, v.acts) for v in val]
                f = val_first
                logits = net.see(f.frame, f.state, f.instruction)[1]
                acc_s = None
                if scenes:
                    frame, st, ins, slot = scene_batch(d, scenes[1], slice(None))
                    acc_s = (net.see(frame, st, ins)[1].argmax(-1) == slot).float().mean().item()
            lv = float(np.mean([o[0].item() for o in out]))
            right = torch.cat([o[2] for o in out])
            acc = right.float().mean().item() if len(right) else None
            lab = f.slot >= 0
            acc0 = (logits[lab].argmax(-1) == f.slot[lab]).float().mean().item() if lab.any() else None
            log.append((step, l.item(), lv, acc, acc0, acc_s))
            shown = '' if acc is None else (
                f'   slot ce {ce.item():.3f}   val slot acc {acc:.2f}   first frame {acc0:.2f}'
            ) + ('' if acc_s is None else f'   scenes {acc_s:.2f}')
            print(f'step {step:6d}   train {l.item():.3f}   val {lv:.3f}{shown}')
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)
