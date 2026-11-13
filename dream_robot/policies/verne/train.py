"""The training loop: random batches, AdamW with a cosine learning rate (lr.py, as steps 02-06; the
VAE uses warmup-stable-decay), gradient clipping and checkpoints (lr.py),
a held-out loss every 1000 steps."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from frozendict import frozendict
from numpy.typing import NDArray

from . import lr as lr_mod
from .data import Demos, get_batch
from .model import Verne, loss


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
) -> frozendict[str, Any]:
    # ckpt: a folder for step<N>.pt (every 2000) / stable.pt (resumes from the latest); extend_from:
    # a shorter run's
    # ckpt folder to continue from its stable.pt
    # out: {'weights': state dict on the cpu, 'log': (step, train loss, val loss) every 1000 steps}
    torch.manual_seed(seed)
    rng, val_rng = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    net = Verne().to(d.latent.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    val = [get_batch(d, val_ids, 64, val_rng) for _ in range(4)]  # fixed held-out batches
    start, log = lr_mod.resume(ckpt, extend_from, steps, decay, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, decay, 2000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        b = get_batch(d, train_ids, B, rng)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = loss(net, b.frame, b.state, b.instruction, b.acts)
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                lv = np.mean(
                    [loss(net, v.frame, v.state, v.instruction, v.acts).item() for v in val]
                )
            log.append((step, l.item(), float(lv)))
            print(f'step {step:6d}   train {l.item():.3f}   val {lv:.3f}')
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)
