"""The training loop: random batches, AdamW with a cosine learning rate (lr -> 0 over the run),
a held-out loss every 1000 steps."""

import numpy as np
import torch
from frozendict import frozendict
from numpy.typing import NDArray

from .data import Demos, get_batch
from .model import Verne, loss


def train(
    d: Demos,
    train_ids: NDArray,
    val_ids: NDArray,
    steps: int,
    B: int = 32,
    lr: float = 3e-4,
    seed: int = 0,
):
    # out: {'weights': state dict on the cpu, 'log': (step, train loss, val loss) every 1000 steps}
    torch.manual_seed(seed)
    rng, val_rng = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    net = Verne().to(d.canvas.device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    # step i trains at lr * (1 + cos(pi * i / steps)) / 2: lr at the start, 0 at the last step
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    val = [get_batch(d, val_ids, 64, val_rng) for _ in range(4)]  # fixed held-out batches
    log = []
    for step in range(steps + 1):
        b = get_batch(d, train_ids, B, rng)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = loss(net, b.frame, b.state, b.acts)
        opt.zero_grad()
        l.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                lv = np.mean([loss(net, v.frame, v.state, v.acts).item() for v in val])
            log.append((step, l.item(), float(lv)))
            print(f'step {step:6d}   train {l.item():.3f}   val {lv:.3f}')
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)
