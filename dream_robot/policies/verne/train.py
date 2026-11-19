"""The training loop: random batches, AdamW with a cosine learning rate (lr.py, as steps 02-06; the
VAE uses warmup-stable-decay), gradient clipping and checkpoints (lr.py),
a held-out loss every 1000 steps.

Step 19g, co-training (cotrain > 0): every step also takes a clip batch from the same training episodes and adds
cotrain x the clip loss (clips.clip_loss, future_by as the clip job's), so the video side keeps predicting the
future while the arm learns to read it. Run time is unchanged: the arm reads the present alone.
Step 19i, paired: the clip batch is the demo batch's own moments (same present, state and prompt; its future
frames), as one Verne sample holds a present, its future and its action chunk. Two passes, but the present's
tokens are the same in both (the clip mask keeps the present from seeing the future), so it equals one joint
pass with the future and the actions masked from each other."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from frozendict import frozendict
from numpy.typing import NDArray

from . import clips
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
    init: dict | None = None,
    cotrain: float = 0.0,
    future_by: str = 'all',
    probe=None,
    probe_every: int = 4000,
    paired: bool = False,
) -> frozendict[str, Any]:
    # ckpt: a folder for step<N>.pt (every 2000) / stable.pt (resumes from the latest); extend_from:
    # a shorter run's ckpt folder to continue from its stable.pt; init: weights to start from (step 15:
    # the stage 1 pretrained Verne; only its video side is loaded, the action side starts fresh: stage 1
    # never trains it, and step 17 changed its shape), else a fresh one
    # out: {'weights': state dict on the cpu, 'log': (step, train loss, val loss) every 1000 steps; with
    # cotrain, (step, train, val, clip loss, probe(net) every probe_every steps or None)}
    torch.manual_seed(seed)
    rng, val_rng = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    net = Verne().to(d.latent.device)
    if init is not None:
        # the action side (action_*, time_in, time_out) starts fresh; the video side, with its clip
        # pretraining parts (time_video, gen_out, ...), is loaded
        fresh = ('action', 'time_in', 'time_out')
        video = {k: v for k, v in init.items() if not k.startswith(fresh)}
        missing, unexpected = net.load_state_dict(video, strict=False)
        # pretraining-only parts an older checkpoint doesn't have start fresh too
        newer = ('edit_out', 'time_video', 'gen_out', 'future_heads', 'change_out')
        assert not unexpected and all(k.startswith(fresh + newer) for k in missing), (
            [k for k in missing if not k.startswith(fresh + newer)],
            unexpected,
        )
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    val = [get_batch(d, val_ids, 64, val_rng) for _ in range(4)]  # fixed held-out batches
    start, log = lr_mod.resume(ckpt, extend_from, steps, decay, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, decay, 2000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        b = get_batch(d, train_ids, B, rng)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = loss(net, b.frame, b.state, b.instruction, b.acts)
            total = l
            if cotrain > 0:
                if (
                    paired
                ):  # step 19i: the demo batch's own moments: same present, its future frames
                    present, state, ins = b.frame, b.state, b.instruction
                    future = clips.future_of(d, b.now)
                else:
                    present, future, state, ins = clips.clip_batch(d, train_ids, B, rng)
                tau = torch.rand(B, device=present.device)
                noise = torch.randn_like(future)
                lc = clips.clip_loss(net, present, future, state, ins, tau, noise, future_by)
                total = l + cotrain * lc
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                lv = float(
                    np.mean(
                        [loss(net, v.frame, v.state, v.instruction, v.acts).item() for v in val]
                    )
                )
            if cotrain > 0:
                pr = probe(net) if probe is not None and step % probe_every == 0 else None
                net.train()
                log.append((step, l.item(), lv, lc.item(), pr))
                print(
                    f'step {step:6d}   train {l.item():.3f}   val {lv:.3f}   clip {lc.item():.3f}'
                )
            else:
                log.append((step, l.item(), lv))
                print(f'step {step:6d}   train {l.item():.3f}   val {lv:.3f}')
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)
