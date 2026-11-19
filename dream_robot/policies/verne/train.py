"""The training loop: random batches, AdamW with a cosine learning rate (lr.py, as steps 02-06; the
VAE uses warmup-stable-decay), gradient clipping and checkpoints (lr.py),
a held-out loss every 1000 steps.

Step 19g, co-training (cotrain > 0): every step also takes a clip batch from the same training episodes and adds
cotrain x the clip loss (clips.clip_loss, future_by as the clip job's), so the video side keeps predicting the
future while the arm learns to read it. Run time is unchanged: the arm reads the present alone.
Step 19i, paired: the clip batch is the demo batch's own moments (same present, state and prompt; its future
frames), as one Verne sample holds a present, its future and its action chunk. Two passes, but the present's
tokens are the same in both (the clip mask keeps the present from seeing the future), so it equals one joint
pass with the future and the actions masked from each other.
Step 21a, erase co-training (erase > 0): every step also takes a batch of reset scenes and adds erase x step 15's
erase loss (pretrain.py: colour word -> the change that hides that cube), so demo training can't wear off the
mark the erase pretraining put on the named cube's patches."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from frozendict import frozendict
from numpy.typing import NDArray

from . import clips
from . import lr as lr_mod
from . import pretrain as pt
from .data import Demos, colour_prompts, get_batch
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
    early: float = 0.0,
    erase: float = 0.0,
    erase_data=None,
) -> frozendict[str, Any]:
    # ckpt: a folder for step<N>.pt (every 2000) / stable.pt (resumes from the latest); extend_from:
    # a shorter run's ckpt folder to continue from its stable.pt; init: weights to start from (step 15:
    # the stage 1 pretrained Verne; only its video side is loaded, the action side starts fresh: stage 1
    # never trains it, and step 17 changed its shape), else a fresh one
    # out: {'weights': state dict on the cpu, 'log': (step, train loss, val loss) every 1000 steps; with
    # cotrain (or erase), (step, train, val, clip (or erase) loss, probe(net) every probe_every steps or None)}
    # erase_data: (scenes, erased, B) for erase > 0 (pretrain.py's; the demos' state stats)
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
    # early (step 19k): that share of training samples (demo batch and clip batch) from the decision window;
    # val batches stay uniform
    decision = clips.decision_steps(d, train_ids) if early > 0 else None
    start, log = lr_mod.resume(ckpt, extend_from, steps, decay, net, opt, rng)

    def losses(net, frame, state, ins, acts, clip, ers):
        # the step's loss terms: action, and the clip or erase loss (None when off)
        l, lc = loss(net, frame, state, ins, acts), None
        if clip is not None:
            lc = clips.clip_loss(net, *clip, future_by)
        if ers is not None:
            ef, es, ei, target = ers
            lc = F.mse_loss(net.edit(ef, es, ei).float(), target)
        return l, lc

    # step 21a: one compiled graph per step, replayed as a CUDA graph (the 4.5M-param model is launch-bound:
    # local 4060 26 -> 15 ms per step with the erase loss, then GPU-bound)
    step_losses = torch.compile(losses, mode='reduce-overhead')
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, decay, 2000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        b = get_batch(d, train_ids, B, rng, early, decision)
        clip = ers = None
        if cotrain > 0:
            if paired:  # step 19i: the demo batch's own moments: same present, its future frames
                present, state, ins = b.frame, b.state, b.instruction
                future = clips.future_of(d, b.now)
            else:
                present, future, state, ins = clips.clip_batch(
                    d, train_ids, B, rng, early, decision
                )
            tau = torch.rand(B, device=present.device)
            clip = (present, future, state, ins, tau, torch.randn_like(future))
        if erase > 0:
            sc, erased, eB = erase_data
            ids = torch.tensor(rng.integers(0, len(sc.slot), eB), device=sc.latent.device)
            which = torch.tensor(rng.integers(0, 2, eB), device=sc.latent.device)
            ers = (
                sc.latent[ids].float(),
                pt.scene_state(sc, ids, d.s_mean, d.s_std),
                colour_prompts(sc, ids, which),
                pt.target(sc, erased, ids, which),
            )
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l, lc = step_losses(net, b.frame, b.state, b.instruction, b.acts, clip, ers)
        total = l + cotrain * lc if cotrain > 0 else l + erase * lc if erase > 0 else l
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
            if cotrain > 0 or erase > 0:
                pr = probe(net) if probe is not None and step % probe_every == 0 else None
                net.train()
                log.append((step, l.item(), lv, lc.item(), pr))
                print(
                    f'step {step:6d}   train {l.item():.3f}   val {lv:.3f}   '
                    f'{"erase" if erase > 0 else "clip"} {lc.item():.4f}',
                    flush=True,
                )
            else:
                log.append((step, l.item(), lv))
                print(f'step {step:6d}   train {l.item():.3f}   val {lv:.3f}')
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)
