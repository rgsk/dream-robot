"""The learning-rate schedule both trainings use, warmup-stable-decay (WSD), and checkpoints that
let a run resume after a crash, be rewound, or be extended to more steps. Gradients are clipped
to norm CLIP: a constant-lr stable phase is where one bad batch can throw a run off (the first
WSD VAE spiked 20x at step 3500 at lr 1e-3).

    lr
     │    ┌──────────────────────────────┐
     │   ╱                                ╲     decay: the last `decay` share of the run,
     │  ╱                                  ╲    cosine-shaped down to 0
     └──────────────────────────────────────── step
       warmup        stable (constant)

The stable phase doesn't depend on the run's length, so the checkpoint saved where the decay
starts (stable.pt) can be continued at the same lr into a longer run: a 10k run's stable.pt is
step 8000, from which a 15k run goes stable to 12000 and decays from there.
"""

import math
from pathlib import Path

import numpy as np
import torch

CLIP = 1.0  # max gradient norm, both trainings


def wsd(i: int, steps: int, warmup: int, decay: float = 0.2) -> float:
    """
    the lr factor at step i of a `steps` run:
        i < warmup:          (i + 1) / warmup                      ramp up
        warmup ≤ i < s₀:     1                                     stable
        i ≥ s₀:              (1 + cos(π · (i − s₀) / (steps − s₀))) / 2      down to 0
    with s₀ = decay_start(steps, decay) = steps − decay · steps
    """
    s0 = decay_start(steps, decay)
    if i < warmup:
        return (i + 1) / warmup
    if i < s0:
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * min(i - s0, steps - s0) / (steps - s0)))


def cosine(i: int, steps: int) -> float:
    # the lr factor at step i: (1 + cos(π · i / steps)) / 2, 1 at the start, 0 at the last step (Verne
    # steps 02-06, which scored 42/44 on T1-wide; back for Verne after WSD runs dropped to 29/23)
    return 0.5 * (1 + math.cos(math.pi * min(i, steps) / steps))


def decay_start(steps: int, decay: float = 0.2) -> int:
    return steps - round(decay * steps)


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g['lr'] = lr


def save(path: Path, net, opt, step: int, rng: np.random.Generator, log: list, steps: int) -> None:
    # everything a resume needs: the state *before* `step` runs (steps 0 .. step-1 are done)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        'step': step, 'steps': steps, 'log': log, 'net': net.state_dict(),
        'opt': opt.state_dict(), 'rng': rng.bit_generator.state,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state(),
    }
    torch.save(state, path.with_suffix('.partial'))
    path.with_suffix('.partial').rename(path)  # never a half-written checkpoint


def resume(
    ckpt: Path | None, extend_from: Path | None, steps: int, decay: float, net, opt, rng
) -> tuple[int, list]:
    # -> (first step to run, log so far). This run's latest ckpt/step<N>.pt if there is one (a
    # crash), else extend_from/stable.pt (a shorter run's state where its decay began; this run
    # decays later)
    path = None
    done = sorted(ckpt.glob('step*.pt'), key=lambda p: int(p.stem[4:])) if ckpt else []
    if done:
        path = done[-1]
    elif extend_from is not None:
        path = extend_from / 'stable.pt'
    if path is None:
        return 0, []
    state = torch.load(path, weights_only=False)
    if path.name != 'stable.pt' and state['steps'] != steps:
        raise ValueError(f'{path} is from a {state["steps"]}-step run, this one is {steps}')
    if path.name == 'stable.pt' and state['step'] > decay_start(steps, decay):
        raise ValueError(f"{path} is at step {state['step']}, past this run's decay start")
    net.load_state_dict(state['net'])
    opt.load_state_dict(state['opt'])
    rng.bit_generator.state = state['rng']
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'])
    print(f'resumed from {path} at step {state["step"]}', flush=True)
    return state['step'], state['log']


def checkpoint(ckpt: Path | None, step: int, steps: int, decay: float, every: int, *state) -> None:
    # at the top of a step: step<N>.pt every `every` steps, all kept (a run can be rewound to before
    # a loss spike), and stable.pt where the decay begins
    if ckpt is None or step == 0:
        return
    if step == decay_start(steps, decay):
        save(ckpt / 'stable.pt', *state, steps)
    if step % every == 0:
        save(ckpt / f'step{step}.pt', *state, steps)
