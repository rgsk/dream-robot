"""Step 17 check: how much does the action side use the frame at each noise level? Val loss at fixed flow
times τ (the sampler's 4 grid points, plus 0.9) with the true frame and with another sample's frame (rolled
within the batch; state and prompt kept), and with another sample's state row (frame kept). gap = shuffled
− true: how much the frame / the state matters at that τ.
Step 16's weights load into step 16's model.py (git show 2cec91a), step 17b's into the current one.

    python experiments/verne/scripts/time_use_check.py
"""

import importlib.util
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from dream_robot.policies.verne import model as model17
from dream_robot.policies.verne.data import get_batch, to_gpu
from dream_robot.policies.verne.run import RUNS, load_task


def old_model(commit):
    src = subprocess.run(
        ['git', 'show', f'{commit}:dream_robot/policies/verne/model.py'],
        capture_output=True, text=True, check=True,
    ).stdout
    name = 'dream_robot.policies.verne.model_old'
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = 'dream_robot.policies.verne'
    sys.modules[name] = mod
    exec(compile(src, 'model_old.py', 'exec'), mod.__dict__)
    return mod


state, action, episode, latent, instruction, slot, train_ids, val_ids, _ = load_task('t5')
d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
rng = np.random.default_rng(0)
batches = [get_batch(d, val_ids, 256, rng) for _ in range(8)]
TAUS = (0.0, 0.18, 0.40, 0.67, 0.9)

old = old_model('2cec91a')
runs = {
    f'{step} seed {k}': (mod, RUNS / folder / f't5/seed{k}/weights.pt')
    # step 17's weights need step 17's model (Wan's gate start), uncommitted and replaced by 17b's; its numbers
    # are in notes.md
    for step, mod, folder in (('16', old, '16_colour_only'), ('17b', model17, '17b_gate1'))
    for k in (0, 1)
}
for name, (mod, path) in runs.items():
    net = mod.Verne().cuda().eval()
    net.load_state_dict(torch.load(path))
    g = torch.Generator('cuda').manual_seed(0)
    print(f'step {name}:   tau    true   frame shuffled (gap)   state shuffled (gap)')
    for tau in TAUS:
        acc = {'true': 0.0, 'frame': 0.0, 'state': 0.0}
        for b in batches:
            noise = torch.randn(b.acts.shape, device='cuda', generator=g)
            t = torch.full((len(noise),), tau, device='cuda')
            x = (1 - tau) * noise + tau * b.acts
            with torch.no_grad():
                for key, frame, st in (
                    ('true', b.frame, b.state),
                    ('frame', b.frame.roll(1, 0), b.state),
                    ('state', b.frame, b.state.roll(1, 0)),
                ):
                    kvs = net.see(frame, st, b.instruction)
                    l = F.mse_loss(net.velocity(kvs, x, t, b.instruction, st), b.acts - noise).item()
                    acc[key] += l / len(batches)
        tr = acc['true']
        print(
            f'           {tau:.2f}   {tr:.3f}   {acc["frame"]:.3f} ({acc["frame"] - tr:.3f})'
            f'          {acc["state"]:.3f} ({acc["state"] - tr:.3f})'
        )
print('------ FINISHED ------')
