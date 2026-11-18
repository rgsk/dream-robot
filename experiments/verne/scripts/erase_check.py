"""Step 15 stage 1 smoke check (local GPU): the erased scenes line up with the plain ones, the change is
local to the hidden cube, colour prompts name the right cube, and a short pretrain (300 steps) runs
through val numbers, the binding probe and edit_picture; the policy path still runs.

    python experiments/verne/scripts/erase_check.py [out.png]
"""

import sys

import imageio.v3 as iio
import numpy as np
import torch

from dream_robot.policies.verne import pretrain as pt
from dream_robot.policies.verne import vae
from dream_robot.policies.verne.data import (
    COLOURS,
    WORDS,
    colour_prompts,
    load_erased,
    load_scenes,
    load_table,
)
from dream_robot.policies.verne.model import Verne, loss, sample
from dream_robot.policies.verne.run import TASKS, split

out = sys.argv[1] if len(sys.argv) > 1 else 'erase_check.png'
sc, val = load_scenes('train', vae.WEIGHTS), load_scenes('val', vae.WEIGHTS)
erased, val_erased = load_erased('train', vae.WEIGHTS), load_erased('val', vae.WEIGHTS)
print('erased', tuple(erased.shape), tuple(val_erased.shape))

# the change is small and local: share of latent cells that move (summed |change| over channels > 0.3)
change = (erased.float() - sc.latent.float()[:, None]).abs().sum(2)  # [N, 2, 16, 32]
moved = (change > 0.3).float().mean((2, 3))
print(f'cells moved per erase: mean {moved.mean():.3f}, max {moved.max():.3f} (of 512)')
# near and far erases move different cells
overlap = ((change[:, 0] > 0.3) & (change[:, 1] > 0.3)).float().mean().item()
print(f'cells moved by both erases: {overlap:.4f}')

ids = torch.arange(3, device='cuda')
for which in (0, 1):
    p = colour_prompts(sc, ids, torch.full((3,), which, device='cuda'))
    print(f'prompt for slot {which}:', [' '.join(w for w, on in zip(WORDS, r) if on) for r in p.tolist()])
    assert (p[:, COLOURS].argmax(1) == sc.colours[ids, which]).all()

state, _, episode = load_table(TASKS['t5'][0])
train_ids, _ = split('t5', int(episode.max()) + 1)
s = torch.tensor(state[np.isin(episode, train_ids)], device='cuda')
s_mean, s_std = s.mean(0), s.std(0)

run = pt.pretrain(sc, erased, val, val_erased, s_mean, s_std, steps=300, seed=0)
print('log:', run['log'])
net = Verne().cuda()
net.load_state_dict(run['weights'])
print('val after 300 steps:', pt.val_numbers(net, val, val_erased, s_mean, s_std))
pt.binding_probe(net, sc, val, s_mean, s_std)
grid = pt.edit_picture(net, vae.load(vae.WEIGHTS), val, val_erased, s_mean, s_std)
iio.imwrite(out, grid.transpose(1, 2, 0))
print('wrote', out)

f, st, ins = sc.latent[:4].float(), pt.scene_state(sc, ids[:1].repeat(4), s_mean, s_std), sc.instruction[:4]
print(
    'policy loss', loss(net, f, st, ins, torch.randn(4, 16, 8, device='cuda')).item(),
    'sample', tuple(sample(net, f, st, ins).shape),
)
print('------ FINISHED ------')
