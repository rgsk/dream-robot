"""Step 14 stage 1 smoke check (local GPU): the scenes with both colours (rendered once), captions, a short
pretrain (300 steps: the loss falls, the val numbers print), the binding probe and denoise picture, and
the policy path (loss + sample) still running on the same model.

    python experiments/verne/scripts/pretrain_check.py [out.png]
"""

import sys

import imageio.v3 as iio
import numpy as np
import torch

from dream_robot.policies.verne import pretrain as pt
from dream_robot.policies.verne import vae
from dream_robot.policies.verne.data import COLOURS, WORDS, captions, load_scenes, load_table
from dream_robot.policies.verne.model import Verne, loss, sample
from dream_robot.policies.verne.run import TASKS, split

out = sys.argv[1] if len(sys.argv) > 1 else 'pretrain_check.png'
sc, val = load_scenes('train', vae.WEIGHTS), load_scenes('val', vae.WEIGHTS)
print('scenes', len(sc.slot), len(val.slot), 'latent', tuple(sc.latent.shape))

# captions agree with the scene: the named cube's colour (prompt) sits in its slot
named = sc.colours[torch.arange(len(sc.slot)), sc.slot]
assert (sc.instruction[:, COLOURS].argmax(1) == named).all()
assert (sc.colours[:, 0] != sc.colours[:, 1]).all()
ids = torch.arange(3, device='cuda')
for which in (0, 1):
    cap = captions(sc, ids, torch.full((3,), which, device='cuda'))
    print(f'captions slot {which}:', [' '.join(w for w, on in zip(WORDS, r) if on) for r in cap.tolist()])
print('colours of the first 3 scenes (near, far):', sc.colours[:3].tolist())

state, _, episode = load_table(TASKS['t5'][0])
train_ids, _ = split('t5', int(episode.max()) + 1)
s = torch.tensor(state[np.isin(episode, train_ids)], device='cuda')
s_mean, s_std = s.mean(0), s.std(0)

run = pt.pretrain(sc, val, s_mean, s_std, steps=300, seed=0)
print('log:', run['log'])
net = Verne().cuda()
net.load_state_dict(run['weights'])
pt.binding_probe(net, sc, val, s_mean, s_std)
iio.imwrite(out, pt.denoise_picture(net, vae.load(vae.WEIGHTS), val, s_mean, s_std).transpose(1, 2, 0))
print('wrote', out)

# the policy path on the same model
f, st, ins = sc.latent[:4].float(), pt.scene_state(sc, ids[:1].repeat(4), s_mean, s_std), sc.instruction[:4]
print('policy loss', loss(net, f, st, ins, torch.randn(4, 16, 8, device='cuda')).item(),
      'sample', tuple(sample(net, f, st, ins).shape))
print('------ FINISHED ------')
