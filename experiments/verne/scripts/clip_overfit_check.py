"""Step 19b check (local GPU): the clip denoiser can learn at all. 8 fixed T5 demo clips, 1500 steps on just those
(loss should fall far), then their dreams next to the real clips: real row, dreamed row, per clip.

    python experiments/verne/scripts/clip_overfit_check.py [out.png]
"""

import sys

import imageio.v3 as iio
import numpy as np
import torch

from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.clips import clip_batch, clip_loss, decode, dream
from dream_robot.policies.verne.data import to_gpu
from dream_robot.policies.verne.model import Verne
from dream_robot.policies.verne.run import load_task

out = sys.argv[1] if len(sys.argv) > 1 else 'scratch/clip_overfit.png'
state, action, episode, latent, instruction, slot, train_ids, val_ids, _ = load_task('t5')
d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
present, future, st, ins = clip_batch(d, train_ids, 8, np.random.default_rng(0))

torch.manual_seed(0)
net = Verne().cuda()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
for step in range(1501):
    tau = torch.rand(8, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = clip_loss(net, present, future, st, ins, tau, torch.randn_like(future))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 250 == 0:
        print(f'step {step:5d}   loss {loss.item():.4f}', flush=True)

fut = dream(net, present, st, ins)
print('dream vs real future, mse per latent value:', torch.nn.functional.mse_loss(fut, future).item(),
      '  (copying the present:', torch.nn.functional.mse_loss(present[:, None].expand_as(future), future).item(), ')')
vae = vae_mod.load(vae_mod.WEIGHTS)
real = decode(vae, torch.cat([present[:4, None], future[:4]], 1))
fake = decode(vae, torch.cat([present[:4, None], fut[:4]], 1))
rows = [np.concatenate(list(r), 1) for pair in zip(real, fake) for r in pair]
iio.imwrite(out, np.concatenate(rows, 0))
print('wrote', out)
