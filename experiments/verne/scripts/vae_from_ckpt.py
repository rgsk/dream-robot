# A VAE checkpoint (lr.py step<N>.pt) -> weights.pt beside its ckpt/ folder, with the latent stats
# (vae.set_latent_stats) over VAE_DATA's training frames, as `run vae` would have written at the end.
# usage: python experiments/verne/scripts/vae_from_ckpt.py experiments/verne/runs/07_instruction/vae_wsd/ckpt/step6000.pt
import sys
from pathlib import Path

import torch

from dream_robot.policies.verne import vae
from dream_robot.policies.verne.run import vae_frames

ckpt = Path(sys.argv[1])
net = vae.VAE().cuda().eval()
net.load_state_dict(torch.load(ckpt, weights_only=False)['net'])
frames, _ = vae_frames()
print(f'{len(frames)} training frames')
vae.set_latent_stats(net, frames)
out = ckpt.parent.parent / 'weights.pt'
torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out)
print(f'{out} (from {ckpt.name})')
