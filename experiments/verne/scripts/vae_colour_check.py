# Does a VAE keep T5's cube colours? On T5's 15 held-out demos (every 20th frame): reconstruction MSE,
# and on the cube pixels (saturated pixels of the real frame) which palette colour each pixel is
# nearest to, in the real frame vs the reconstruction -> per colour, the share that comes back the
# same colour. Picture: real over reconstruction for each VAE -> scratch/verne07/vae_colour.png
# usage: python experiments/verne/scripts/vae_colour_check.py <weights or ckpt .pt> [more ...]
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dream_robot.policies.verne import vae
from dream_robot.policies.verne.data import REPO, canvas, load_cameras, load_table

PALETTE = {'red': (217, 26, 26), 'green': (26, 179, 51), 'blue': (26, 64, 230), 'yellow': (242, 217, 26)}
P = np.array(list(PALETTE.values()), dtype=float)

name = 'instructed_pick_place_square_noise005_n150'
episode = load_table(name)[2]
val_ids = np.random.default_rng(0).permutation(150)[:15]
rows = np.flatnonzero(np.isin(episode, val_ids))[::20]
frames = canvas(*(c[rows] for c in load_cameras(name)))
print(f'{len(frames)} held-out T5 canvases from {len(val_ids)} demos')


def colour_ids(img):  # [N, 3, H, W] uint8 -> nearest palette index per pixel [N, H, W]
    x = img.transpose(0, 2, 3, 1).astype(float)
    return ((x[..., None, :] - P) ** 2).sum(-1).argmin(-1)


real = frames.transpose(0, 2, 3, 1).astype(float)
sat = real.max(-1) - real.min(-1)  # cube pixels are saturated; table, bins, robot are grey/white
mask = sat > 100
real_c = colour_ids(frames)
recons = {}
for path in map(Path, sys.argv[1:]):
    state = torch.load(path, weights_only=False)
    net = vae.VAE().cuda().eval()
    net.load_state_dict(state.get('net', state))  # a ckpt (step<N>.pt) or a weights.pt
    rec = np.concatenate([vae.reconstruct(net, frames[i : i + 32]) for i in range(0, len(frames), 32)])
    recons[path] = rec
    mse = float(((rec / 127.5 - frames / 127.5) ** 2).mean())
    rec_c = colour_ids(rec)
    same = {k: float((rec_c[mask & (real_c == j)] == j).mean()) for j, k in enumerate(PALETTE)}
    n = {k: int((mask & (real_c == j)).sum()) for j, k in enumerate(PALETTE)}
    print(f'{path.parent.parent.name}/{path.name}: mse {mse:.4f}; cube pixels back the same colour: '
          + ', '.join(f'{k} {same[k]:.0%}' for k in PALETTE), f'(pixels {n})')
pick = np.linspace(0, len(frames) - 1, 5).astype(int)
fig, ax = plt.subplots(1 + len(recons), len(pick), figsize=(4 * len(pick), 1.7 * (1 + len(recons))))
for j, i in enumerate(pick):
    ax[0, j].imshow(frames[i].transpose(1, 2, 0))
    for k, rec in enumerate(recons.values(), 1):
        ax[k, j].imshow(rec[i].transpose(1, 2, 0))
for a in ax.flat:
    a.axis('off')
ax[0, 0].set_title('held-out T5 (top | wrist)', loc='left', fontsize=9)
for k, path in enumerate(recons, 1):
    ax[k, 0].set_title(f'{path.parent.parent.name}/{path.name}', loc='left', fontsize=9)
out = REPO / 'scratch/verne07/vae_colour.png'
fig.savefig(out, dpi=100, bbox_inches='tight')
print(out)
