"""Step 19a film check, on the 150 T5 demos (already VAE latents): clips as data.clip_rows cuts them, a present
frame at a random step and FUTURE frames STRIDE steps apart, decoded back through the VAE. One row per clip
(present first), captioned with the prompt and the row's start step / episode length. Does every clip, from
any start, show the named cube going to the named box?

    python experiments/verne/scripts/clip_film_check.py [out.png]
"""

import sys

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw

from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.data import FUTURE, STRIDE, WORDS, clip_rows
from dream_robot.policies.verne.run import load_task

out = sys.argv[1] if len(sys.argv) > 1 else 'scratch/clip_films.png'
state, action, episode, latent, instruction, slot, train_ids, val_ids, _ = load_task('t5')
vae = vae_mod.load(vae_mod.WEIGHTS)
rng = np.random.default_rng(0)

first = np.searchsorted(episode, np.arange(episode.max() + 1))
length = np.bincount(episode)
eps = rng.choice(train_ids, 6, replace=False)
# starts: 0 (the reset), then random steps, the last two late enough that future frames run past the end
starts = [0, *(rng.integers(0, length[e] - 1) for e in eps[1:4]), length[eps[4]] - 100, length[eps[5]] - 20]
now = np.array([first[e] + t for e, t in zip(eps, starts)])
rows = clip_rows(episode, now)  # [6, 8]
print('rows - present:', (rows - now[:, None]).tolist())

z = torch.tensor(latent[rows.flatten()], device='cuda').float()
with torch.no_grad():
    x = vae.dec(z * vae.z_std[:, None, None] + vae.z_mean[:, None, None]).clamp(-1, 1)
img = ((x + 1) * 127.5).round().byte().cpu().numpy().transpose(0, 2, 3, 1)  # [48, 128, 256, 3]
img = img.reshape(len(eps), 1 + FUTURE, 128, 256, 3)
H = 18
grid = np.full((len(eps) * (128 + H), (1 + FUTURE) * 256, 3), 255, np.uint8)
for i in range(len(eps)):
    grid[i * (128 + H) + H : (i + 1) * (128 + H)] = np.concatenate(list(img[i]), 1)
pic = Image.fromarray(grid)
draw = ImageDraw.Draw(pic)
for i, e in enumerate(eps):
    words = ' '.join(w for w, on in zip(WORDS, instruction[now[i]]) if on)
    draw.text(
        (4, i * (128 + H) + 3),
        f'"{words}"  episode {e}, start step {starts[i]} of {length[e]}  '
        f'(frames every {STRIDE} steps; present first)',
        fill=(0, 0, 0),
    )
iio.imwrite(out, np.asarray(pic))
print('wrote', out)
