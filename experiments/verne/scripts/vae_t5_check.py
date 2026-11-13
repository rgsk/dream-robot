# Does the step 03 VAE (trained on T1 frames) keep T5's scene and colours?
# Every 500th canvas of T1-wide and T5 -> reconstruction MSE; a strip of T5 originals over
# reconstructions -> scratch/verne07/vae_t5.png
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dream_robot.policies.verne import vae
from dream_robot.policies.verne.data import REPO, canvas, load_cameras

enc = vae.load(vae.WEIGHTS)
for name in ('pick_place_cube_wide_n100', 'instructed_pick_place_clean_n150'):
    f = canvas(*(c[::500] for c in load_cameras(name)))
    r = vae.reconstruct(enc, f)
    print(name, len(f), 'mse', round(float(((r / 127.5 - f / 127.5) ** 2).mean()), 4))
pick = [0, 10, 25, 40, 60]
fig, ax = plt.subplots(2, len(pick), figsize=(4 * len(pick), 3.2))
for j, i in enumerate(pick):
    ax[0, j].imshow(f[i].transpose(1, 2, 0))
    ax[1, j].imshow(r[i].transpose(1, 2, 0))
    for a in ax[:, j]:
        a.axis('off')
ax[0, 0].set_title('T5 frames (top | wrist)', loc='left')
ax[1, 0].set_title('step 03 VAE', loc='left')
out = REPO / 'scratch/verne07/vae_t5.png'
fig.savefig(out, dpi=110, bbox_inches='tight')
print(out)
