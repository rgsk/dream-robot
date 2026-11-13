# Both VAEs on T1-wide-low's held-out demos at the grasp (first frame whose action closes the gripper, and
# 10 frames before it): reconstruction MSE overall and on the wrist half, and a picture: real / step 03 VAE /
# step 07 VAE -> scratch/verne07/vae_grasp.png
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from dream_robot.policies.verne import vae
from dream_robot.policies.verne.data import REPO, canvas, load_cameras, load_table
from dream_robot.policies.verne.run import split

name = 'pick_place_cube_wide_square_low_noise005_n100'
_, action, episode = load_table(name)
rows = []
for e in split('t1-wide-low')[1]:
    idx = np.flatnonzero(episode == e)
    close = idx[np.argmax(action[idx, 7] < 0.5)]
    rows += [close - 10, close]
frames = canvas(*(c[rows] for c in load_cameras(name)))
vaes = {'step 03 VAE': REPO / 'experiments/verne/runs/03_vae_latents/vae/weights.pt', 'step 07 VAE': vae.WEIGHTS}
recs = {}
for k, path in vaes.items():
    net = vae.VAE().cuda().eval()
    net.load_state_dict(torch.load(path))
    recs[k] = vae.reconstruct(net, frames)
    err = (recs[k] / 127.5 - frames / 127.5) ** 2
    print(f'{k}: mse all {err.mean():.4f}, wrist half {err[..., 128:].mean():.4f}, top half {err[..., :128].mean():.4f}')
pick = [1, 3, 5, 7, 9]  # at-close frames of 5 held-out demos
fig, ax = plt.subplots(3, len(pick), figsize=(4 * len(pick), 5.4))
for j, i in enumerate(pick):
    for r, img in enumerate([frames[i], recs['step 03 VAE'][i], recs['step 07 VAE'][i]]):
        ax[r, j].imshow(img.transpose(1, 2, 0))
        ax[r, j].axis('off')
for r, t in enumerate(['real, at the grasp (top | wrist)', 'step 03 VAE', 'step 07 VAE']):
    ax[r, 0].set_title(t, loc='left', fontsize=9)
out = REPO / 'scratch/verne07/vae_grasp.png'
fig.savefig(out, dpi=100, bbox_inches='tight')
print(out)
