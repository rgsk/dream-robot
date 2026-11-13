# Step 10: what the video side imagines. Held-out T5 demos at t = 0 (the choice) and 40: three rows each, real /
# one-shot (pure noise, tau 0: the mean guess) / sampled (10 Euler steps, x-prediction v = (x_hat - x)/(1 - tau));
# columns = present top camera, then the future frames (data.FUTURE steps ahead). Latents through the VAE decoder.
# -> <run>/future_frames.png + .txt
# usage: python experiments/verne/scripts/future_frames.py experiments/verne/runs/10_future_video/t5-clip/seed0
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.data import FUTURE, REPO, to_gpu
from dream_robot.policies.verne.model import Verne
from dream_robot.policies.verne.run import TASKS, load_task

run = Path(sys.argv[1])
state, action, episode, latent, instruction, train_ids, val_ids, _ = load_task('t5')
d = to_gpu(latent, state, instruction, action, episode, train_ids)
net = Verne().cuda().eval()
net.load_state_dict(torch.load(run / 'weights.pt'))
vae = vae_mod.load(vae_mod.WEIGHTS)
kept = [a for a in json.loads((REPO / 'data/robosuite' / TASKS['t5'][0] / 'instructions.json').read_text())['attempts'] if a['kept']]


def decode(z):  # normalised latents [B, Z, 16, w] -> uint8 [B, 128, 8w, 3]
    x = vae.dec(z * vae.z_std[:, None, None] + vae.z_mean[:, None, None]).clamp(-1, 1)
    return ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def strip(present, clip):  # present top [Z, 16, 16], clip [F, Z, 16, 16] -> one image row
    return np.concatenate(list(decode(torch.cat([present[None], clip]))), 1)


torch.manual_seed(0)
rows, captions = [], []
for ep in val_ids[:4]:
    first, last = np.searchsorted(episode, ep), np.searchsorted(episode, ep, side='right') - 1
    for t in (0, 40):
        now = min(first + t, last)
        frame = d.latent[[now]].float()
        real = d.latent[np.minimum(now + np.array(FUTURE), last)][..., :16].float()[None]
        s, ins = (d.state[[now]] - d.s_mean) / d.s_std, d.instruction[[now]]
        with torch.no_grad():
            noise = torch.randn_like(real)
            one = net.see(frame, s, ins, noise, torch.zeros(1, device='cuda'))[1].float()
            x, taus = noise, torch.linspace(0, 1, 11, device='cuda')
            for i in range(10):
                x_hat = net.see(frame, s, ins, x, taus[i : i + 1])[1].float()
                x = x + (x_hat - x) / (1 - taus[i]) * (taus[i + 1] - taus[i])
            top = frame[0, ..., :16]
            rows += [strip(top, real[0]), strip(top, one[0]), strip(top, x[0])]
        a = kept[ep]
        mse = [((p - real) ** 2).mean((0, 2, 3, 4)).cpu().numpy().round(2).tolist() for p in (one, x)]
        captions.append(f'ep {ep} t {t}: {a["prompt"]} (named slot {a["target_slot"]})  per-frame latent mse '
                        f'one-shot {mse[0]} sampled {mse[1]}')
iio.imwrite(run / 'future_frames.png', np.concatenate(rows, 0))
(run / 'future_frames.txt').write_text(
    f'rows per (episode, t): real / one-shot / sampled; columns: present, then +{FUTURE}\n' + '\n'.join(captions) + '\n')
print('\n'.join(captions))
