# Is "which cube is the named colour" learnable from the VAE latents? T5 scenes from env resets (first frame,
# [top | wrist] through vae.WEIGHTS), target = the named cube's slot (0/1), input = latent + the colour one-hot
# tiled as 4 extra channels. A small conv net, trained on 135 scenes (the demo count) and on 1000, scored on
# 300 held-out scenes. Chance 50%. Latents cached in scratch/verne07/t5_scenes.npz.
# usage: python experiments/verne/scripts/t5_colour_binding.py
import numpy as np
import torch
from torch import nn

from dream_robot.policies.verne import rollouts, vae
from dream_robot.policies.verne.data import REPO, WORDS, frame_of

cache = REPO / 'scratch/verne07/t5_scenes.npz'
if not cache.exists():
    enc, env = vae.load(vae.WEIGHTS), rollouts.make_env('t5')
    Z, C, Y = [], [], []
    for seed in range(5000, 6300):
        obs = env.reset(seed=seed)
        Z.append(frame_of(obs, enc)[0].cpu().numpy())
        C.append(WORDS.index(env.instruction.color))
        Y.append(env.instruction.target_slot)
    env.close()
    np.savez(cache, z=np.stack(Z), c=np.array(C), y=np.array(Y))
d = np.load(cache)
z, c, y = torch.tensor(d['z']).cuda(), torch.tensor(d['c']).cuda(), torch.tensor(d['y']).cuda()
print(f'{len(z)} scenes, latent {tuple(z.shape[1:])}, named slot 1 share {y.float().mean():.2f}')


def inputs(ids):
    oh = torch.nn.functional.one_hot(c[ids], 4).float()[:, :, None, None].expand(-1, 4, 16, 32)
    return torch.cat([z[ids].float(), oh], 1)


def run(n_train, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Conv2d(8, 64, 3, padding=1), nn.GELU(), nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2),
    ).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    tr, va = torch.arange(n_train).cuda(), torch.arange(1000, 1300).cuda()
    for step in range(3000):
        ids = tr[torch.randint(0, n_train, (64,), device='cuda')]
        loss = nn.functional.cross_entropy(net(inputs(ids)), y[ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = lambda ids: (net(inputs(ids)).argmax(1) == y[ids]).float().mean().item()
        return acc(tr), acc(va)


for n in (135, 1000):
    tr_acc, va_acc = run(n)
    print(f'{n:4d} training scenes: train {tr_acc:.0%}, held-out {va_acc:.0%}', flush=True)
