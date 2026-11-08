"""Reliance probe for option b (wam_c_b.py): does the head's arm still read the dream?

Pass 1: the 10-step joint loop on val rows (every 4th), recording the trunk's clean guess of the dream
(x_hat_z, x_hat_f) that the head reads at each step. Pass 2: the head alone, re-run with the same noise
on its own recorded dream, or another row's (swap future / z only / frames only).
Same numbers as heads_reliance.py (jzf on frozen WAM-C s0: future 0.385/0.658, z 0.038/0.356, f 0.283/0.330).

    .venv/bin/python experiments/lam/scripts/b_reliance.py experiments/lam/wamb_grad_s0
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lam import NAME, REPO, load, split
from wam_c import H, K, M, WAMC, at_acts, mask, tokens
from wam_c_heads import Head

dev = 'cuda'


def main():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f'dream_robot/{NAME}', root=REPO / 'data/robosuite' / NAME)
    action = torch.tensor(np.asarray(ds.hf_dataset['action'], dtype=np.float32), device=dev)
    state_np, episode, _ = load(NAME)
    state = torch.tensor(state_np, device=dev)
    eps = split()
    lab10 = torch.tensor(np.isin(episode, eps['labelled']), device=dev)
    a_mean, a_std = action[lab10].mean(0), action[lab10].std(0)
    s_mean, s_std = state[lab10].mean(0), state[lab10].std(0)
    lat = torch.load(REPO / 'cache/lam/vae_lat.pt', weights_only=False)['lat'].to(dev)
    last = torch.tensor(np.searchsorted(episode, episode, side='right') - 1, device=dev)
    rows = torch.tensor(np.flatnonzero(np.isin(episode, eps['val']))[::4], device=dev)
    st = (state[rows] - s_mean) / s_std
    acts = (action[torch.minimum(rows[:, None] + torch.arange(H, device=dev), last[rows][:, None])] - a_mean) / a_std
    perm = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).to(dev)
    is_act = torch.zeros(len(mask), dtype=torch.bool)
    is_act[at_acts.flatten()] = True
    blocked = ~mask.to(dev) | is_act.to(dev)[None, :]

    for run in map(Path, sys.argv[1:]):
        net = nn.ModuleDict(dict(trunk=WAMC(8), head=Head(0, 1, 1))).to(dev).eval()
        ck = torch.load(run / 'model.pt', weights_only=False)
        net.load_state_dict(ck['state_dict'])
        clean = ck['args'].get('dream_steps', 0)  # head reads the finished dream at every action step
        hz, hf = torch.zeros(10, len(rows), K, 8, device=dev), torch.zeros(10, len(rows), K * 64, 16, device=dev)
        torch.manual_seed(0)
        with torch.no_grad():
            for sl in torch.arange(len(rows), device=dev).split(8):
                B = len(sl)
                seen = torch.zeros(B, 1 + M * K, 64, 16, device=dev)
                seen[:, 0] = tokens(lat[rows[sl]])
                s4 = st[sl][:, None].expand(B, M, -1)
                x_f = torch.randn(B, M, K * 64, 16, device=dev)
                x_z = torch.randn(B, M, K, 8, device=dev)
                tau = torch.zeros(B, M, device=dev)
                for i in range(10):
                    tau[:, 0] = i / 10
                    v_f, v_z, _ = net['trunk'](seen, s4, x_f, x_z, torch.zeros(B, M, H, 8, device=dev), tau, blocked)
                    hz[i, sl] = x_z[:, 0] + (1 - i / 10) * v_z[:, 0]
                    hf[i, sl] = x_f[:, 0] + (1 - i / 10) * v_f[:, 0]
                    x_f[:, 0] += v_f[:, 0] / 10
                    x_z[:, 0] += v_z[:, 0] / 10
                if clean:
                    hz[:, sl], hf[:, sl] = x_z[:, 0], x_f[:, 0]

            def sample(z_src, f_src):  # head loop with recorded dreams; z_src/f_src = row order
                torch.manual_seed(1)
                x = torch.randn(len(rows), H, 8, device=dev)
                for i in range(10):
                    t = torch.full((len(rows),), i / 10, device=dev)
                    x += net['head'](st, None, hz[i, z_src], hf[i, f_src], x, t) / 10
                return x

            ar = torch.arange(len(rows), device=dev)
            own = sample(ar, ar)
            fmt = lambda e: f'{e[..., :7].mean():.3f}/{e[..., 7].mean():.3f}'
            d = lambda a: fmt((a - own).square())
            print(f'{run.name}: {len(rows)} val rows. change when swapped, arm / gripper')
            print(f'  future {d(sample(perm, perm))}   z {d(sample(perm, ar))}   f {d(sample(ar, perm))}'
                  f'   (own err {fmt((own - acts).square())})', flush=True)


if __name__ == '__main__':
    main()
