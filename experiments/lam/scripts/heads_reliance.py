"""Reliance probe for the 8 heads: how much do a head's actions move when one input is swapped
for another val row's (same sampling noise)? Offline, val demos every 4th row, one dream per row.

    swap now     : the now frame
    swap future  : dreamed z AND frames together (from another row's dream)
    swap z / f   : one of the two only
Change = mean squared action change (std units), arm / gripper. ~0 = the head ignores that input.

    .venv/bin/python experiments/lam/scripts/heads_reliance.py experiments/lam/wamc_l3
"""

import sys
from pathlib import Path

import numpy as np
import torch

from lam import NAME, REPO, load, split
from wam_c import H, K, M, WAMC, mask, tokens
from wam_c_heads import HEADS, Head

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
    now = tokens(lat[rows])
    acts = (action[torch.minimum(rows[:, None] + torch.arange(H, device=dev), last[rows][:, None])] - a_mean) / a_std
    perm = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).to(dev)

    for run in [Path(a) for a in sys.argv[1:] if a.startswith('experiments')]:
        net = WAMC(8).to(dev).eval()
        net.load_state_dict(torch.load(run / 'model.pt', weights_only=False)['state_dict'])
        torch.manual_seed(0)
        fs, zs = [], []
        with torch.no_grad():
            for sl in torch.arange(len(rows), device=dev).split(32):
                B = len(sl)
                seen = torch.zeros(B, 1 + M * K, 64, 16, device=dev)
                seen[:, 0] = now[sl]
                x_f = torch.randn(B, M, K * 64, 16, device=dev)
                x_z = torch.randn(B, M, K, 8, device=dev)
                x_a = torch.randn(B, M, H, 8, device=dev)
                tau = torch.zeros(B, M, device=dev)
                for i in range(10):
                    tau[:, 0] = i / 10
                    v_f, v_z, v_a = net(seen, st[sl][:, None].expand(B, M, -1), x_f, x_z, x_a, tau, ~mask.to(dev))
                    x_f[:, 0] += v_f[:, 0] / 10
                    x_z[:, 0] += v_z[:, 0] / 10
                    x_a[:, 0] += v_a[:, 0] / 10
                fs.append(x_f[:, 0])
                zs.append(x_z[:, 0])
        f, z = torch.cat(fs), torch.cat(zs)
        print(f'\n{run.name}: {len(rows)} val rows. change when swapped, arm / gripper  (err vs truth: own)')
        names = sys.argv[sys.argv.index('--heads') + 1].split(',') if '--heads' in sys.argv else list(HEADS)
        for name in names:
            use = HEADS[name.split('_')[0]]
            head = Head(*use).to(dev).eval()
            head.load_state_dict(torch.load(run / 'heads' / f'{name}.pt', weights_only=False)['state_dict'])

            def sample(n, zz, ff):
                torch.manual_seed(1)
                return torch.cat([head.sample(st[s], n[s], zz[s], ff[s]) for s in torch.arange(len(rows), device=dev).split(256)])

            own = sample(now, z, f)
            d = lambda a: (a - own).square()
            fmt = lambda e: f'{e[..., :7].mean():.3f}/{e[..., 7].mean():.3f}'
            swaps = {'now': (now[perm], z, f), 'future': (now, z[perm], f[perm]), 'z': (now, z[perm], f), 'f': (now, z, f[perm])}
            want = {'now': use[0], 'future': use[1] or use[2], 'z': use[1] and use[2], 'f': use[1] and use[2]}
            cols = '   '.join(f'{k} {fmt(d(sample(*v)))}' if want[k] else f'{k} {"-":>11s}' for k, v in swaps.items())
            print(f'  {name:5s} {cols}   (own err {fmt((own - acts).square())})', flush=True)


if __name__ == '__main__':
    main()
