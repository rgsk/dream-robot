"""How much is lost going from real to dreamed z? Offline, val demos (every 4th row).

Each z-WAM dreams chunk 1 (10 steps, as on the robot) from the now frame + joints; its 2 z notes
go into decoders/z.pt (+ joints) -> 16 actions. Scored against the real chunk (std units),
next to the same decoder fed the real z, and the bc decoder (no future).

    .venv/bin/python experiments/lam/scripts/dreamed_z_check.py experiments/lam/wamc_l3 experiments/lam/wamc_l3_z_s1 experiments/lam/wamc_l3_z_s2
"""

import sys
from pathlib import Path

import numpy as np
import torch

from decoder import H, Decoder
from lam import NAME, REPO, load, split
from wam_c import K, M, WAMC, mask, tokens

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
    train_rows = ~np.isin(episode, eps['val'])

    rows = torch.tensor(np.flatnonzero(np.isin(episode, eps['val']))[::4], device=dev)
    ahead = lambda k: torch.minimum(rows + k, last[rows])
    st = (state[rows] - s_mean) / s_std
    acts = (action[torch.minimum(rows[:, None] + torch.arange(H, device=dev), last[rows][:, None])] - a_mean) / a_std

    dec = {}
    for kind in ['z', 'bc']:
        ck = torch.load(REPO / 'experiments/lam/decoders' / f'{kind}.pt', weights_only=False)
        dec[kind] = Decoder(kind).to(dev).eval()
        dec[kind].load_state_dict(ck['state_dict'])
    err = lambda a: ((a - acts).square()[..., :7].mean().item(), (a - acts).square()[..., 7].mean().item())

    with torch.no_grad():
        Z = torch.load(REPO / 'experiments/lam/wamc_l3/z.pt').to(dev)
        Zn = (Z - Z[train_rows].mean(0)) / Z[train_rows].std(0)
        true_z = torch.stack([Zn[rows], Zn[ahead(8)]], 1)
        print(f'{len(rows)} val rows')
        print('decoder, real z     arm %.3f  gripper %.3f' % err(dec['z'](st, z=true_z)))
        print('decoder bc, no z    arm %.3f  gripper %.3f' % err(dec['bc'](st, lat[rows][:, None])))
        for run in map(Path, sys.argv[1:]):
            net = WAMC(8).to(dev).eval()
            net.load_state_dict(torch.load(run / 'model.pt', weights_only=False)['state_dict'])
            torch.manual_seed(0)
            dz, own_a = [], []
            for sl in torch.arange(len(rows), device=dev).split(32):
                B = len(sl)
                seen = torch.zeros(B, 1 + M * K, 64, 16, device=dev)
                seen[:, 0] = tokens(lat[rows[sl]])
                s4 = st[sl][:, None].expand(B, M, -1)
                x_f = torch.randn(B, M, K * 64, 16, device=dev)
                x_z = torch.randn(B, M, K, 8, device=dev)
                x_a = torch.randn(B, M, H, 8, device=dev)
                tau = torch.zeros(B, M, device=dev)
                for i in range(10):
                    tau[:, 0] = i / 10
                    v_f, v_z, v_a = net(seen, s4, x_f, x_z, x_a, tau, ~mask.to(dev))
                    x_f[:, 0] += v_f[:, 0] / 10
                    x_z[:, 0] += v_z[:, 0] / 10
                    x_a[:, 0] += v_a[:, 0] / 10
                dz.append(x_z[:, 0])
                own_a.append(x_a[:, 0])
            dz, own_a = torch.cat(dz), torch.cat(own_a)
            z_mse = (dz - true_z).square().mean().item()
            print(f'{run.name:14s} dreamed z err {z_mse:.3f}   decoder(dreamed z) arm %.3f gripper %.3f'
                  % err(dec['z'](st, z=dz)) + '   WAM own actions arm %.3f gripper %.3f' % err(own_a))


if __name__ == '__main__':
    main()
