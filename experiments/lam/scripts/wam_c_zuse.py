"""How much does a trained WAM-C lean on z, or on its dreamed frames (--frames)? Chunk-1 actions dreamed on val demos three ways:

    own    z denoised freely with frames + actions (what the robot does)
    true   z clamped to the LAM's notes of the real future (inpainting: x_z kept on the line
           noise -> true z at every step)
    wrong  z clamped to another sample's true notes

Scores: action MSE vs the real chunk (std units, arm 7 / gripper 1), and own-z error vs true z.
If wrong z moves the actions a lot, the policy reads z; if true z beats own z, better z would help.

    .venv/bin/python experiments/lam/scripts/wam_c_zuse.py experiments/lam/wamc_l3 experiments/lam/wamc_l3_z_s1 experiments/lam/wamc_l3_z_s2
"""

import sys
from pathlib import Path

import numpy as np
import torch

from lam import NAME, REPO, load, split
from wam_c import H, K, M, WAMC, is_z, mask, tokens

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
    first, lens = np.searchsorted(episode, np.arange(100)), np.bincount(episode)
    train_rows = ~np.isin(episode, eps['val'])

    # 512 val samples, fixed: now anywhere in the demo (to_end, as training)
    rng = np.random.default_rng(7)
    e = rng.choice(eps['val'], 512)
    last = (first[e] + lens[e] - 1)[:, None]
    now = first[e] + rng.integers(0, lens[e])
    f_rows = torch.tensor(np.minimum(now[:, None] + 8 * np.arange(M * K + 1), last), device=dev)
    a_rows = torch.tensor(np.minimum(now[:, None] + np.arange(H), last), device=dev)
    real_a = (action[a_rows] - a_mean) / a_std  # [B, 16, 8]

    frames_mode = '--frames' in sys.argv
    true_f = tokens(lat[f_rows[:, 1:1 + K]]).flatten(1, 2)  # [B, 128, 16] chunk 1's real frames
    for run in map(Path, [a for a in sys.argv[1:] if not a.startswith('--')]):
        ck = torch.load(run / 'model.pt', weights_only=False)
        net = WAMC(8).to(dev).eval()
        net.load_state_dict(ck['state_dict'])
        no_z = ck['args'].get('no_z', False)
        blocked = ~mask.to(dev) | (is_z.to(dev)[None, :] if no_z else False)
        Z = torch.load(run / 'z.pt').to(dev)
        Zn = (Z - Z[train_rows].mean(0)) / Z[train_rows].std(0)
        true_z = Zn[f_rows[:, :K]]  # [B, 2, 8]: t->t+8, t+8->t+16
        res = {}
        for mode in ['own', 'true', 'wrong']:
            torch.manual_seed(0)
            outs, zs = [], []
            for sl in torch.arange(512, device=dev).split(32):
                B = len(sl)
                seen = torch.zeros(B, 1 + M * K, 64, 16, device=dev)
                seen[:, 0] = tokens(lat[f_rows[sl, 0]])
                st = ((state[a_rows[sl, 0]] - s_mean) / s_std)[:, None].expand(B, M, -1)
                x_f = torch.randn(B, M, K * 64, 16, device=dev)
                x_z = torch.randn(B, M, K, 8, device=dev)
                x_a = torch.randn(B, M, H, 8, device=dev)
                noise_z, noise_f = x_z[:, 0].clone(), x_f[:, 0].clone()
                src = true_f if frames_mode else true_z
                given = src[sl] if mode == 'true' else src[sl.roll(1)] if mode == 'wrong' else None
                tau = torch.zeros(B, M, device=dev)
                with torch.no_grad():
                    for i in range(10):
                        tau[:, 0] = i / 10
                        if given is not None and frames_mode:
                            x_f[:, 0] = (1 - i / 10) * noise_f + i / 10 * given
                        elif given is not None:
                            x_z[:, 0] = (1 - i / 10) * noise_z + i / 10 * given
                        v_f, v_z, v_a = net(seen, st, x_f, x_z, x_a, tau, blocked)
                        x_f[:, 0] += v_f[:, 0] / 10
                        x_z[:, 0] += v_z[:, 0] / 10
                        x_a[:, 0] += v_a[:, 0] / 10
                outs.append(x_a[:, 0])
                zs.append(x_z[:, 0])
            a, z = torch.cat(outs), torch.cat(zs)
            err = (a - real_a).square()
            res[mode] = (err[..., :7].mean().item(), err[..., 7].mean().item())
            if mode == 'own':
                z_err = (z - true_z).square().mean().item()
        what = 'frames' if frames_mode else 'z'
        print(f'{run.name:16s} [{what} clamped] own z vs true z: {z_err:.3f}   action MSE arm / gripper:  '
              + '   '.join(f'{m} {v[0]:.3f} / {v[1]:.3f}' for m, v in res.items()))


if __name__ == '__main__':
    main()
