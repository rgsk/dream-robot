"""Re-check saved LAMs without retraining: fair "is z used" numbers + a nonlinear probe.

    drawing on val pairs (forward, gap 8), / copy:  true z | average z | another pair's z
    probe R^2 on val demos: straight line (as lam.py) and nearest neighbours (k=20: average the
    true motion of the 20 most similar train z's), fit on 90 demos and on 3

    .venv/bin/python experiments/lam/scripts/lam_eval.py experiments/lam/z8 experiments/lam/z8_det_rev
"""

import sys
from pathlib import Path

import numpy as np
import torch

from lam import GAP, LAM, NAME, load, ridge_r2, split


def knn_r2(x_tr, y_tr, x_te, y_te, k=20):
    m, s = x_tr.mean(0), x_tr.std(0) + 1e-6
    a, b = torch.tensor((x_tr - m) / s), torch.tensor((x_te - m) / s)
    near = torch.cdist(b, a).topk(k, largest=False).indices.numpy()
    pred = y_tr[near].mean(1)
    return 1 - ((y_te - pred) ** 2).sum(0) / ((y_te - y_te.mean(0)) ** 2).sum(0)


def main():
    state, episode, top = load(NAME)
    eps = split()
    ok = np.zeros(len(episode), bool)
    ok[:-GAP] = episode[:-GAP] == episode[GAP:]
    train_rows = np.flatnonzero(ok & ~np.isin(episode, eps['val']))
    val_rows = np.flatnonzero(ok & np.isin(episode, eps['val']))
    frames = lambda rows: torch.from_numpy(np.stack([top[r] for r in rows])).cuda().float() / 255
    motion = lambda r: state[r + GAP] - state[r]
    tr = train_rows[::4]
    lab3 = train_rows[np.isin(episode[train_rows], eps['labelled'][:3])]

    for run in map(Path, sys.argv[1:]):
        ck = torch.load(run / 'lam.pt', weights_only=False)
        net = LAM(ck['args']['z_dim']).cuda().eval()
        net.load_state_dict(ck['state_dict'])

        @torch.no_grad()
        def z_of(rows):
            return torch.cat([net.encode(frames(r), frames(r + GAP))[0].cpu()
                              for r in np.array_split(rows, max(1, len(rows) // 256))]).numpy()

        with torch.no_grad():
            r = val_rows[np.random.default_rng(1).choice(len(val_rows), 512, replace=False)]
            f0, f1 = frames(r), frames(r + GAP)
            mu = net.encode(f0, f1)[0]
            err = lambda z: (net.decode(f0, z) - f1).pow(2).mean().item() / (f0 - f1).pow(2).mean().item()
            draw = err(mu), err(mu.mean(0).expand_as(mu)), err(mu.roll(1, 0))
        print(f'{run.name}: drawing/copy  true z {draw[0]:.3f}  average z {draw[1]:.3f}  wrong z {draw[2]:.3f}')

        zv = z_of(val_rows)
        for tag, rows, k in [('90 demos', tr, 20), ('3 demos', lab3, 10)]:
            zt = z_of(rows)
            lin = ridge_r2(zt, motion(rows), zv, motion(val_rows))
            nn_ = knn_r2(zt, motion(rows), zv, motion(val_rows), k)
            print(f'  fit on {tag:8s}  line: arm {lin[:7].mean():.3f} gripper {lin[7]:.3f}'
                  f'   neighbours: arm {nn_[:7].mean():.3f} gripper {nn_[7]:.3f}')


if __name__ == '__main__':
    main()
