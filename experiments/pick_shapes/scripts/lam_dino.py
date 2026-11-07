"""LAM on wrist DINOv2 features (experiments/lam's recipe, T1 -> pick_shapes).

    encoder  (features t, features t+8)  [8x8x384 each] -> z (8 numbers)
    decoder  (features t, z) -> features t+8, as  t + change   (standardised feature units)

Recipe from experiments/lam/notes.md: deterministic z (no noise, no KL), half the pairs reversed.
Trained on every clip the policy may see as video: apple n30 minus its 5 val demos, apple video
n50 x2, fruit video n100 + n200 minus the last 20 fruit demos (held out for the probe).
Never onion / tomato.

Checks (actions = answer key only):
- drawing / copy on held-out pairs with the right z, the average z, another pair's z
- probe: z -> true motion over the 8 steps (state[t+8] - state[t]), nearest neighbours (k=20)
  and straight line, fit on train pairs, scored on apple val demos and on held-out fruit demos;
  control = untrained encoder
At the end z for every row of every recording -> <out>/z_<name>.npy (pair t, min(t+8, demo end)).

    PYTHONUNBUFFERED=1 .venv/bin/python experiments/pick_shapes/scripts/lam_dino.py \
        --out experiments/pick_shapes/lam/z8
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from uwm_cam import REPO
from uwm_dino import load

sys.path.insert(0, str(REPO / 'experiments/lam/scripts'))
from lam import ridge_r2  # noqa: E402
from lam_eval import knn_r2  # noqa: E402

GAP, C, Z = 8, 384, 8
NAMES = ['pick_apple_rung0_n30', 'pick_apple_rung0_video_n50', 'pick_apple_rung0_video2_n50',
         'pick_fruit_video_n100', 'pick_fruit_video2_n200']
dev = 'cuda'


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


class LAMf(nn.Module):
    def __init__(self, w=256):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(2 * C, w, 1), block(w, w), block(w, w))
        self.enc_out = nn.Linear(w, Z)
        self.dec_in = nn.Sequential(nn.Conv2d(C, w, 1), block(w, w))
        self.film = nn.Linear(Z, 2 * w)
        self.dec_out = nn.Sequential(block(w, w), nn.Conv2d(w, C, 1))

    def note(self, g0, g1):  # [B, 384, 8, 8] each -> [B, Z]
        return self.enc_out(self.enc(torch.cat([g0, g1], 1)).mean((2, 3)))

    def redraw(self, g0, z):
        scale, shift = self.film(z)[..., None, None].chunk(2, 1)
        return g0 + self.dec_out(self.dec_in(g0) * (1 + scale) + shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # all recordings stacked; episodes renumbered so they never collide
    grid, state, episode, src = [], [], [], []
    for i, name in enumerate(NAMES):
        d = load(name)
        grid.append(d['grid'])
        state.append(d['state'])
        episode.append(d['episode'] + (episode[-1].max() + 1 if episode else 0))
        src.append(np.full(len(d['episode']), i))
    grid, state, episode, src = map(np.concatenate, (grid, state, episode, src))
    ok = np.zeros(len(episode), bool)
    ok[:-GAP] = episode[:-GAP] == episode[GAP:]

    apple_eps = np.unique(episode[src == 0])
    apple_val = apple_eps[np.random.default_rng(0).permutation(len(apple_eps))[:5]]  # uwm_dino's split
    fruit_eps = np.unique(episode[src == 4])
    fruit_held = fruit_eps[-20:]
    held = np.isin(episode, np.concatenate([apple_val, fruit_held]))
    rows = {'train': np.flatnonzero(ok & ~held), 'apple_val': np.flatnonzero(ok & np.isin(episode, apple_val)),
            'fruit_held': np.flatnonzero(ok & np.isin(episode, fruit_held))}
    print({k: len(v) for k, v in rows.items()}, flush=True)

    G = torch.tensor(grid, device=dev)  # f16 [N, 8, 8, 384], ~1.1 GB
    tr = G[torch.tensor(rows['train'][::7], device=dev)].float()
    g_mean, g_std = tr.mean((0, 1, 2)), tr.std((0, 1, 2)) + 1e-4
    del tr
    feats = lambda r: ((G[torch.as_tensor(r, device=dev)].float() - g_mean) / g_std).permute(0, 3, 1, 2)

    net = LAMf().to(dev)
    print(f'{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params', flush=True)

    @torch.no_grad()
    def z_of(r):
        net.eval()
        out = torch.cat([net.note(feats(c), feats(np.minimum(c + GAP, last[c]))) for c in np.array_split(r, max(1, len(r) // 512))])
        net.train()
        return out.cpu().numpy()

    last = np.searchsorted(episode, episode, side='right') - 1
    motion = lambda r: state[r + GAP] - state[r]

    def probe(tag):
        fit = rows['train'][::5]
        zf = z_of(fit)
        res = {}
        for split in ['apple_val', 'fruit_held']:
            zv, r = z_of(rows[split]), rows[split]
            nn_ = knn_r2(zf, motion(fit), zv, motion(r))
            lin = ridge_r2(zf, motion(fit), zv, motion(r))
            res[split] = {'nn_arm': float(nn_[:7].mean()), 'nn_gripper': float(nn_[7]),
                          'line_arm': float(lin[:7].mean()), 'line_gripper': float(lin[7])}
            print(f"probe [{tag}] {split:10s} neighbours: arm {res[split]['nn_arm']:.3f} gripper "
                  f"{res[split]['nn_gripper']:.3f}   line: arm {res[split]['line_arm']:.3f} gripper "
                  f"{res[split]['line_gripper']:.3f}", flush=True)
        return res

    @torch.no_grad()
    def validate():
        r = np.random.default_rng(1).choice(np.concatenate([rows['apple_val'], rows['fruit_held']]), 512, replace=False)
        g0, g1 = feats(r), feats(r + GAP)
        z = net.note(g0, g1)
        copy = (g0 - g1).square().mean()
        return [((net.redraw(g0, x) - g1).square().mean() / copy).item() for x in (z, z.mean(0).expand_as(z), z.roll(1, 0))]

    history = {'probe_untrained': probe('untrained'), 'log': []}
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr / 100)
    rng, t0 = np.random.default_rng(args.seed), time.time()
    for step in range(1, args.steps + 1):
        r0 = rng.choice(rows['train'], args.batch)
        r1 = r0 + GAP
        back = rng.random(args.batch) < 0.5  # reversed pairs: frame t alone can't tell the direction
        r0, r1 = np.where(back, r1, r0), np.where(back, r0, r1)
        g0, g1 = feats(r0), feats(r1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = (net.redraw(g0, net.note(g0, g1)).float() - g1).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0:
            v = validate()
            history['log'].append([step, loss.item(), *v])
            print(f'step {step:5d}  train {loss.item():.4f}   held-out / copy: right z {v[0]:.3f}  '
                  f'average z {v[1]:.3f}  wrong z {v[2]:.3f}  ({time.time() - t0:.0f}s)', flush=True)
    torch.save({'state_dict': net.state_dict(), 'g_mean': g_mean.cpu(), 'g_std': g_std.cpu(),
                'args': vars(args) | {'out': str(args.out)}}, args.out / 'lam.pt')
    history['probe'] = probe('trained')
    (args.out / 'history.json').write_text(json.dumps(history, indent=1))
    for i, name in enumerate(NAMES):
        np.save(args.out / f'z_{name}.npy', z_of(np.flatnonzero(src == i)))
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {args.out}')


if __name__ == '__main__':
    main()
