"""Latent-action model (LAM) on T1 top-camera video: name the motion between two frames, no actions.

    encoder  (frame t, frame t+8) -> z (8 numbers)          "what changed"
    decoder  (frame t, z)         -> frame t+8              predicted as frame t + change

z is squeezed to 8 numbers (+ a small KL pull toward 0), so it can't carry the whole next
frame, only the change. Trained on all 90 train demos (dreamzero split: 10 val held out);
the actions on disk are never read in training.

Probe (the check, after training): the sim recorded the true robot motion for every demo, used
here only as an answer key. A straight-line fit z -> true motion over the same 8 steps
(state[t+8] - state[t]: 7 joints + gripper), fit on train demos, scored on the 10 val demos as
R^2 (1 = z tells the motion exactly, 0 = no better than guessing the average). Control: the same
probe on an untrained encoder's z.

    PYTHONUNBUFFERED=1 .venv/bin/python experiments/lam/scripts/lam.py --out experiments/lam/z8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

REPO = Path(__file__).resolve().parents[3]
NAME = 'pick_place_cube_n100'
GAP = 8  # steps between the two frames: the WAM's frame spacing (H // K)


def load(name: str) -> tuple[NDArray, NDArray, NDArray]:
    # state [N, 8], episode [N], top [N, 3, 128, 128] uint8; frames decoded once, then cached
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f'dream_robot/{name}', root=REPO / 'data/robosuite' / name)
    state = np.asarray(ds.hf_dataset['observation.state'], dtype=np.float32)
    episode = np.asarray(ds.hf_dataset['episode_index'], dtype=np.int64)
    path = REPO / 'cache/lam' / f'{name}_top.npy'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        top = np.zeros((len(ds), 3, 128, 128), dtype=np.uint8)
        for i in range(len(ds)):
            top[i] = (ds[i]['observation.images.top'] * 255).round().to(torch.uint8).numpy()
        np.save(path, top)
    return state, episode, np.load(path, mmap_mode='r')


def split() -> dict[str, NDArray]:
    # as rough/dreamzero.ipynb (and uwm2): 10 val, 10 labelled, 80 video-only
    val = np.random.default_rng(0).permutation(100)[:10]
    train = np.setdiff1d(np.arange(100), val)
    labelled = np.random.default_rng(1).choice(train, 10, replace=False)
    return {'val': val, 'labelled': labelled, 'video': np.setdiff1d(train, labelled)}


def conv_down(c_in: int) -> nn.Sequential:
    # 128x128 -> 16x16x128, three stride-2 convs
    layers, c = [], c_in
    for c_out in [32, 64, 128]:
        layers += [nn.Conv2d(c, c_out, 4, 2, 1), nn.GroupNorm(8, c_out), nn.SiLU()]
        c = c_out
    return nn.Sequential(*layers, nn.Conv2d(c, 128, 3, 1, 1))


class LAM(nn.Module):
    def __init__(self, z_dim: int = 8):
        super().__init__()
        self.enc = conv_down(6)
        self.enc_out = nn.Linear(128, 2 * z_dim)  # mean, log-variance; after a global average
        self.dec_in = conv_down(3)
        self.film = nn.Linear(z_dim, 2 * 128)  # z scales and shifts every cell of frame t's map
        self.dec_up = nn.Sequential(
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
        )

    def encode(self, f0: Tensor, f1: Tensor) -> tuple[Tensor, Tensor]:
        # frames [B, 3, 128, 128] in [0, 1] -> z mean, log-variance [B, z_dim]
        h = self.enc(torch.cat([f0, f1], 1)).mean((2, 3))
        return self.enc_out(h).chunk(2, -1)

    def decode(self, f0: Tensor, z: Tensor) -> Tensor:
        scale, shift = self.film(z)[..., None, None].chunk(2, 1)
        return f0 + self.dec_up(self.dec_in(f0) * (1 + scale) + shift)


def kl_of(mu: Tensor, logvar: Tensor) -> Tensor:
    '''
    KL( N(μ, σ²) ‖ N(0, 1) ) per dim = ½ (μ² + σ² − log σ² − 1), summed over z, averaged over B.
    '''
    return 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(-1).mean()


def ridge_r2(x_tr: NDArray, y_tr: NDArray, x_te: NDArray, y_te: NDArray, lam: float = 1e-3) -> NDArray:
    '''
    Straight-line fit with a tiny ridge:  W = (XᵀX + λI)⁻¹ XᵀY,  X standardised + a bias column.
    R² per output dim = 1 − Σ(y − ŷ)² / Σ(y − ȳ)², on held-out rows.
    '''
    m, s = x_tr.mean(0), x_tr.std(0) + 1e-6
    bias = lambda x: np.concatenate([(x - m) / s, np.ones((len(x), 1))], 1)
    a, b = bias(x_tr), bias(x_te)
    w = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y_tr)
    resid = ((y_te - b @ w) ** 2).sum(0)
    return 1 - resid / ((y_te - y_te.mean(0)) ** 2).sum(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--z-dim', type=int, default=8)
    ap.add_argument('--beta', type=float, default=1e-5, help='KL weight vs per-pixel MSE')
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max-gap', type=int, default=0, help='train gaps 1..max-gap (0 = always 8)')
    ap.add_argument('--deterministic', action='store_true', help='z = mean, no noise, no KL')
    ap.add_argument('--reverse', action='store_true', help='half the train pairs played backwards')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = 'cuda'
    args.out.mkdir(parents=True, exist_ok=True)

    state, episode, top = load(NAME)
    eps = split()
    # pairs (t, t+8) inside one demo; frames stay on the CPU (the notebook kernel holds the GPU)
    ok = np.zeros(len(episode), bool)
    ok[:-GAP] = episode[:-GAP] == episode[GAP:]
    last = np.searchsorted(episode, episode, side='right') - 1  # each row's demo end
    train_rows = np.flatnonzero(ok & ~np.isin(episode, eps['val']))
    val_rows = np.flatnonzero(ok & np.isin(episode, eps['val']))
    print(f'{NAME}: {len(train_rows)} train pairs (90 demos), {len(val_rows)} val pairs (10 demos)', flush=True)

    frames = lambda rows: torch.from_numpy(np.stack([top[r] for r in rows])).to(dev).float() / 255
    net = LAM(args.z_dim).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 100)
    print(f'{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params', flush=True)

    @torch.no_grad()
    def z_of(rows: NDArray) -> NDArray:
        net.eval()
        out = [net.encode(frames(r), frames(r + GAP))[0].cpu() for r in np.array_split(rows, max(1, len(rows) // 256))]
        net.train()
        return torch.cat(out).numpy()

    def probe(tag: str) -> dict:
        # answer key: true motion over the same 8 steps; fit on train pairs, score on val pairs
        tr = train_rows[:: 4]
        motion = lambda r: state[r + GAP] - state[r]
        r2 = ridge_r2(z_of(tr), motion(tr), z_of(val_rows), motion(val_rows))
        lab3 = train_rows[np.isin(episode[train_rows], eps['labelled'][:3])]
        r2_3 = ridge_r2(z_of(lab3), motion(lab3), z_of(val_rows), motion(val_rows))
        res = {'arm_r2': float(r2[:7].mean()), 'gripper_r2': float(r2[7]),
               'arm_r2_fit_on_3': float(r2_3[:7].mean()), 'gripper_r2_fit_on_3': float(r2_3[7]),
               'per_dim': r2.round(3).tolist()}
        print(f"probe [{tag}]  fit on 90 demos: arm R2 {res['arm_r2']:.3f}  gripper R2 {res['gripper_r2']:.3f}"
              f"   fit on 3: arm {res['arm_r2_fit_on_3']:.3f}  gripper {res['gripper_r2_fit_on_3']:.3f}", flush=True)
        return res

    @torch.no_grad()
    def validate() -> dict:
        r = val_rows[np.random.default_rng(1).choice(len(val_rows), 512, replace=False)]
        f0, f1 = frames(r), frames(r + GAP)
        mu, _ = net.encode(f0, f1)
        copy = (f0 - f1).pow(2).mean().item()
        with_z = (net.decode(f0, mu) - f1).pow(2).mean().item()
        # fair "no note": z's own average (z = 0 is off the map once nothing pulls z to 0)
        mean_z = (net.decode(f0, mu.mean(0).expand_as(mu)) - f1).pow(2).mean().item()
        wrong_z = (net.decode(f0, mu.roll(1, 0)) - f1).pow(2).mean().item()  # another pair's note
        return {'val_over_copy': with_z / copy, 'mean_z_over_copy': mean_z / copy,
                'wrong_z_over_copy': wrong_z / copy}

    history = {'probe_untrained': probe('untrained encoder (control)'), 'log': []}
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    for step in range(1, args.steps + 1):
        r = rng.choice(train_rows, args.batch)
        r1 = r + (rng.integers(1, args.max_gap + 1, args.batch) if args.max_gap else GAP)
        r1 = np.minimum(r1, last[r])  # stay inside the demo
        if args.reverse:
            back = rng.random(args.batch) < 0.5
            r, r1 = np.where(back, r1, r), np.where(back, r, r1)
        f0, f1 = frames(r), frames(r1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            mu, logvar = net.encode(f0, f1)
            z = mu if args.deterministic else mu + (0.5 * logvar).exp() * torch.randn_like(mu)
            mse = (net.decode(f0, z).float() - f1).pow(2).mean()
            kl = kl_of(mu.float(), logvar.float())
        opt.zero_grad(set_to_none=True)
        (mse + (0 if args.deterministic else args.beta) * kl).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            v = validate()
            history['log'].append({'step': step, 'mse': mse.item(), 'kl': kl.item(), **v})
            print(f"step {step:5d}  mse {mse.item():.5f}  kl {kl.item():6.2f}  "
                  f"val/copy {v['val_over_copy']:.3f}  mean z {v['mean_z_over_copy']:.3f}  "
                  f"wrong z {v['wrong_z_over_copy']:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    torch.save({'state_dict': net.state_dict(), 'args': vars(args) | {'out': str(args.out)}}, args.out / 'lam.pt')
    history['probe'] = probe('trained')
    (args.out / 'history.json').write_text(json.dumps(history, indent=1))
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {args.out}')


if __name__ == '__main__':
    main()
