"""Future-feature predictor for variant B (notes.md, step 10): DINOv2 features now -> at t+16.

    in   features of 4 wrist frames (t-12, t-8, t-4, t), 8x8 x 384 each (dino_cache.py)
    out  features of frame t+16, 8x8 x 384
    loss squared error against the real t+16 features (cached) -- no actions anywhere

Features are compared per channel standardised (mean/std over the training frames), so every
channel counts the same. The output is "now + change": the change head starts at zero, so step 1
equals the copy baseline ("the future looks like now") and any gain over it is visible directly.

Which clips it learns from is the whole point: --video 0 trains on the 10 labelled clips only
(arm B); --video 15 adds the 15 video-only clips (arm B+video). Same split as uwm_cam / uwm_dino.
--more-video adds clips from other recordings (raw only, cached by dino_cache.py), all video-only;
--more-n uses its first n (the video-amount curve: 15, 15 + 25, 15 + 50, ...). Validation stays the
same 5 held-out episodes of --data.

    .venv/bin/python experiments/pick_shapes/scripts/dino_predict.py --labelled 10 --video 0 \
        --out experiments/pick_shapes/uwm_dino/pred_l10_v0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from uwm_cam import AHEAD, GRID, PAST, rows_at
from uwm_dino import C, load

D = 256                            # wider than the action head: it has to rebuild 384-dim features


class Predictor(nn.Module):
    """Features of the seen frames -> features of frame t+16 (raw DINOv2 units)."""

    def __init__(self, f_mean: Tensor, f_std: Tensor, layers: int = 4):
        super().__init__()
        self.register_buffer('f_mean', f_mean)           # [384], per channel, training frames
        self.register_buffer('f_std', f_std)
        self.feat_in = nn.Linear(C, D)
        self.when = nn.Embedding(12 + AHEAD + 1, D)      # row = step offset + 12
        self.where = nn.Embedding(GRID * GRID, D)
        self.query = nn.Parameter(torch.zeros(D))        # the 64 future slots, before position
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, 0.0, 'gelu', batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Linear(D, C)
        nn.init.zeros_(self.out.weight)                  # start as the copy baseline
        nn.init.zeros_(self.out.bias)

    def forward(self, feats: Tensor) -> Tensor:
        # feats [B, 4, 64, 384] raw -> predicted t+16 [B, 64, 384], standardised units
        z = (feats - self.f_mean) / self.f_std
        when, where = self.when.weight, self.where.weight
        seen = self.feat_in(z) + when[[o + 12 for o in PAST]][:, None] + where
        future = (self.query + when[12 + AHEAD] + where).expand(len(z), -1, -1)
        out = self.body(torch.cat([seen.flatten(1, 2), future], 1))
        return z[:, -1] + self.out(out[:, -GRID * GRID:])      # now + change

    @torch.no_grad()
    def predict(self, feats: Tensor) -> Tensor:
        # raw in, raw out: what the B action head reads, in the same units as the seen features
        return self(feats).float() * self.f_std + self.f_mean


def picture(now: np.ndarray, pred: np.ndarray, true: np.ndarray, steps: list[int], out: Path) -> None:
    # rows: now / predicted t+16 / real t+16, as the first 3 PCA directions in RGB
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    flat = true.reshape(-1, true.shape[-1])
    mu = flat.mean(0)
    _, _, vt = np.linalg.svd(flat - mu, full_matrices=False)
    proj = lambda f: (f - mu) @ vt[:3].T
    lo, hi = np.percentile(proj(flat), [2, 98], axis=0)
    rgb = lambda f: np.clip((proj(f) - lo) / (hi - lo), 0, 1).reshape(GRID, GRID, 3)
    fig, axes = plt.subplots(3, len(steps), figsize=(1.8 * len(steps), 5.6))
    for j, t in enumerate(steps):
        for i, (f, name) in enumerate([(now, 'now'), (pred, 'predicted +16'), (true, 'real +16')]):
            axes[i, j].imshow(rgb(f[j]), interpolation='nearest')
            axes[i, j].set_title(f'{name}\nstep {t}' if i == 0 else name, fontsize=8)
            axes[i, j].axis('off')
    fig.tight_layout()
    fig.savefig(out, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='pick_apple_rung0_n30')
    ap.add_argument('--labelled', type=int, default=10)
    ap.add_argument('--video', type=int, default=0, help='video-only clips, after the labelled ones')
    ap.add_argument('--more-video', nargs='*', default=[], help='other dino-cached recordings, every clip video-only')
    ap.add_argument('--more-n', type=int, help='use only the first n clips of each (default all)')
    ap.add_argument('--val', type=int, default=5)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--save-every', type=int, default=2500)
    ap.add_argument('--lr', type=float, default=1e-3, help='peak lr; cosine-decays to lr/100')
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = 'cuda'
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data)
    eps = np.unique(d['episode'])
    order = np.random.default_rng(0).permutation(eps)       # the split uwm_cam.py uses
    val_eps = order[:args.val]
    lab_eps = order[args.val:args.val + args.labelled]
    vid_eps = order[args.val + args.labelled:args.val + args.labelled + args.video]
    assert len(lab_eps) == args.labelled and len(vid_eps) == args.video, 'not enough episodes'
    train = np.isin(d['episode'], np.concatenate([lab_eps, vid_eps]))
    grid = d['grid'].reshape(len(d['grid']), GRID * GRID, C).astype(np.float32)
    n_more = 0
    for name in args.more_video:       # appended after --data, episode ids shifted past the last
        m = load(name)
        keep = m['episode'] < (args.more_n if args.more_n is not None else m['episode'].max() + 1)
        n = len(np.unique(m['episode'][keep]))
        assert args.more_n is None or n == args.more_n, f'only {n} clips in {name}'
        n_more += n
        d['episode'] = np.concatenate([d['episode'], m['episode'][keep] + d['episode'].max() + 1])
        grid = np.concatenate([grid, m['grid'][keep].reshape(-1, GRID * GRID, C).astype(np.float32)])
        train = np.concatenate([train, np.ones(keep.sum(), bool)])
    f_mean = grid[train].mean((0, 1))
    f_std = grid[train].std((0, 1)) + 1e-4
    g = lambda x: torch.as_tensor(x, device=dev)
    FE = g(grid)
    PASTR, AFTER = g(rows_at(d['episode'], PAST)), g(rows_at(d['episode'], [AHEAD])[:, 0])
    train_rows = g(np.flatnonzero(train))
    val_rows = g(np.flatnonzero(np.isin(d['episode'], val_eps)))

    net = Predictor(g(f_mean), g(f_std)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 100)
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema)) if args.ema else None
    model = lambda: ema.module if ema is not None else net
    print(f'{args.data}: predictor on labelled {len(lab_eps)} + video-only {len(vid_eps)} + {n_more} more eps, '
          f'val {len(val_eps)}; '
          f'{len(train_rows)} train rows; {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params', flush=True)

    def loss(rows: Tensor, m: nn.Module | None = None) -> Tensor:
        target = (FE[AFTER[rows]] - net.f_mean) / net.f_std
        return ((m or net)(FE[PASTR[rows]]).float() - target).pow(2).mean()

    with torch.no_grad():      # copy baseline on val: predict "t+16 looks like t"
        z = lambda rows: (FE[rows] - net.f_mean) / net.f_std
        copy_val = (z(PASTR[val_rows][:, -1]) - z(AFTER[val_rows])).pow(2).mean().item()
    print(f'copy baseline (future = now), val: {copy_val:.4f}', flush=True)

    history, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        rows = train_rows[torch.randint(0, len(train_rows), (args.batch,), device=dev)]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lt = loss(rows)
        opt.zero_grad(set_to_none=True)
        lt.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update_parameters(net)
        if step % 500 == 0 or step == 1:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                lv = loss(val_rows, m=model()).item()
            history.append({'step': step, 'train': lt.item(), 'val': lv, 'copy_val': copy_val})
            print(f'step {step:5d}  train {lt.item():.4f}  val {lv:.4f}  val/copy {lv / copy_val:.3f}  '
                  f"lr {opt.param_groups[0]['lr']:.1e}  ({time.time() - t0:.0f}s)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({'state_dict': model().state_dict(), 'step': step, 'kind': 'dino_pred',
                        'args': vars(args) | {'out': str(args.out)},
                        'split': {'val': val_eps.tolist(), 'labelled': lab_eps.tolist(), 'video': vid_eps.tolist(),
                                  'more_video': n_more}},
                       args.out / f'step{step}.pt')
    (args.out / 'history.json').write_text(json.dumps(history, indent=1))

    # one held-out episode: now / predicted / real t+16, from the start of the reach to the lift
    ep_rows = np.flatnonzero(d['episode'] == val_eps[0])
    steps = list(np.linspace(0, len(ep_rows) - AHEAD - 1, 7).round().astype(int))
    rows = g(ep_rows[steps])
    pred = model().predict(FE[PASTR[rows]]).cpu().numpy()
    picture(grid[ep_rows[steps]], pred, FE[AFTER[rows]].cpu().numpy(), steps, args.out / 'predicted.png')
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {args.out}')


if __name__ == '__main__':
    main()
