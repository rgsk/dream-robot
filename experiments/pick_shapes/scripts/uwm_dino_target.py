"""Future as a training *target*, not an input (notes.md, step 14).

    in   DINOv2 features of 4 wrist frames (t-12, t-8, t-4, t)              -- as arm A
    out  16 actions (flow matching, as A)  +  DINOv2 features of frame t+16 (64 learned slots)

One transformer does both jobs. The future slots are learned queries filled by the network
itself: at test time nothing outside is read, so there is no predicted future to over-trust
(the failure of feeding a predictor's guess, step 12). Video can only help by shaping what the
shared transformer learns.

    labelled rows    action loss + future loss
    video-only rows  future loss only (action tokens pure noise, tau = 0, action loss masked)

Batches: --batch labelled rows + --batch video-only rows per step (the pixel runs' lesson: the
action side then sees exactly what a no-video run sees). The future loss is squared error in
per-channel standardised feature units, predicted as "now + change" (as dino_predict.py).

best.pt = lowest val action (every 500 steps); val future is printed as val/copy.

    .venv/bin/python experiments/pick_shapes/scripts/uwm_dino_target.py --labelled 10 \
        --more-video pick_apple_rung0_video_n50 pick_apple_rung0_video2_n50 \
        --out experiments/pick_shapes/uwm_dino/T_l10_v100
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

from uwm_cam import CHUNK, GRID, PAST, rows_at, targets
from uwm_dino import C, D, DinoAct, load


class DinoTarget(DinoAct):
    """A's network + 64 future slots read out as t+16 features (standardised)."""

    def __init__(self, a_mean: Tensor, a_std: Tensor, f_mean: Tensor, f_std: Tensor, layers: int = 4):
        super().__init__(a_mean, a_std, layers)
        self.register_buffer('f_mean', f_mean)
        self.register_buffer('f_std', f_std)
        self.query = nn.Parameter(torch.zeros(D))
        self.fut_out = nn.Linear(D, C)
        nn.init.zeros_(self.fut_out.weight)              # starts as "the future looks like now"
        nn.init.zeros_(self.fut_out.bias)

    def both(self, feats: Tensor, x_a: Tensor, tau_a: Tensor) -> tuple[Tensor, Tensor]:
        # feats [B, 4, 64, 384] raw -> clean actions [B, 16, 8], future [B, 64, 384] standardised
        when, where, role = self.when.weight, self.where.weight, self.role.weight
        seen = self.feat_in(feats) + when[[o + 12 for o in PAST]][:, None] + where + role[0]
        fut = (self.query + when[12 + CHUNK] + where + role[1]).expand(len(feats), -1, -1)
        act = self.act_in(x_a) + self.tau_in(tau_a[:, None])[:, None] + when[12:12 + CHUNK] + role[2]
        n = len(PAST) * GRID * GRID
        out = self.body(torch.cat([seen.flatten(1, 2), fut, act], 1))
        now = (feats[:, -1] - self.f_mean) / self.f_std
        return self.act_out(out[:, -CHUNK:]), now + self.fut_out(out[:, n:n + GRID * GRID])

    def forward(self, feats: Tensor, x_a: Tensor, tau_a: Tensor, future: Tensor | None = None) -> Tensor:
        assert future is None, 'the future is a target here, never an input'
        return self.both(feats, x_a, tau_a)[0]          # plan() (inherited) uses this


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='pick_apple_rung0_n30')
    ap.add_argument('--labelled', type=int, default=10)
    ap.add_argument('--more-video', nargs='*', default=[], help='dino-cached recordings, every clip video-only')
    ap.add_argument('--future-weight', type=float, default=1.0, help='future loss weight vs action loss')
    ap.add_argument('--val', type=int, default=5)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--save-every', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-3)
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
    val_eps, lab_eps = order[:args.val], order[args.val:args.val + args.labelled]
    assert len(lab_eps) == args.labelled, 'not enough episodes'
    chunk, real = targets(d['state'], d['action'], d['episode'])
    lab = np.isin(d['episode'], lab_eps)
    a_mean = chunk[lab][real[lab]].mean(0)
    a_std = chunk[lab][real[lab]].std(0) + 1e-4

    # video-only recordings appended after --data; their actions exist on disk but are never read
    episode, grid, n_vid = d['episode'], [d['grid']], 0
    for name in args.more_video:
        m = load(name)
        n_vid += len(np.unique(m['episode']))
        episode = np.concatenate([episode, m['episode'] + episode.max() + 1])
        grid.append(m['grid'])
    grid = np.concatenate(grid).reshape(-1, GRID * GRID, C)
    n_main = len(d['episode'])
    vid = np.zeros(len(episode), bool)
    vid[n_main:] = True
    train = np.concatenate([lab, np.zeros(len(episode) - n_main, bool)]) | vid
    f_mean = grid[train].astype(np.float32).mean((0, 1))
    f_std = grid[train].astype(np.float32).std((0, 1)) + 1e-4

    g = lambda x: torch.as_tensor(x, device=dev)
    FE = g(grid)                                                  # f16 [N, 64, 384]
    pad = lambda x: np.concatenate([x, np.zeros((len(episode) - n_main,) + x.shape[1:], x.dtype)])
    CH, RE = g(pad((chunk - a_mean) / a_std)), g(pad(real))
    PASTR, AFTER = g(rows_at(episode, PAST)), g(rows_at(episode, [CHUNK])[:, 0])
    lab_rows, vid_rows = g(np.flatnonzero(lab)), g(np.flatnonzero(vid))
    val_rows = g(np.flatnonzero(np.isin(d['episode'], val_eps)))

    net = DinoTarget(g(a_mean), g(a_std), g(f_mean), g(f_std)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 100)
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema)) if args.ema else None
    model = lambda: ema.module if ema is not None else net
    print(f'{args.data}: labelled {len(lab_eps)} eps ({len(lab_rows)} rows) + video-only {n_vid} eps '
          f'({len(vid_rows)} rows), val {len(val_eps)}; {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params',
          flush=True)

    def losses(rows: Tensor, has_a: Tensor, m: nn.Module | None = None) -> tuple[Tensor, Tensor]:
        feats = FE[PASTR[rows]].float()
        clean = CH[rows]
        tau = torch.where(has_a, torch.rand(len(rows), device=dev), torch.zeros(len(rows), device=dev))
        x_a = (1 - tau[:, None, None]) * torch.randn_like(clean) + tau[:, None, None] * clean
        a_hat, f_hat = (m or net).both(feats, x_a, tau)
        keep = (RE[rows] & has_a[:, None])[..., None].float()
        loss_a = (((a_hat - clean) ** 2) * keep).sum() / (keep.sum() * 8).clamp(min=1)
        target = (FE[AFTER[rows]].float() - net.f_mean) / net.f_std
        return loss_a, (f_hat.float() - target).pow(2).mean()

    with torch.no_grad():
        z = lambda r: (FE[r].float() - net.f_mean) / net.f_std
        copy_val = (z(PASTR[val_rows][:, -1]) - z(AFTER[val_rows])).pow(2).mean().item()

    @torch.no_grad()
    def validate() -> dict:
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(1234)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                r = val_rows[:1024]
                la, lf = losses(r, torch.ones(len(r), dtype=torch.bool, device=dev), m=model())
                r = val_rows[::4][:256]
                plan = model().plan(FE[PASTR[r]].float())
            truth = CH[r] * net.a_std + net.a_mean
            err = (plan[:, 0, :7] - truth[:, 0, :7]).abs().mean().item() * 1000
        return {'val_action': la.item(), 'val_future_over_copy': lf.item() / copy_val, 'first_action_mrad': err}

    history, t0, best = [], time.time(), (0, float('inf'))
    pick = lambda r: r[torch.randint(0, len(r), (args.batch,), device=dev)]
    for step in range(1, args.steps + 1):
        rows = torch.cat([pick(lab_rows)] + ([pick(vid_rows)] if len(vid_rows) else []))
        has_a = torch.arange(len(rows), device=dev) < args.batch
        with torch.autocast('cuda', dtype=torch.bfloat16):
            la, lf = losses(rows, has_a)
        opt.zero_grad(set_to_none=True)
        (la + args.future_weight * lf).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update_parameters(net)
        if step % 500 == 0 or step == 1:
            v = validate()
            history.append({'step': step, 'action': la.item(), 'future': lf.item(), **v})
            print(f"step {step:5d}  train action {la.item():.4f} future {lf.item():.4f}  "
                  f"val action {v['val_action']:.4f}  val future/copy {v['val_future_over_copy']:.3f}  "
                  f"first-action {v['first_action_mrad']:6.2f} mrad  lr {opt.param_groups[0]['lr']:.1e}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        blob = lambda: {'state_dict': model().state_dict(), 'step': step, 'kind': 'dino_target',
                        'args': vars(args) | {'out': str(args.out)},
                        'split': {'val': val_eps.tolist(), 'labelled': lab_eps.tolist(), 'more_video': n_vid}}
        if step % 500 == 0 and v['val_action'] < best[1]:   # best.pt: lowest val action so far
            best = (step, v['val_action'])
            torch.save(blob(), args.out / 'best.pt')
        if step % args.save_every == 0 or step == args.steps:
            torch.save(blob(), args.out / f'step{step}.pt')
    (args.out / 'history.json').write_text(json.dumps(history, indent=1))
    (args.out / 'best.json').write_text(json.dumps({'step': best[0], 'val_action': best[1]}))
    print(f'best val action {best[1]:.4f} at step {best[0]} -> best.pt')
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {args.out}')


if __name__ == '__main__':
    main()
