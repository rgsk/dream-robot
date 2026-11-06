"""Frozen-DINOv2 policy on pick_shapes: arm A of variant B (notes.md, step 10).

    in   DINOv2-S features of 4 wrist frames (t-12, t-8, t-4, t), 8x8 x 384 each (dino_cache.py)
    out  16 actions (t .. t+15) by flow matching, head predicts the clean chunk (as uwm_cam.py)

Same as uwm_cam.py's action side except the input: frozen features instead of our own conv on
pixels, and no frame head (A has no future). Same episode split, same targets (arm = change from
q[t], gripper = command), same cosine lr + EMA, so A vs uwm_cam's l10_v0 (9/60) isolates the input.
No shift augmentation: the features are cached per frame (an 8x8 grid can't move by 4 px anyway).

Arm B: --predictor adds 64 more tokens, the features of frame t+16 as imagined by a frozen
dino_predict.py checkpoint from the same 4 seen frames (role 1). The head is otherwise identical.
Validation also scores B with its future replaced by "now" (val_action_nofuture): if that is no
worse than val_action, the head isn't using the future.

    .venv/bin/python experiments/pick_shapes/scripts/uwm_dino.py --labelled 10 \
        --out experiments/pick_shapes/uwm_dino/A_l10_s0
    .venv/bin/python experiments/pick_shapes/scripts/uwm_dino.py --labelled 10 \
        --predictor experiments/pick_shapes/uwm_dino/pred_l10_v0/step5000.pt \
        --out experiments/pick_shapes/uwm_dino/B_l10_s0
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

from uwm_cam import CHUNK, GRID, PAST, REPO, rows_at, targets

D, C = 128, 384                    # token width; DINOv2-S feature width


def load(name: str) -> dict[str, np.ndarray]:
    # grid [N, 8, 8, 384] f16, cls, state, action, episode -- written by dino_cache.py
    return dict(np.load(REPO / 'data/robocasa' / f'{name}_dino.npz'))


class DinoAct(nn.Module):
    """Features of the seen frames + noisy actions -> clean actions (std units)."""

    def __init__(self, a_mean: Tensor, a_std: Tensor, layers: int = 4, mixed: bool = False):
        super().__init__()
        self.register_buffer('a_mean', a_mean)
        if mixed:                                       # dino+vjepa_last: tag for the V-JEPA slot
            self.src = nn.Parameter(torch.zeros(D))
        self.register_buffer('a_std', a_std)
        self.feat_in = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, D))
        self.act_in = nn.Linear(8, D)
        self.tau_in = nn.Sequential(nn.Linear(1, D), nn.SiLU(), nn.Linear(D, D))
        self.when = nn.Embedding(12 + CHUNK + 1, D)      # row = step offset + 12
        self.where = nn.Embedding(GRID * GRID, D)
        self.role = nn.Embedding(3, D)                   # seen features, (B: future features), action
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, 0.0, 'gelu', batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.act_out = nn.Linear(D, 8)
        nn.init.zeros_(self.act_out.weight)              # start by saying 0
        nn.init.zeros_(self.act_out.bias)

    def forward(self, feats: Tensor, x_a: Tensor, tau_a: Tensor, future: Tensor | None = None) -> Tensor:
        # feats [B, 4, 64, 384]; x_a [B, 16, 8]; tau_a [B]; future [B, 64, 384] (arm B) or None (A)
        # out: clean actions [B, 16, 8]
        when, where, role = self.when.weight, self.where.weight, self.role.weight
        n = feats.shape[1]                              # 4 slots; 1 (vjepa_last); 5 (4 DINOv2 + V-JEPA newest)
        offs = PAST + (0,) if n == len(PAST) + 1 else PAST[-n:]
        seen = self.feat_in(feats) + when[[o + 12 for o in offs]][:, None] + where + role[0]
        if n == len(PAST) + 1:                          # mark the V-JEPA tokens as a different source
            seen = seen + torch.cat([torch.zeros_like(self.src).expand(n - 1, 1, -1), self.src[None, None]])
        act = self.act_in(x_a) + self.tau_in(tau_a[:, None])[:, None] + when[12:12 + CHUNK] + role[2]
        parts = [seen.flatten(1, 2), act]
        if future is not None:          # same feature path as the seen frames, at step t+16
            parts.insert(1, self.feat_in(future) + when[12 + CHUNK] + where + role[1])
        out = self.body(torch.cat(parts, 1))
        return self.act_out(out[:, -CHUNK:])

    @torch.no_grad()
    def plan(self, feats: Tensor, future: Tensor | None = None, steps: int = 10) -> Tensor:
        # [B, 4, 64, 384] (+ future [B, 64, 384]) -> [B, 16, 8] (arm change, gripper)
        B, dev = feats.shape[0], feats.device
        x_a = torch.randn(B, CHUNK, 8, device=dev)
        for k in range(steps):
            clean = self(feats, x_a, torch.full((B,), k / steps, device=dev), future)
            x_a = x_a + (clean.float() - x_a) / (1 - k / steps) / steps   # velocity from clean
        return x_a * self.a_std + self.a_mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='pick_apple_rung0_n30')
    ap.add_argument('--features', default='dino', choices=('dino', 'vjepa', 'vjepa_last', 'dino+vjepa_last'),
                    help='dino: 4 frames t-12..t (dino_cache); vjepa: 4 moments of one clip t-14..t '
                         '(vjepa_cache); vjepa_last: only its newest moment')
    ap.add_argument('--labelled', type=int, default=10)
    ap.add_argument('--val', type=int, default=5)
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--save-every', type=int, default=2500)
    ap.add_argument('--lr', type=float, default=1e-3, help='peak lr; cosine-decays to lr/100')
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--predictor', type=Path, help='dino_predict.py checkpoint -> arm B (reads its t+16 guess)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    assert not (args.predictor and args.features != 'dino'), '--predictor is DINOv2-only'
    torch.manual_seed(args.seed)
    dev = 'cuda'
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data) if args.features.startswith('dino') else dict(np.load(REPO / 'data/robocasa' / f'{args.data}_vjepa.npz'))
    eps = np.unique(d['episode'])
    order = np.random.default_rng(0).permutation(eps)       # the split uwm_cam.py uses
    val_eps, lab_eps = order[:args.val], order[args.val:args.val + args.labelled]
    assert len(lab_eps) == args.labelled, 'not enough episodes'

    chunk, real = targets(d['state'], d['action'], d['episode'])
    lab = np.isin(d['episode'], lab_eps)
    a_mean = chunk[lab][real[lab]].mean(0)
    a_std = chunk[lab][real[lab]].std(0) + 1e-4
    g = lambda x: torch.as_tensor(x, device=dev)
    if args.features == 'dino':
        FE = g(d['grid'].reshape(len(d['grid']), GRID * GRID, C))             # f16 [N, 64, 384] per frame
        seen_of = lambda rows: FE[PASTR[rows]].float()                        # 4 past frames -> [B, 4, 64, 384]
    elif args.features == 'dino+vjepa_last':
        FE = g(d['grid'].reshape(len(d['grid']), GRID * GRID, C))
        VJ = g(np.load(REPO / 'data/robocasa' / f'{args.data}_vjepa.npz')['grid'][:, -1])  # [N, 64, 384]
        seen_of = lambda rows: torch.cat([FE[PASTR[rows]], VJ[rows][:, None]], 1).float()  # [B, 5, 64, 384]
    else:
        FE = g(d['grid'])                                                     # f16 [N, 4, 64, 384] per row's clip
        last = args.features == 'vjepa_last'
        seen_of = lambda rows: FE[rows][:, -1:].float() if last else FE[rows].float()
    CH, RE = g((chunk - a_mean) / a_std), g(real)
    PASTR = g(rows_at(d['episode'], PAST))
    FUT = None
    if args.predictor:                 # frozen: its guess for every row, computed once
        from dino_predict import Predictor
        pred = Predictor(torch.zeros(C), torch.ones(C)).to(dev).eval()
        pred.load_state_dict(torch.load(args.predictor, map_location=dev, weights_only=False)['state_dict'])
        FUT = torch.cat([pred.predict(FE[PASTR[i:i + 512]].float()) for i in range(0, len(FE), 512)]).half()
        del pred
    lab_rows = g(np.flatnonzero(lab))
    val_rows = g(np.flatnonzero(np.isin(d['episode'], val_eps)))

    net = DinoAct(g(a_mean), g(a_std), mixed=args.features == 'dino+vjepa_last').to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 100)
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema)) if args.ema else None
    model = lambda: ema.module if ema is not None else net
    print(f'{args.data}: labelled {len(lab_eps)} eps, val {len(val_eps)}; {len(lab_rows)} train rows; '
          f'{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params', flush=True)

    def future(rows: Tensor, blank: bool = False) -> Tensor | None:
        # B's t+16 tokens; blank: "now" in their place (does the head use the future at all?)
        if FUT is None:
            return None
        return FE[PASTR[rows][:, -1]].float() if blank else FUT[rows].float()

    def loss(rows: Tensor, m: nn.Module | None = None, blank: bool = False) -> Tensor:
        feats = seen_of(rows)
        clean = CH[rows]
        tau = torch.rand(len(rows), device=dev)
        x_a = (1 - tau[:, None, None]) * torch.randn_like(clean) + tau[:, None, None] * clean
        a_hat = (m or net)(feats, x_a, tau, future(rows, blank))
        keep = RE[rows][..., None].float()
        return (((a_hat - clean) ** 2) * keep).sum() / (keep.sum() * 8)

    @torch.no_grad()
    def validate() -> dict:
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(1234)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                la = loss(val_rows[:1024], m=model())
                r = val_rows[::4][:256]
                plan = model().plan(seen_of(r), future(r))
            truth = CH[r] * net.a_std + net.a_mean
            err = (plan[:, 0, :7] - truth[:, 0, :7]).abs().mean().item() * 1000
            out = {'val_action': la.item(), 'first_action_mrad': err}
            if FUT is not None:
                torch.manual_seed(1234)        # same noise as val_action, only the future differs
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    out['val_action_nofuture'] = loss(val_rows[:1024], m=model(), blank=True).item()
        return out

    history, t0, best = [], time.time(), (0, float('inf'))
    for step in range(1, args.steps + 1):
        rows = lab_rows[torch.randint(0, len(lab_rows), (args.batch,), device=dev)]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            la = loss(rows)
        opt.zero_grad(set_to_none=True)
        la.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if ema is not None:
            ema.update_parameters(net)
        if step % 500 == 0 or step == 1:
            v = validate()
            history.append({'step': step, 'action': la.item(), **v})
            nf = f"  (no future {v['val_action_nofuture']:.4f})" if FUT is not None else ''
            print(f"step {step:5d}  train action {la.item():.4f}  val action {v['val_action']:.4f}{nf}  "
                  f"first-action {v['first_action_mrad']:6.2f} mrad  lr {opt.param_groups[0]['lr']:.1e}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        blob = lambda: {'state_dict': model().state_dict(), 'step': step, 'kind': 'dino',
                        'args': vars(args) | {'out': str(args.out)},
                        'split': {'val': val_eps.tolist(), 'labelled': lab_eps.tolist()}}
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
