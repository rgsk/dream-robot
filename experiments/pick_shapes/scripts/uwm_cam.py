"""Camera-only UWM on pick_shapes: wrist frames in; future frame + action chunk out.

The thesis test (notes.md, step 6): does video *without actions* help a policy that has few
labelled demos? One transformer, two jobs sharing it (rough/uwm_parts.py's UWMxx, trimmed):

    in   4 wrist frames (t-12, t-8, t-4, t), 128 px, 8x8 tokens each        -- nothing else
    out  frame t+16 (pixel patches)  and  16 actions (t .. t+15)
         both by flow matching, each part at its own noise level tau; heads predict the clean
         thing (UWMxx: as good as velocity heads on T1, and simpler)

No joints, no touch. So the arm action is a *change*:  a[t+k][:7] - q[t][:7]  (k = 0..15, all
from the joints at planning time t); at run time the robot adds its own q[t] back. That is
units, not input: the network never sees q. The gripper stays the 0..1 command.

Labelled episodes train both heads. Video-only episodes train the frame head only: their action
tokens are pure noise (tau_a = 0) and their action loss is masked out.

    .venv/bin/python experiments/pick_shapes/scripts/uwm_cam.py --labelled 25 --video 0 \
        --out experiments/pick_shapes/uwm_cam/l25_v0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

GRID, D, HW = 8, 128, 128          # 8 x 8 tokens per frame, D-dim, frames 128 px
PAST = (-12, -8, -4, 0)            # frames seen, as step offsets
CHUNK, AHEAD = 16, 16              # actions predicted; how far ahead the imagined frame is
REPO = Path(__file__).resolve().parents[3]


# --- data ------------------------------------------------------------------------------

def to_128(frames: np.ndarray) -> np.ndarray:
    # [N, H, W, 3] or [N, 3, H, W] uint8 -> [N, 3, 128, 128] uint8, antialiased; train and run time
    x = torch.from_numpy(frames)
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    x = F.interpolate(x.float(), size=(HW, HW), mode='bilinear', antialias=True)
    return x.round().clamp(0, 255).to(torch.uint8).numpy()


def load(name: str, camera: str = 'wrist') -> dict[str, np.ndarray]:
    # state [N, 8], action [N, 8], episode [N], frames [N, 3, 128, 128]; cached next to the dataset
    root = REPO / 'data/robocasa' / name
    cache = root.with_name(f'{name}_{camera}128.npz')
    if cache.exists():
        return dict(np.load(cache))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f'dream_robot/{name}', root=root)
    t = ds.hf_dataset
    out = {
        'state': np.asarray(t['observation.state'], np.float32),
        'action': np.asarray(t['action'], np.float32),
        'episode': np.asarray(t['episode_index'], np.int64),
    }
    frames = np.stack([(ds[i][f'observation.images.{camera}'] * 255).round().byte().numpy()
                       for i in range(len(ds))])                      # [N, 3, 256, 256]
    out['frames'] = to_128(frames)
    np.savez(cache, **out)
    return out


def rows_at(episode: np.ndarray, offsets) -> np.ndarray:
    # [N, len(offsets)]: row of step t + o for every row t, clipped to t's own episode
    first = np.searchsorted(episode, episode, side='left')
    last = np.searchsorted(episode, episode, side='right') - 1
    idx = np.arange(len(episode))[:, None] + np.asarray(offsets)
    return np.clip(idx, first[:, None], last[:, None])


def targets(state: np.ndarray, action: np.ndarray, episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # out: chunk [N, 16, 8] (arm as change from q[t], gripper as command), real [N, 16] bool
    chunk_rows = rows_at(episode, range(CHUNK))
    real = chunk_rows == np.arange(len(episode))[:, None] + np.arange(CHUNK)
    chunk = action[chunk_rows].copy()
    chunk[..., :7] -= state[:, None, :7]
    return chunk.astype(np.float32), real


# --- model -----------------------------------------------------------------------------

class Patches(nn.Module):
    # a frame -> 64 tokens: [B, 3, 128, 128] in [0, 1] -> [B, 64, D]
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),       # 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),      # 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),     # 16
            nn.Conv2d(128, D, 4, 2, 1),                 # 8
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x).flatten(2).transpose(1, 2)


def cut(x: Tensor) -> Tensor:
    # [B, 3, 128, 128] -> [B, 64, 768]: each 16 x 16 patch's pixels, same order as Patches
    return F.unfold(x, 16, stride=16).transpose(1, 2)


class UWMCam(nn.Module):
    """Seen frames + noisy future frame + noisy actions -> clean future frame, clean actions."""

    def __init__(self, a_mean: Tensor, a_std: Tensor, layers: int = 4):
        super().__init__()
        self.register_buffer('a_mean', a_mean)
        self.register_buffer('a_std', a_std)
        self.patches = Patches()
        self.frame_in = nn.Linear(3 * 16 * 16, D)
        self.act_in = nn.Linear(8, D)
        self.tau_in = nn.Sequential(nn.Linear(1, D), nn.SiLU(), nn.Linear(D, D))
        self.when = nn.Embedding(12 + AHEAD + 1, D)     # row = step offset + 12
        self.where = nn.Embedding(GRID * GRID, D)
        self.role = nn.Embedding(3, D)                  # seen frame, noisy frame, action
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, 0.0, 'gelu', batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.frame_out = nn.Linear(D, 3 * 16 * 16)
        self.act_out = nn.Linear(D, 8)
        for head in (self.frame_out, self.act_out):     # start by saying 0
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, seen: Tensor, x_f: Tensor, x_a: Tensor, tau_v: Tensor, tau_a: Tensor):
        # seen [B, 4, 3, 128, 128] in [0, 1]; x_f [B, 64, 768]; x_a [B, 16, 8]; tau_* [B]
        # out: clean frame patches [B, 64, 768], clean actions [B, 16, 8] (std units)
        B = seen.shape[0]
        when, where, role = self.when.weight, self.where.weight, self.role.weight
        grids = self.patches(seen.flatten(0, 1)).unflatten(0, (B, len(PAST)))
        seen_tok = grids + when[[o + 12 for o in PAST]][:, None] + where + role[0]
        frame_tok = self.frame_in(x_f) + self.tau_in(tau_v[:, None])[:, None] + when[12 + AHEAD] + where + role[1]
        act_tok = self.act_in(x_a) + self.tau_in(tau_a[:, None])[:, None] + when[12:12 + CHUNK] + role[2]
        n_seen = len(PAST) * GRID * GRID
        out = self.body(torch.cat([seen_tok.flatten(1, 2), frame_tok, act_tok], 1))
        return self.frame_out(out[:, n_seen:n_seen + 64]), self.act_out(out[:, n_seen + 64:])

    @torch.no_grad()
    def plan(self, seen: Tensor, steps: int = 10) -> Tensor:
        # the policy job: frame left as noise, actions denoised -> [B, 16, 8] (arm change, gripper)
        B, dev = seen.shape[0], seen.device
        x_f = torch.randn(B, 64, 768, device=dev)
        tau_v = torch.zeros(B, device=dev)
        x_a = torch.randn(B, CHUNK, 8, device=dev)
        for k in range(steps):
            tau_a = torch.full((B,), k / steps, device=dev)
            _, clean = self(seen, x_f, x_a, tau_v, tau_a)
            x_a = x_a + (clean.float() - x_a) / (1 - k / steps) / steps   # velocity from clean
        return x_a * self.a_std + self.a_mean


# --- training --------------------------------------------------------------------------

def shift(x: Tensor, pad: int = 4) -> Tensor:
    # random translation by up to pad px, the same for every frame of a sample: [B, T, 3, H, W]
    B, T, C, H, W = x.shape
    x = F.pad(x.flatten(1, 2), (pad,) * 4, mode='replicate')
    i, j = np.random.randint(0, 2 * pad + 1, 2)
    return x[..., i:i + H, j:j + W].unflatten(1, (T, C))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='pick_apple_rung0_n30')
    ap.add_argument('--labelled', type=int, default=25, help='train episodes that keep their actions')
    ap.add_argument('--video', type=int, default=0, help='train episodes used as video only')
    ap.add_argument('--val', type=int, default=5)
    ap.add_argument('--balanced', action='store_true',
                    help='each step: --batch labelled rows (as a no-video run) + --batch video-only rows '
                         '(frame loss only), so the action side matches the run without video exactly')
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--save-every', type=int, default=2500)
    ap.add_argument('--lr', type=float, default=1e-3, help='peak lr; cosine-decays to lr/100 by the last step')
    ap.add_argument('--flat-lr', action='store_true', help='no decay (the first anchor run, l25_v0_s0)')
    ap.add_argument('--ema', type=float, default=0.999, help='weight average kept, validated and saved; 0 = off')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda'
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data)
    eps = np.unique(d['episode'])
    order = np.random.default_rng(0).permutation(eps)       # split fixed across runs (seed 0)
    val_eps = order[:args.val]
    lab_eps = order[args.val:args.val + args.labelled]
    vid_eps = order[args.val + args.labelled:args.val + args.labelled + args.video]
    assert len(lab_eps) == args.labelled and len(vid_eps) == args.video, 'not enough episodes'

    chunk, real = targets(d['state'], d['action'], d['episode'])
    lab = np.isin(d['episode'], lab_eps)
    a_mean = chunk[lab][real[lab]].mean(0)
    a_std = chunk[lab][real[lab]].std(0) + 1e-4
    g = lambda x: torch.as_tensor(x, device=dev)
    FR, CH, RE = g(d['frames']), g((chunk - a_mean) / a_std), g(real)
    PASTR, AFTER = g(rows_at(d['episode'], PAST)), g(rows_at(d['episode'], [AHEAD])[:, 0])
    HAS_A = g(lab)
    train_rows = g(np.flatnonzero(np.isin(d['episode'], np.concatenate([lab_eps, vid_eps]))))
    lab_rows = g(np.flatnonzero(np.isin(d['episode'], lab_eps)))
    vid_rows = g(np.flatnonzero(np.isin(d['episode'], vid_eps)))
    assert not args.balanced or len(vid_rows), '--balanced needs --video > 0'
    val_rows = g(np.flatnonzero(np.isin(d['episode'], val_eps)))

    net = UWMCam(g(a_mean), g(a_std)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    # Cosine: fast start at lr, calm finish at lr/100, so the last weights are not one jittery snapshot.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr / 100)
    # EMA: the average of recent weights (decay 0.999 ~ the last 1000 steps) is what gets validated
    # and saved, as in the FM trainer.
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema)) if args.ema else None
    model = lambda: ema.module if ema is not None else net
    print(f'{args.data}: labelled {len(lab_eps)} eps, video-only {len(vid_eps)}, val {len(val_eps)}; '
          f'{len(train_rows)} train rows; {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params', flush=True)

    def losses(rows: Tensor, augment: bool, scored: bool = False, m: nn.Module | None = None) -> tuple[Tensor, Tensor]:
        # scored: treat every row as labelled (validation scores actions on held-out episodes)
        seen = FR[PASTR[rows]].float() / 255
        if augment:
            seen = shift(seen)
        clean_f = cut(FR[AFTER[rows]].float() / 255 * 2 - 1)
        clean_a = CH[rows]
        has_a = torch.ones_like(HAS_A[rows]) if scored else HAS_A[rows]
        tau_v, tau_a = torch.rand(2, len(rows), device=dev)
        tau_a = torch.where(has_a, tau_a, torch.zeros_like(tau_a))   # video-only: actions stay noise
        x_f = (1 - tau_v[:, None, None]) * torch.randn_like(clean_f) + tau_v[:, None, None] * clean_f
        x_a = (1 - tau_a[:, None, None]) * torch.randn_like(clean_a) + tau_a[:, None, None] * clean_a
        f_hat, a_hat = (m or net)(seen, x_f, x_a, tau_v, tau_a)
        keep = (RE[rows] & has_a[:, None])[..., None].float()
        loss_a = (((a_hat - clean_a) ** 2) * keep).sum() / (keep.sum() * 8).clamp(min=1)
        return F.mse_loss(f_hat, clean_f), loss_a

    @torch.no_grad()
    def validate() -> dict:
        # val losses (fixed noise) + first-action error of a sampled plan, arm, in mrad
        with torch.random.fork_rng(devices=[dev]):
            torch.manual_seed(1234)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lf, la = losses(val_rows[:1024], augment=False, scored=True, m=model())
                r = val_rows[::4][:256]
                plan = model().plan(FR[PASTR[r]].float() / 255)
            truth = CH[r] * net.a_std + net.a_mean
            err = (plan[:, 0, :7] - truth[:, 0, :7]).abs().mean().item() * 1000
        return {'val_frame': lf.item(), 'val_action': la.item(), 'first_action_mrad': err}

    history, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        if args.balanced:
            pick = lambda r: r[torch.randint(0, len(r), (args.batch,), device=dev)]
            rows = torch.cat([pick(lab_rows), pick(vid_rows)])
        else:
            rows = train_rows[torch.randint(0, len(train_rows), (args.batch,), device=dev)]
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lf, la = losses(rows, augment=True)
        opt.zero_grad(set_to_none=True)
        (lf + la).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if not args.flat_lr:
            sched.step()
        if ema is not None:
            ema.update_parameters(net)
        if step % 500 == 0 or step == 1:
            v = validate()
            history.append({'step': step, 'frame': lf.item(), 'action': la.item(), **v})
            print(f"step {step:5d}  train frame {lf.item():.4f} action {la.item():.4f}  "
                  f"val frame {v['val_frame']:.4f} action {v['val_action']:.4f}  "
                  f"first-action {v['first_action_mrad']:6.2f} mrad  lr {opt.param_groups[0]['lr']:.1e}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            torch.save({'state_dict': model().state_dict(), 'step': step, 'args': vars(args) | {'out': str(args.out)},
                        'split': {'val': val_eps.tolist(), 'labelled': lab_eps.tolist(), 'video': vid_eps.tolist()}},
                       args.out / f'step{step}.pt')
    (args.out / 'history.json').write_text(json.dumps(history, indent=1))
    print(f'done in {(time.time() - t0) / 60:.1f} min -> {args.out}')


if __name__ == '__main__':
    main()
