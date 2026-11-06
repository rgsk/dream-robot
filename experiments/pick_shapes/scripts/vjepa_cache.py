"""Frozen V-JEPA 2 features: one short clip per row, the encoder swap for DINOv2 (notes.md, step 19).

    8 wrist frames t-14, t-12, ..., t (clamped at the episode start; same span as DINOv2's t-12..t)
      -> V-JEPA 2 ViT-L (frozen) -> 4 time-chunks (frame pairs) x 16x16 patches x 1024
      -> each chunk averaged 2x2 -> 8x8, then PCA 1024 -> 384 (fitted on this recording)
      => grid [N, 4, 64, 384]: per row, 4 moments x 64 cells, DINOv2's shape per moment

The chunk tokens attend across the whole clip, so the last chunk also carries the clip's motion
(--features vjepa_last in the trainers feeds only it). Writes data/robocasa/<name>_vjepa.npz with
grid, pca_mean, pca_comp (needed live), state, action, episode -- rows in uwm_cam.load order.

    .venv/bin/python experiments/pick_shapes/scripts/vjepa_cache.py --data pick_apple_rung0_n30

fp32 (live and cached must match; fp16 across torch versions drifted ~1.7%).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from uwm_cam import REPO, load  # noqa: E402

MODEL = 'facebook/vjepa2-vitl-fpc64-256'
CLIP = tuple(range(-14, 1, 2))      # 8 frames, 2 steps apart
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class VJepa:
    """uint8 clips [B, 8, 256, 256, 3] -> chunk features [B, 4, 64, 1024] (before PCA), float32."""

    def __init__(self, device: str = 'cuda'):
        from transformers import VJEPA2Model

        # fp32 everywhere: in fp16, live (torch 2.7) and cached (2.10) features differed by ~1.7%
        self.net = VJEPA2Model.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
        self.device = device
        self.mean = torch.tensor(MEAN, device=device)[:, None, None]
        self.std = torch.tensor(STD, device=device)[:, None, None]

    @torch.no_grad()
    def __call__(self, clips: np.ndarray) -> np.ndarray:
        B, T = clips.shape[:2]
        x = torch.from_numpy(clips).to(self.device).flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        if x.shape[-1] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', antialias=True)
        x = ((x - self.mean) / self.std).unflatten(0, (B, T))                  # [B, 8, 3, 256, 256]
        h = self.net.get_vision_features(x).float()                           # [B, 4*16*16, 1024], time-major
        h = h.reshape(B, T // 2, 8, 2, 8, 2, -1).mean((3, 5))                 # [B, 4, 8, 8, 1024]
        return h.reshape(B, T // 2, 64, -1).cpu().numpy()


def project(feats: np.ndarray, mean: np.ndarray, comp: np.ndarray) -> np.ndarray:
    # [..., 1024] -> [..., 384]
    return (feats - mean) @ comp.T


def clip_rows(episode: np.ndarray) -> np.ndarray:
    # [N, 8]: rows of t-14 .. t (step 2), clamped to t's own episode start
    first = np.searchsorted(episode, episode, side='left')
    return np.maximum(np.arange(len(episode))[:, None] + np.asarray(CLIP), first[:, None])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='pick_apple_rung0_n30')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--dims', type=int, default=384)
    args = ap.parse_args()
    raw = sorted((REPO / 'data/robocasa' / f'{args.data}_raw').glob('ep*.npz'))
    wrist = np.concatenate([np.load(f)['wrist'] for f in raw])                 # [N, 256, 256, 3]
    if (REPO / 'data/robocasa' / args.data).exists():
        ref = load(args.data)
        state = np.concatenate([np.load(f)['state'] for f in raw])
        assert np.allclose(state, ref['state'], atol=1e-5), 'raw episodes not in LeRobot order'
    else:
        eps = [np.load(f) for f in raw]
        ref = {'state': np.concatenate([e['state'] for e in eps]), 'action': np.concatenate([e['action'] for e in eps]),
               'episode': np.concatenate([np.full(len(e['state']), i) for i, e in enumerate(eps)])}
    rows = clip_rows(ref['episode'])
    print(f'{len(raw)} episodes, {len(wrist)} rows', flush=True)

    t0 = time.time()
    enc = VJepa()
    print(f'model loaded {time.time() - t0:.0f}s', flush=True)
    t0 = time.time()
    feats = np.concatenate([enc(wrist[rows[i:i + args.batch]]).astype(np.float16)
                            for i in range(0, len(rows), args.batch)])            # [N, 4, 64, 1024]
    print(f'features {time.time() - t0:.0f}s ({len(rows) / (time.time() - t0):.0f} clips/s)', flush=True)

    sample = feats.reshape(-1, feats.shape[-1])[::37].astype(np.float32)
    mean = sample.mean(0)
    _, sv, vt = np.linalg.svd(sample - mean, full_matrices=False)
    comp = vt[:args.dims]
    kept = (sv[:args.dims] ** 2).sum() / (sv ** 2).sum()
    print(f'PCA 1024 -> {args.dims}: keeps {kept * 100:.1f}% of the variance', flush=True)
    grid = np.concatenate([project(feats[i:i + 512].astype(np.float32), mean, comp).astype(np.float16)
                           for i in range(0, len(feats), 512)])                   # [N, 4, 64, 384]
    out = REPO / 'data/robocasa' / f'{args.data}_vjepa.npz'
    np.savez(out, grid=grid, pca_mean=mean, pca_comp=comp,
             state=ref['state'], action=ref['action'], episode=ref['episode'])
    print('wrote', out.relative_to(REPO), grid.shape, flush=True)


if __name__ == '__main__':
    main()
