"""Frozen DINOv2-small features for every wrist frame of a recording (variant B, step 1).

    wrist frame 256 px -> 224 px -> DINOv2 ViT-S/14 (frozen) -> 16x16 patches x 384 + CLS
                                  -> patches averaged 2x2 -> 8x8 x 384  (what B's networks read)

Reads the raw npz recording (lossless frames, the same pixels the kitchen gives at run time),
not the LeRobot copy (video-compressed). Writes data/robocasa/<name>_dino.npz:
grid [N, 8, 8, 384] f16, cls [N, 384] f16, state, action, episode -- rows in the same order
as uwm_cam.load(name) (checked), so its episode split carries over. A recording with no LeRobot
copy (extra video-only clips, e.g. pick_apple_rung0_video_n50) is read from its raw files alone.

Also draws experiments/pick_shapes/dino/features.png: frames along one episode, each with
(a) the first 3 PCA directions of its 16x16 features as RGB, (b) the same at 8x8, (c) cosine
similarity of every 8x8 cell to the apple (reference =
the 2x2 cells at rows 5-6, cols 3-4 just before the gripper closes: the apple, between the fingers).

    .venv/bin/python experiments/pick_shapes/scripts/dino_cache.py --data pick_apple_rung0_n30 --picture
    .venv/bin/python experiments/pick_shapes/scripts/dino_cache.py --data pick_apple_rung0_video_n50
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

MODEL = 'facebook/dinov2-small'
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Dino:
    # uint8 frames [N, H, W, 3] -> (patches [N, 16, 16, 384], cls [N, 384]), float32; train and run time
    def __init__(self, device: str = 'cuda') -> None:
        from transformers import AutoModel

        self.net = AutoModel.from_pretrained(MODEL).to(device).eval()
        self.device = device
        self.mean = torch.tensor(MEAN, device=device)[:, None, None]
        self.std = torch.tensor(STD, device=device)[:, None, None]

    @torch.no_grad()
    def __call__(self, frames: np.ndarray, batch: int = 64) -> tuple[np.ndarray, np.ndarray]:
        patches, cls = [], []
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i:i + batch]).to(self.device).permute(0, 3, 1, 2).float() / 255
            x = F.interpolate(x, size=(224, 224), mode='bilinear', antialias=True)
            h = self.net(pixel_values=(x - self.mean) / self.std).last_hidden_state   # [B, 1+256, 384]
            cls.append(h[:, 0].float().cpu())
            patches.append(h[:, 1:].reshape(-1, 16, 16, h.shape[-1]).float().cpu())
        return torch.cat(patches).numpy(), torch.cat(cls).numpy()


def pool8(patches: np.ndarray) -> np.ndarray:
    # [N, 16, 16, C] -> [N, 8, 8, C], mean over 2x2 cells
    n, _, _, c = patches.shape
    return patches.reshape(n, 8, 2, 8, 2, c).mean((2, 4))


def picture(wrist: np.ndarray, patches: np.ndarray, action: np.ndarray, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = pool8(patches)
    close = int(np.argmax(action[:, 7] < 0.5))               # first step told to close (1 = open, 0 = shut)
    picks = np.linspace(0, len(wrist) - 1, 6).round().astype(int)
    picks[np.argmin(abs(picks - close))] = max(close - 1, 0)
    # PCA basis from all patches of this episode, so colours mean the same thing across frames
    flat = patches.reshape(-1, patches.shape[-1])
    mu = flat.mean(0)
    _, _, vt = np.linalg.svd(flat[::7] - mu, full_matrices=False)

    def rgb(f: np.ndarray) -> np.ndarray:
        p = (f - mu) @ vt[:3].T
        lo, hi = np.percentile(flat[::7] @ vt[:3].T - mu @ vt[:3].T, [2, 98], axis=0)
        return np.clip((p - lo) / (hi - lo), 0, 1)

    ref_frame = grid[max(close - 1, 0)]
    centre = ref_frame[5:7, 3:5].reshape(-1, ref_frame.shape[-1]).mean(0)   # apple: low centre, between the fingers
    unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    fig, axes = plt.subplots(4, len(picks), figsize=(2.2 * len(picks), 9))
    for j, t in enumerate(picks):
        sim = unit(grid[t]) @ unit(centre)
        for i, (im, title) in enumerate([(wrist[t], f'step {t}' + (' (close-1)' if t == close - 1 else '')),
                                         (rgb(patches[t]), 'PCA 16x16'), (rgb(grid[t]), 'PCA 8x8'),
                                         (sim, 'sim to apple')]):
            ax = axes[i, j]
            ax.imshow(im, vmin=0, vmax=1, cmap='magma', interpolation='nearest')
            ax.set_title(title, fontsize=8)
            ax.axis('off')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='pick_apple_rung0_n30')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--picture', action='store_true', help='redraw dino/features.png from episode 0')
    args = ap.parse_args()
    raw = sorted((REPO / 'data/robocasa' / f'{args.data}_raw').glob('ep*.npz'))
    eps = [dict(np.load(f)) for f in raw]
    state = np.concatenate([e['state'] for e in eps])
    if (REPO / 'data/robocasa' / args.data).exists():        # has a LeRobot copy: train in its order
        ref = load(args.data)
        assert np.allclose(state, ref['state'], atol=1e-5), 'raw episodes not in LeRobot order'
        print(f'{len(eps)} episodes, {len(state)} frames; order matches uwm_cam.load', flush=True)
    else:                                                    # raw only: file order
        ref = {'state': state, 'action': np.concatenate([e['action'] for e in eps]),
               'episode': np.concatenate([np.full(len(e['state']), i) for i, e in enumerate(eps)])}
        print(f'{len(eps)} episodes, {len(state)} frames; raw only', flush=True)

    t0 = time.time()
    dino = Dino(args.device)
    print(f'model loaded {time.time() - t0:.0f}s', flush=True)
    t0 = time.time()
    patches, cls = [], []
    for e in eps:
        p, c = dino(e['wrist'])
        patches.append(p)
        cls.append(c)
    print(f'features {time.time() - t0:.0f}s ({len(state) / (time.time() - t0):.0f} frames/s)', flush=True)

    out = REPO / 'data/robocasa' / f'{args.data}_dino.npz'
    np.savez(out, grid=pool8(np.concatenate(patches)).astype(np.float16),
             cls=np.concatenate(cls).astype(np.float16),
             state=ref['state'], action=ref['action'], episode=ref['episode'])
    print('wrote', out.relative_to(REPO), flush=True)
    if args.picture:
        fig = REPO / 'experiments/pick_shapes/dino/features.png'
        picture(eps[0]['wrist'], patches[0], eps[0]['action'], fig)
        print('wrote', fig.relative_to(REPO), flush=True)


if __name__ == '__main__':
    main()
