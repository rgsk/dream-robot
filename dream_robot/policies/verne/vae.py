"""A small image VAE for the camera canvas: [top | wrist] 3 x 128 x 256 -> a 4 x 16 x 32 latent grid
(8x smaller each side, 48x fewer numbers).

Verne (../verne-v0.5/src/verne/model/vae.py) encodes every frame alone with the frozen Wan2.2 VAE
encoder: residual conv blocks with RMS norm + SiLU, stride-2 downsampling, a mean and log-variance
head, and the policy reads the mean normalised per channel by fixed latent stats (LATENT_MEAN,
LATENT_STD). This one keeps that shape and drops the rest for now: 2-D convs (one frame, so
Verne's causal 3-D convs see only their last tap anyway), no attention block, no average-pool
shortcuts, 3 halvings instead of 4, 4 channels instead of 48, trained here instead of pretrained.
The decoder only exists to train the encoder.
"""

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import lr as lr_mod

# the VAE every step reads through: step 07's, on T1, T1-wide (low grasp, +-45 deg bias) and T5 frames
# (`run vae 07_instruction`, WSD at 5e-4 on a 4090; its step6000.pt, before a loss spike at step 7000,
# made into weights.pt by experiments/verne/scripts/vae_from_ckpt.py). Keeps T5's colours: 97-99% of
# held-out cube pixels (step 03's: green 7%, blue 0%)
WEIGHTS = Path(
    os.environ.get(  # a one-off run on another VAE sets VERNE_VAE_WEIGHTS (inherited by sweep workers)
        'VERNE_VAE_WEIGHTS',
        Path(__file__).resolve().parents[3]
        / 'experiments/verne/runs/07_instruction/vae_wsd/weights.pt',
    )
)
Z = 4  # latent channels
DIMS = (32, 64, 128)  # channels at 128x256, 64x128, 32x64; the latent is at 16x32


class RMSNorm(nn.Module):
    # over the channel axis, as Verne's VAE
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * x.shape[1] ** 0.5 * self.gamma


class ResBlock(nn.Module):
    def __init__(self, i: int, o: int):
        super().__init__()
        self.body = nn.Sequential(
            RMSNorm(i),
            nn.SiLU(),
            nn.Conv2d(i, o, 3, padding=1),
            RMSNorm(o),
            nn.SiLU(),
            nn.Conv2d(o, o, 3, padding=1),
        )
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x) + self.skip(x)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        enc: list[nn.Module] = [nn.Conv2d(3, DIMS[0], 3, padding=1)]
        for i, o in zip(DIMS, (*DIMS[1:], DIMS[-1])):
            enc += [ResBlock(i, o), nn.Conv2d(o, o, 3, stride=2, padding=1)]  # halve h, w
        enc += [ResBlock(DIMS[-1], DIMS[-1]), RMSNorm(DIMS[-1]), nn.SiLU()]
        enc += [nn.Conv2d(DIMS[-1], 2 * Z, 3, padding=1)]  # mean, log-variance
        self.enc = nn.Sequential(*enc)
        dec: list[nn.Module] = [nn.Conv2d(Z, DIMS[-1], 3, padding=1), ResBlock(DIMS[-1], DIMS[-1])]
        for i, o in zip(DIMS[::-1], (*DIMS[::-1][1:], DIMS[0])):
            dec += [nn.Upsample(scale_factor=2), nn.Conv2d(i, i, 3, padding=1), ResBlock(i, o)]
        dec += [RMSNorm(DIMS[0]), nn.SiLU(), nn.Conv2d(DIMS[0], 3, 3, padding=1)]
        self.dec = nn.Sequential(*dec)
        # per-channel stats of the mean over the training frames, set after training (Verne's
        # LATENT_MEAN / LATENT_STD): encode() gives latents of mean 0, std 1 per channel
        self.register_buffer('z_mean', torch.zeros(Z))
        self.register_buffer('z_std', torch.ones(Z))

    def encode(self, x: Tensor) -> Tensor:
        # canvas [B, 3, 128, 256] in [-1, 1] -> normalised mean latent [B, Z, 16, 32]
        mean = self.enc(x)[:, :Z]
        return (mean - self.z_mean[:, None, None]) / self.z_std[:, None, None]


"""
The VAE's loss for one image x, with the encoder giving μ, v = log σ² per latent number:

    z = μ + σ·ε,  ε ~ N(0, 1)             (a sample, differentiable in μ and σ)
    loss = ‖dec(z) − x‖² / n  +  β · KL

    KL( N(μ, σ²) ‖ N(0, 1) ) = ½ (μ² + e^v − 1 − v)     per number, 0 at μ = 0, σ = 1

The reconstruction term wants z to carry everything about x; the KL term pulls every z toward
N(0, 1), so nearby latents decode to similar images. β small (1e-3): sharp images first.
"""


def loss(vae: VAE, x: Tensor) -> tuple[Tensor, Tensor]:
    # x [B, 3, 128, 256] in [-1, 1] -> reconstruction MSE, KL per latent number
    mean, logvar = vae.enc(x).chunk(2, dim=1)
    z = mean + torch.randn_like(mean) * (0.5 * logvar).exp()
    kl = 0.5 * (mean.square() + logvar.exp() - 1 - logvar).mean()
    return F.mse_loss(vae.dec(z), x), kl


def to_float(frames: np.ndarray) -> Tensor:
    # uint8 [B, 3, 128, 256] (cpu numpy) -> [-1, 1] on the GPU
    return torch.from_numpy(frames).cuda().float() / 127.5 - 1


def train(
    frames: np.ndarray,
    val: np.ndarray,
    steps: int = 10000,
    B: int = 64,
    lr: float = 5e-4,  # stable-phase lr; 1e-3 spiked 20x twice in the WSD stable phase
    warmup: int = 250,
    decay: float = 0.2,
    beta: float = 1e-3,
    seed: int = 0,
    ckpt: Path | None = None,
    extend_from: Path | None = None,
) -> dict[str, Any]:
    # frames: uint8 canvases [N, 3, 128, 256] (cpu) to train on, val: a few held-out ones
    # lr: warmup-stable-decay (lr.py); ckpt: a folder for step<N>.pt / stable.pt (resumes from the
    # latest step<N>.pt);
    # extend_from: a shorter run's ckpt folder to continue from its stable.pt
    # out: {'weights': state dict incl. latent stats (cpu), 'log': [step, recon, KL, val recon, KL]}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vae = VAE().cuda()
    opt = torch.optim.AdamW(vae.parameters(), lr=lr)
    start, log = lr_mod.resume(ckpt, extend_from, steps, decay, vae, opt, rng)
    vx = to_float(val)
    # batch 64 at 128x256 needs > 8 GB in one pass: on a small card, halves with gradients summed
    parts = 1 if torch.cuda.get_device_properties(0).total_memory > 20e9 else 2
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, decay, 1000, vae, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.wsd(step, steps, warmup, decay))
        ids = np.sort(rng.integers(0, len(frames), B))
        opt.zero_grad()
        for part in np.array_split(ids, parts):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                rec, kl = loss(vae, to_float(frames[part]))
            ((rec + beta * kl) / parts).backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 500 == 0:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                chunks = [loss(vae, vx[i : i + 64]) for i in range(0, len(vx), 64)]  # 8 GB card
                v_rec, v_kl = (torch.stack(x).mean() for x in zip(*chunks))
            log.append([step, rec.item(), kl.item(), v_rec.item(), v_kl.item()])
            print(f'step {step:5d}   train {rec:.4f} KL {kl:.3f}   val {v_rec:.4f} KL {v_kl:.3f}')
    set_latent_stats(vae, frames)
    return {'weights': {k: v.cpu() for k, v in vae.state_dict().items()}, 'log': log}


@torch.no_grad()
def set_latent_stats(vae: VAE, frames: np.ndarray) -> None:
    # z_mean / z_std: the encoder mean's per-channel mean and std over every 10th training frame
    every10 = np.arange(0, len(frames), 10)
    chunks = np.array_split(every10, len(every10) // 64 + 1)  # 8 GB card
    mu = torch.cat([vae.enc(to_float(frames[ids]))[:, :Z] for ids in chunks])
    vae.z_mean.copy_(mu.mean((0, 2, 3)))
    vae.z_std.copy_(mu.std((0, 2, 3)))
    print('latent mean', vae.z_mean.cpu().numpy().round(3), 'std', vae.z_std.cpu().numpy().round(3))


@torch.no_grad()
def reconstruct(vae: VAE, frames: np.ndarray) -> np.ndarray:
    # uint8 canvases -> uint8 reconstructions from the mean latent (no sampling)
    mean = vae.enc(to_float(frames))[:, :Z]
    x = vae.dec(mean).clamp(-1, 1)
    return ((x + 1) * 127.5).round().byte().cpu().numpy()


def load(path) -> VAE:
    # a trained VAE (weights.pt from `run vae`) on the GPU, frozen
    vae = VAE().cuda().eval().requires_grad_(False)
    vae.load_state_dict(torch.load(path))
    return vae


def load_latents(name: str, weights_path) -> np.ndarray:
    # out: normalised mean latents [N, Z, 16, 32] float16 by table row, every canvas of dataset
    # `name` through the VAE at `weights_path`. Encoded once into data/robosuite/<name>/decoded/
    # latents_<first 8 of the weights' sha256>.npy (a new VAE gets its own file), then read back
    from .data import REPO, canvas, load_cameras

    tag = hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()[:8]
    path = REPO / 'data/robosuite' / name / 'decoded' / f'latents_{tag}.npy'
    if not path.exists():
        vae, frames = load(weights_path), canvas(*load_cameras(name))
        print(f'encoding {len(frames)} canvases once -> {path}', flush=True)
        with torch.no_grad():
            z = [
                vae.encode(to_float(frames[i : i + 64])).half().cpu()
                for i in range(0, len(frames), 64)
            ]
        np.save(path.with_suffix('.partial.npy'), torch.cat(z).numpy())
        path.with_suffix('.partial.npy').rename(path)  # never a half-written file
    return np.load(path)
