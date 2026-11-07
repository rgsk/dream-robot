"""DINO, the whole game in one file (dino.ipynb step 2): a tiny ViT learns CIFAR-10 features
with no labels. Later notebook steps rebuild each piece and ablate it.

    image -> 2 big crops (32 px) + 4 small crops (16 px), each randomly flipped / recoloured
    student (ViT + head) sees all 6 crops; teacher (EMA copy of the student) sees the 2 big ones
    loss: for every pair (teacher's big crop i, student's crop j != i)
          cross_entropy( softmax((t_i - center) / 0.04),  softmax(s_j / 0.1) )
    teacher <- m * teacher + (1 - m) * student, m: 0.996 -> 1 (cosine)
    center  <- 0.9 * center + 0.1 * mean of teacher outputs        (stops one output winning)

Scored by kNN accuracy on the teacher's CLS features (labels only for scoring).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from frozendict import frozendict
from torch import Tensor, nn

MEAN = torch.tensor([0.491, 0.482, 0.447])[:, None, None]
STD = torch.tensor([0.247, 0.243, 0.262])[:, None, None]

# every training knob; an ablation is CFG | {'n_local': 0}
CFG = frozendict(
    epochs=20,
    batch=256,
    lr=1e-3,
    n_local=4,  # small crops per image
    t_s=0.1,  # student temperature
    t_t=0.04,  # teacher temperature (sharper)
    m0=0.996,  # teacher EMA momentum at the start (-> 1)
    center_m=0.9,  # center EMA momentum
    k=4096,  # head outputs ("votes")
    seed=0,
)


# --- the backbone: a tiny ViT ------------------------------------------------------------


class ViT(nn.Module):
    """[B, 3, H, W] (H = W, a multiple of 4) -> CLS [B, dim], patches [B, (H/4)^2, dim]."""

    def __init__(self, dim: int = 192, depth: int = 6, heads: int = 3, grid: int = 8):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, 4, 4)  # one token per 4x4 patch
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, grid * grid, dim) * 0.02)  # 8x8 grid (32 px)
        # positional: ffn width, dropout 0, GELU, LayerNorm eps, batch_first, pre-norm
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, 0.0, 'gelu', 1e-5, True, True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def pos_for(self, g: int) -> Tensor:
        # position embeddings resized to a g x g grid (small crops have 4x4 patches)
        n = int(self.pos.shape[1] ** 0.5)
        if g == n:
            return self.pos
        p = self.pos.reshape(1, n, n, -1).permute(0, 3, 1, 2)
        p = F.interpolate(p, size=(g, g), mode='bicubic', align_corners=False)
        return p.permute(0, 2, 3, 1).flatten(1, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        t = self.embed(x).flatten(2).transpose(1, 2)  # [B, g*g, dim]
        t = t + self.pos_for(x.shape[-1] // 4)
        t = self.norm(self.blocks(torch.cat([self.cls.expand(len(t), -1, -1), t], 1)))
        return t[:, 0], t[:, 1:]


# --- the head: features -> K-way "votes" --------------------------------------------------


class Head(nn.Module):
    """MLP 192 -> 512 -> 512 -> 128, L2-normalised, then a weight-normalised linear layer to K."""

    def __init__(self, dim: int = 192, k: int = 4096, width: int = 512, bottleneck: int = 128):
        super().__init__()
        layers = [nn.Linear(dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU()]
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(width, bottleneck)
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck, k, bias=False))

    def forward(self, x: Tensor) -> Tensor:
        return self.last(F.normalize(self.proj(self.mlp(x)), dim=-1))


# --- crops and colour changes, on the GPU --------------------------------------------------


def random_crops(x: Tensor, n: int, size: int, scale: tuple[float, float]) -> list[Tensor]:
    """n random resized crops per image, flipped half the time: x [B, 3, 32, 32] in [0, 1]."""
    B, dev = len(x), x.device
    out = []
    for _ in range(n):
        s = torch.empty(B, device=dev).uniform_(*scale).sqrt()  # crop side / image side
        cx, cy = [(torch.rand(B, device=dev) * 2 - 1) * (1 - s) for _ in range(2)]
        flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
        theta = torch.zeros(B, 2, 3, device=dev)
        theta[:, 0, 0], theta[:, 0, 2], theta[:, 1, 1], theta[:, 1, 2] = s * flip, cx, s, cy
        grid = F.affine_grid(theta, [B, 3, size, size], align_corners=False)
        out.append(F.grid_sample(x, grid, padding_mode='reflection', align_corners=False))
    return out


def recolour(x: Tensor) -> Tensor:
    """Random brightness / contrast / saturation (p = 0.8) and grayscale (p = 0.2), per image."""
    B, dev = len(x), x.device
    on = (torch.rand(B, 1, 1, 1, device=dev) < 0.8).float()
    jitter = lambda: 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.4 * on
    x = x * jitter()  # brightness
    mean = x.mean((1, 2, 3), keepdim=True)
    x = (x - mean) * jitter() + mean  # contrast
    gray = x.mean(1, keepdim=True)
    x = (x - gray) * jitter() + gray  # saturation
    to_gray = torch.rand(B, 1, 1, 1, device=dev) < 0.2
    return torch.where(to_gray, x.mean(1, keepdim=True).expand_as(x), x).clamp(0, 1)


def views(x: Tensor, n_local: int) -> list[Tensor]:
    # 2 big crops (32 px, 40-100% of the image) + n_local small ones (16 px, 5-40%), normalised
    crops = random_crops(x, 2, 32, (0.4, 1.0)) + random_crops(x, n_local, 16, (0.05, 0.4))
    return [(recolour(c) - MEAN.to(x.device)) / STD.to(x.device) for c in crops]


# --- the loss --------------------------------------------------------------------------------


def dino_loss(s_out: list[Tensor], t_out: list[Tensor], center: Tensor, t_s: float, t_t: float):
    # s_out: logits per crop (2 big first); t_out: per big crop. Same-crop pairs are skipped.
    targets = [F.softmax((t - center) / t_t, dim=-1) for t in t_out]
    total, n = 0.0, 0
    for i, p in enumerate(targets):
        for j, s in enumerate(s_out):
            if i != j:
                total = total + (-p * F.log_softmax(s / t_s, dim=-1)).sum(-1).mean()
                n += 1
    return total / n


# --- scoring ---------------------------------------------------------------------------------


@torch.no_grad()
def features(vit: ViT, images: np.ndarray, dev: str = 'cuda') -> Tensor:
    # CLS features of whole 32 px images, no augmentation: [N, dim]
    out = []
    for i in range(0, len(images), 1000):
        x = torch.as_tensor(images[i : i + 1000], device=dev).permute(0, 3, 1, 2) / 255
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(vit((x - MEAN.to(dev)) / STD.to(dev))[0].float())
    return torch.cat(out)


@torch.no_grad()
def knn(train_f: Tensor, train_y: np.ndarray, test_f: Tensor, test_y: np.ndarray, k: int = 20):
    # each test image votes with the classes of its k most similar train images (cosine)
    a, b = F.normalize(train_f, dim=1), F.normalize(test_f, dim=1)
    ya, yb = torch.as_tensor(train_y, device=a.device), torch.as_tensor(test_y, device=a.device)
    nearest = torch.cat([(b[i : i + 1000] @ a.T).topk(k).indices for i in range(0, len(b), 1000)])
    return (F.one_hot(ya[nearest], 10).sum(1).argmax(1) == yb).float().mean().item()


# --- training --------------------------------------------------------------------------------


def train_dino(cifar, cfg=CFG, dev: str = 'cuda'):
    """cifar: frozendict(train=(images, labels), test=(...)). Returns (teacher ViT, history)."""
    torch.manual_seed(cfg['seed'])
    images, labels = cifar['train']
    test_images, test_labels = cifar['test']
    data = torch.as_tensor(images, device=dev).permute(0, 3, 1, 2)  # uint8 on the GPU
    student = nn.ModuleDict({'vit': ViT(), 'head': Head(k=cfg['k'])}).to(dev)
    teacher = copy.deepcopy(student).requires_grad_(False)
    center = torch.zeros(cfg['k'], device=dev)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg['lr'], weight_decay=0.05)
    per_epoch = len(data) // cfg['batch']
    total, warm = cfg['epochs'] * per_epoch, 2 * per_epoch
    cosine = lambda step: 0.5 * (1 + math.cos(math.pi * step / total))  # 1 -> 0
    score = lambda vit: knn(features(vit, images), labels, features(vit, test_images), test_labels)
    history, t0, step = [], time.time(), 0
    for epoch in range(cfg['epochs']):
        order = torch.randperm(len(data), device=dev)
        for b in range(per_epoch):
            for g in opt.param_groups:  # warmup, then cosine
                g['lr'] = cfg['lr'] * min(1, step / warm) * cosine(step)
            x = data[order[b * cfg['batch'] : (b + 1) * cfg['batch']]] / 255
            v = views(x, cfg['n_local'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    t_out = [teacher['head'](teacher['vit'](c)[0]).float() for c in v[:2]]
                s_out = [student['head'](student['vit'](c)[0]).float() for c in v]
                loss = dino_loss(s_out, t_out, center, cfg['t_s'], cfg['t_t'])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step()
            with torch.no_grad():
                m = 1 - (1 - cfg['m0']) * cosine(step)  # teacher momentum: m0 -> 1
                for pt, ps in zip(teacher.parameters(), student.parameters()):
                    pt.mul_(m).add_(ps.detach(), alpha=1 - m)
                cm = cfg['center_m']
                center.mul_(cm).add_(torch.cat(t_out).mean(0), alpha=1 - cm)
            step += 1
        acc = score(teacher['vit'])
        history.append({'epoch': epoch + 1, 'loss': loss.item(), 'knn': acc})
        took = time.time() - t0
        print(f'epoch {epoch + 1:3d}  loss {loss.item():.3f}  kNN {acc:.1%}  {took:.0f}s')
    return teacher['vit'], history
