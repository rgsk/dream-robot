"""Step 03, VAE latents: a video transformer reads the [top | wrist] canvas once, as a VAE
latent grid, an action
transformer denoises the chunk while reading the video layers' keys and values.

Verne's shape (wam.py, mot.py), everything else as simple as possible for now:
the two cameras side by side as one wide image, as Verne's canvas (no head crop),
the canvas through a small frozen VAE (vae.py, not Wan's), its latent grid cut into 2x2
squares, one frame (no memory), joints (no poses), the state as one
more token (no text), the flow time added to the action tokens (no adaptive norm),
10 even Euler steps.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .data import CHUNK
from .vae import Z

P = 2  # patch side in latent cells: latent 16 x 32 -> 8 x 16 = 128 patches (16 x 16 px each)
N_PATCH = (16 // P) * (32 // P)
HEADS, HEAD_DIM = 4, 32  # shared by both transformers, so their keys and values can mix
D_VIDEO, D_ACTION, LAYERS = 256, 128, 4  # the action transformer is the narrow one


class Block(nn.Module):
    # a pre-norm transformer layer, split in two so a caller can see its keys and values
    # and give it extra ones before attending
    def __init__(self, dim: int):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * HEADS * HEAD_DIM)
        self.out = nn.Linear(HEADS * HEAD_DIM, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def qkv_of(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # [B, S, dim] -> q, k, v, each [B, HEADS, S, HEAD_DIM]
        qkv = rearrange(self.qkv(self.norm1(x)), 'b s (n h d) -> n b h s d', n=3, h=HEADS)
        return qkv[0], qkv[1], qkv[2]

    def finish(self, x: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        a = rearrange(F.scaled_dot_product_attention(q, k, v), 'b h s d -> b s (h d)')
        x = x + self.out(a)
        return x + self.mlp(self.norm2(x))


class Verne(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_in = nn.Linear(Z * P * P, D_VIDEO)
        self.state_in = nn.Linear(8, D_VIDEO)
        self.video_pos = nn.Parameter(torch.randn(1, 1 + N_PATCH, D_VIDEO) * 0.02)
        self.video = nn.ModuleList([Block(D_VIDEO) for _ in range(LAYERS)])
        self.action_in = nn.Linear(8, D_ACTION)
        self.time_in = nn.Sequential(
            nn.Linear(1, D_ACTION), nn.SiLU(), nn.Linear(D_ACTION, D_ACTION)
        )
        self.action_pos = nn.Parameter(torch.randn(1, CHUNK, D_ACTION) * 0.02)
        self.action = nn.ModuleList([Block(D_ACTION) for _ in range(LAYERS)])
        self.action_out = nn.Linear(D_ACTION, 8)

    def see(self, frame: Tensor, state: Tensor) -> list[tuple[Tensor, Tensor]]:
        # latent [B, Z, 16, 32] (normalised), state [B, 8] -> each video layer's keys and values
        patches = rearrange(frame, 'b c (h p) (w q) -> b (h w) (p q c)', p=P, q=P)
        x = torch.cat([self.state_in(state)[:, None], self.patch_in(patches)], 1)
        x = x + self.video_pos
        kvs = []
        for block in self.video:
            q, k, v = block.qkv_of(x)
            kvs.append((k, v))
            x = block.finish(x, q, k, v)
        return kvs

    def velocity(self, kvs: list[tuple[Tensor, Tensor]], x: Tensor, tau: Tensor) -> Tensor:
        # noisy chunk x [B, CHUNK, 8] at flow time tau [B] -> velocity [B, CHUNK, 8];
        # the action tokens attend to the video tokens of the same layer and to each other
        y = self.action_in(x) + self.action_pos + self.time_in(tau[:, None])[:, None]
        for block, (k_video, v_video) in zip(self.action, kvs):
            q, k, v = block.qkv_of(y)
            y = block.finish(y, q, torch.cat([k_video, k], 2), torch.cat([v_video, v], 2))
        return self.action_out(y)


def loss(net: Verne, frame: Tensor, state: Tensor, acts: Tensor) -> Tensor:
    # flow matching: on the line from noise (tau 0) to the chunk (tau 1), predict the direction
    noise = torch.randn_like(acts)
    tau = torch.rand(len(acts), device=acts.device)
    x = (1 - tau[:, None, None]) * noise + tau[:, None, None] * acts
    return F.mse_loss(net.velocity(net.see(frame, state), x, tau), acts - noise)


@torch.no_grad()
def sample(net: Verne, frame: Tensor, state: Tensor, steps: int = 10) -> Tensor:
    # noise -> a chunk [B, CHUNK, 8] in std units; the video side runs once
    kvs = net.see(frame, state)
    x = torch.randn(len(frame), CHUNK, 8, device=frame.device)
    for i in range(steps):
        tau = torch.full((len(frame),), i / steps, device=frame.device)
        x = x + net.velocity(kvs, x, tau) / steps
    return x
