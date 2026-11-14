"""Step 11: the instruction's words as cross-attention rows, read by every video and action block
(Wan's block: self-attention -> cross-attention -> MLP). Step 10's future branch is gone (named cube
at chance).
Step 04, the sampler: a video transformer reads the [top | wrist] canvas once, as a VAE latent
grid; an action transformer denoises the chunk while reading the video layers' keys and values.

Verne's shape (wam.py, mot.py), everything else as simple as possible for now:
the two cameras side by side as one wide image, as Verne's canvas (no head crop),
the canvas through a small frozen VAE (vae.py, not Wan's), its latent grid cut into 2x2
squares, one frame (no memory), joints (no poses), the state as one more token, the instruction as
word rows (each side its own 7-row table: 'no word', then WORDS; no text encoder), masked to the
prompt's words plus 'no word', so a prompt without words (T1) reads 'no word' alone,
the flow time added to the action tokens (no adaptive norm), and Verne's sampler: 4 Euler
steps on a grid shifted by 1.5.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .data import CHUNK, WORDS
from .vae import Z

P = 2  # patch side in latent cells: latent 16 x 32 -> 8 x 16 = 128 patches (16 x 16 px each)
N_PATCH = (16 // P) * (32 // P)
HEADS, HEAD_DIM = 4, 32  # shared by both transformers, so their keys and values can mix
D_VIDEO, D_ACTION, LAYERS = 256, 128, 4  # the action transformer is the narrow one


class Block(nn.Module):
    # a pre-norm transformer layer with cross-attention to the word rows, split in two so a caller
    # can see its keys and values and give it extra ones before attending
    def __init__(self, dim: int):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * HEADS * HEAD_DIM)
        self.out = nn.Linear(HEADS * HEAD_DIM, dim)
        self.cq = nn.Linear(dim, HEADS * HEAD_DIM)  # cross-attention: queries from the tokens,
        self.ckv = nn.Linear(dim, 2 * HEADS * HEAD_DIM)  # keys and values from the word rows
        self.cout = nn.Linear(HEADS * HEAD_DIM, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def qkv_of(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # [B, S, dim] -> q, k, v, each [B, HEADS, S, HEAD_DIM]
        qkv = rearrange(self.qkv(self.norm1(x)), 'b s (n h d) -> n b h s d', n=3, h=HEADS)
        return qkv[0], qkv[1], qkv[2]

    def finish(
        self, x: Tensor, q: Tensor, k: Tensor, v: Tensor, rows: Tensor, rmask: Tensor
    ) -> Tensor:
        # rows [B, 7, dim] the word rows, rmask [B, 7] bool, True = may attend
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(rearrange(a, 'b h s d -> b s (h d)'))
        cq = rearrange(self.cq(self.norm3(x)), 'b s (h d) -> b h s d', h=HEADS)
        ck, cv = rearrange(self.ckv(rows), 'b s (n h d) -> n b h s d', n=2, h=HEADS)
        a = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=rmask[:, None, None])
        x = x + self.cout(rearrange(a, 'b h s d -> b s (h d)'))
        return x + self.mlp(self.norm2(x))


def words_of(table: nn.Embedding, ins: Tensor) -> tuple[Tensor, Tensor]:
    # instruction one-hot [B, 6] -> all 7 rows [B, 7, dim] and the mask [B, 7]: 'no word' + the
    # prompt's words
    # (step 11b tried Verne's way, 6 rows with absent words zero and no mask: 13/13, wrong cube 29/26)
    rows = table.weight.expand(len(ins), -1, -1)
    return rows, torch.cat([torch.ones_like(ins[:, :1]), ins], 1).bool()


class Verne(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_in = nn.Linear(Z * P * P, D_VIDEO)
        self.state_in = nn.Linear(8, D_VIDEO)
        self.video_pos = nn.Parameter(torch.randn(1, 1 + N_PATCH, D_VIDEO) * 0.02)
        self.video_words = nn.Embedding(1 + len(WORDS), D_VIDEO)  # row 0 'no word', then WORDS
        self.video = nn.ModuleList([Block(D_VIDEO) for _ in range(LAYERS)])
        self.action_in = nn.Linear(8, D_ACTION)
        self.time_in = nn.Sequential(
            nn.Linear(1, D_ACTION), nn.SiLU(), nn.Linear(D_ACTION, D_ACTION)
        )
        self.action_pos = nn.Parameter(torch.randn(1, CHUNK, D_ACTION) * 0.02)
        self.action_words = nn.Embedding(1 + len(WORDS), D_ACTION)
        self.action = nn.ModuleList([Block(D_ACTION) for _ in range(LAYERS)])
        self.action_out = nn.Linear(D_ACTION, 8)

    def see(self, frame: Tensor, state: Tensor, ins: Tensor) -> list[tuple[Tensor, Tensor]]:
        # latent [B, Z, 16, 32] (normalised), state [B, 8], instruction [B, 6] -> each video layer's
        # keys and values. Tokens: [state, 128 patches]; the words enter by cross-attention only
        patches = rearrange(frame, 'b c (h p) (w q) -> b (h w) (p q c)', p=P, q=P)
        x = torch.cat([self.state_in(state)[:, None], self.patch_in(patches)], 1) + self.video_pos
        rows, rmask = words_of(self.video_words, ins)
        kvs = []
        for block in self.video:
            q, k, v = block.qkv_of(x)
            kvs.append((k, v))
            x = block.finish(x, q, k, v, rows, rmask)
        return kvs

    def velocity(
        self, kvs: list[tuple[Tensor, Tensor]], x: Tensor, tau: Tensor, ins: Tensor
    ) -> Tensor:
        # noisy chunk x [B, CHUNK, 8] at flow time tau [B], instruction [B, 6] -> velocity
        # [B, CHUNK, 8]; the action tokens attend to the video tokens of the same layer and to each
        # other, then to the words
        y = self.action_in(x) + self.action_pos + self.time_in(tau[:, None])[:, None]
        rows, rmask = words_of(self.action_words, ins)
        for block, (k_video, v_video) in zip(self.action, kvs):
            q, k, v = block.qkv_of(y)
            k, v = torch.cat([k_video, k], 2), torch.cat([v_video, v], 2)
            y = block.finish(y, q, k, v, rows, rmask)
        return self.action_out(y)


def loss(net: Verne, frame: Tensor, state: Tensor, ins: Tensor, acts: Tensor) -> Tensor:
    # flow matching: on the line from noise (tau 0) to the chunk (tau 1), predict the direction
    noise = torch.randn_like(acts)
    tau = torch.rand(len(acts), device=acts.device)
    x = (1 - tau[:, None, None]) * noise + tau[:, None, None] * acts
    return F.mse_loss(net.velocity(net.see(frame, state, ins), x, tau, ins), acts - noise)


"""
Verne's sampler (sampler.py): the noise level σ = 1 − τ is stepped down on a shifted grid. With u even
from 1 to 0 (steps + 1 points),

    σ(u) = s·u / (1 + (s − 1)·u)

s = 1 gives σ = u: even steps (steps 0-03 used 10 even steps). s > 1 bends the grid so σ stays near 1 longer:
small steps at high noise, where the chunk's overall shape is decided, bigger ones near clean.
s = 1.5, 4 steps (Verne's test config):

    u   1     0.75   0.5   0.25   0
    σ   1     0.818  0.6   0.333  0         τ = 1 − σ:  0, 0.18, 0.40, 0.67, 1
                                            step sizes:  0.18, 0.22, 0.27, 0.33

Each step is one Euler step of the velocity: x ← x + v(x, τ) · (τ_next − τ).
"""


@torch.no_grad()
def sample(
    net: Verne, frame: Tensor, state: Tensor, ins: Tensor, steps: int = 4, shift: float = 1.5
) -> Tensor:
    # noise -> a chunk [B, CHUNK, 8] in std units; the video side runs once
    kvs = net.see(frame, state, ins)
    x = torch.randn(len(frame), CHUNK, 8, device=frame.device)
    u = torch.linspace(1.0, 0.0, steps + 1, device=frame.device)
    tau = 1 - shift * u / (1 + (shift - 1) * u)  # [steps + 1], 0 -> 1
    for i in range(steps):
        x = x + net.velocity(kvs, x, tau[i].expand(len(frame)), ins) * (tau[i + 1] - tau[i])
    return x
