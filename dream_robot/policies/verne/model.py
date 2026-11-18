"""Step 18 (roadmap box 08): positions by RoPE, as Verne's (wan_dit.py rope_cos_sin / frame_rope, action_dit.py
chunk_rope): each head's queries and keys are rotated by their token's place, video patches by (frame, row,
column) of the canvas grid (the present frame is frame 0; channels 12 / 10 / 10 of the 32), action tokens by
their step in the chunk (all 32 channels). The learned position vectors (video_pos, action_pos) are gone.
Cross-attention rows (words, state) have no place and stay unrotated.
Box 04 kept as step 17e. Tried and dropped (experiments/verne/notes.md, step 17-17f): Wan's small start
on every gate with cross-attention ungated (17: frame-blind 2 of 2), all gates at 1 (17b: 1 of 2), gate 3 on
cross-attention all small (17d: 1 of 1) or all at 1 (17c), the MLP open too (17f: tracks 3 of 3, wrong cube
7-9 vs 17e's 4-6). Rule: the frame's path must start at least as open as the words + state path.
Step 17e: the frame's path gets a head start: self-attention (where the action tokens read the video keys
and values) is gated by 1 + gate, starting near 1, while MLP and cross-attention (words + state) keep Wan's
small start (gate as is, about ±0.1-0.3). Step 17d (all small) went frame-blind, 19/23.
Step 17d: all three gates (attention, MLP, cross-attention) start small, as Wan's (modulation
randn / √dim, time_out at PyTorch's default init, gate used as is, not 1 + gate): every path starts nearly
shut and opens as it helps. Step 17 did this with cross-attention ungated, so words + state started open
and the frame's path shut.
Step 17c: cross-attention (the words + the state) gets its own shift, scale and gate too (9 vectors per
action layer, not 6; all start at 0), so the noise level can turn the state and words up or down as it
can the frame's path. Step 17b (17b seed 1: 20/21) read the frame little at high noise, as did step 17.
Step 17b: every gate starts at 1 and every shift / scale at 0 (gate = 1 + learned; modulation and
time_out's last layer zero at init), so the action side starts exactly as step 16's. Step 17 started them
as Wan does (gates about ±0.1-0.3): the frame's path (self-attention) began nearly shut while the words and
state (cross-attention, ungated) came in at full strength; it read the frame 35-50% less at high noise,
T5 19/22 and 25/25 vs step 16's 42/42 and 36/36.
Step 17 (roadmap box 04): the action side's flow time as Wan's: the noise level σ = 1 − τ, times 1000,
-> 256 sinusoid features -> MLP -> per action layer 6 vectors (shift, scale, gate for attention and MLP:
adaptive layer norm), in place of raw τ through a small MLP added to the action tokens. The video side
is unchanged (no time: its frame is clean).
Step 16: demos train with their recorded prompts only (colour + side); near / far stay in WORDS for the
pretraining's sake but no demo prompt uses them.
Step 15: the video side also edits (pretraining, pretrain.py): given a scene and a colour word, edit_out
reads the patches entering the last video layer (what the last action layer reads) and gives the change
that erases the named cube. The last video layer's own output is never computed (nothing reads it).
Step 14 (captioned denoising: paints right, probe 53%) is gone.
Step 13: near / far join the words (the cube's place: slot 0 is nearer the robot); training prompts
name the T5 cube by colour, place or both (data.reword), and T5 is scored with each.
Step 12 (roadmap box 09): the state as one more cross-attention row on both sides, next to the word
rows, holding the previous control step and now (16 numbers: a sense of velocity); the state token is
gone, so the action side reads the state directly, not through the video layers' keys and values.
Step 11: the instruction's words as cross-attention rows, read by every video and action block
(Wan's block: self-attention -> cross-attention -> MLP). Step 10's future branch is gone (named cube
at chance).
Step 04, the sampler: a video transformer reads the [top | wrist] canvas once, as a VAE latent
grid; an action transformer denoises the chunk while reading the video layers' keys and values.

Verne's shape (wam.py, mot.py), everything else as simple as possible for now:
the two cameras side by side as one wide image, as Verne's canvas (no head crop),
the canvas through a small frozen VAE (vae.py, not Wan's), its latent grid cut into 2x2
squares, one frame (no memory), joints (no poses), the state as a cross-attention row, the instruction as
word rows (each side its own 9-row table: 'no word', then WORDS; no text encoder), masked to the
prompt's words plus 'no word', so a prompt without words (T1) reads 'no word' alone,
the flow time by adaptive norm on the action side only, and Verne's sampler: 4 Euler
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
ROPE_PARTS = (
    12,
    10,
    10,
)  # a head's channels for frame, row, column (Wan's rope_parts: d − 2⌊d/3⌋, ⌊d/3⌋, ⌊d/3⌋)
D_VIDEO, D_ACTION, LAYERS = 256, 128, 4  # the action transformer is the narrow one
FREQ = 256  # sinusoid features of the noise level (Wan's freq_dim)


"""
The noise level as sinusoids (Wan's sinusoidal_embedding_1d): t = 1000·σ, half = FREQ / 2,

    ω_i = 10000^(−i / half),   i = 0 .. half − 1
    feature(t) = [cos(t·ω_0) .. cos(t·ω_{half−1}), sin(t·ω_0) .. sin(t·ω_{half−1})]

ω runs from 1 (a turn every 6.3 units of t: σ 0.90 vs 0.95 is t 900 vs 950, far apart) down to
~1e-4 (one slow, near-linear feature). Raw τ through one SiLU only gives smooth functions of τ.

Adaptive norm (Wan's Block): the time vector e [B, 9, dim] = time_out(time_in(feature)), plus each
layer's own learned modulation [1, 9, dim], gives shift, scale, gate for attention, MLP and (step 17c, not
Wan) cross-attention:

    h = LayerNorm(x)·(1 + scale) + shift          (LayerNorm without its own weights)
    x ← x + (1 + gate) · attention(h)              (step 17e: starts open)
    x ← x + gate'' · cross-attention(LayerNorm(x)·(1 + scale'') + shift'', rows)
    x ← x + gate' · MLP(LayerNorm(x)·(1 + scale') + shift')

(gate, gate', gate'' all start at Wan's random small values, about ±0.1-0.3; steps 17b-c used 1 + gate on all
three, starting at 0)

so the noise level reaches every layer and can turn whole branches up or down, not just add one
vector at the input.
"""


"""
RoPE (Su et al.; Wan's rope_cos_sin, rope_apply). A head's channels in pairs (2i, 2i+1), each pair a point
in the plane, turned by angle p·ω_i, p the token's place on one axis, ω_i = 10000^(−2i/d) over that axis's d
channels:

    (x, y) -> (x·cos − y·sin,  x·sin + y·cos)

Turning a query by p and a key by p' leaves q·k depending on p − p' only (cos(a − b) = cos a cos b + sin a sin
b): attention sees how far apart two tokens are, not where each is. The video side splits its channels
(frame, row, column) and turns each part by that axis's place. The action side turns all channels by the
chunk step; an action query against a video key then compares chunk step with the patch's places, as Verne.
"""


def rope_angles(dim: int, positions: Tensor) -> Tensor:
    # [S] places on one axis -> [S, dim / 2] angles p·ω_i
    w = 10000.0 ** (-torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    return torch.outer(positions.double(), w).float()


def video_angles() -> Tensor:
    # [N_PATCH, HEAD_DIM / 2]: patch (row r, column c) of frame 0, in the patches' order (rows, then columns)
    r, c = torch.meshgrid(torch.arange(16 // P), torch.arange(32 // P), indexing='ij')
    places = (torch.zeros(N_PATCH), r.flatten(), c.flatten())
    return torch.cat([rope_angles(d, pl) for d, pl in zip(ROPE_PARTS, places)], 1)


def rope(x: Tensor, angles: Tensor) -> Tensor:
    # x [B, H, S, D], angles [S, D / 2] -> x with each channel pair turned by its angle (in float32)
    a, b = x.float()[..., 0::2], x.float()[..., 1::2]
    cos, sin = angles.cos(), angles.sin()
    return torch.stack([a * cos - b * sin, a * sin + b * cos], -1).flatten(-2).to(x.dtype)


def sinusoid(t: Tensor) -> Tensor:
    # [B] -> [B, FREQ]
    half = FREQ // 2
    w = torch.pow(10000, -torch.arange(half, dtype=torch.float64, device=t.device) / half)
    a = torch.outer(t.double(), w)
    return torch.cat([torch.cos(a), torch.sin(a)], 1).float()


class Block(nn.Module):
    # a pre-norm transformer layer with cross-attention to the word rows, split in two so a caller
    # can see its keys and values and give it extra ones before attending; adaln: the time vector
    # e [B, 9, dim] (passed to qkv_of and finish) shifts, scales and gates it
    def __init__(self, dim: int, adaln: bool = False):
        super().__init__()
        self.norm1, self.norm2 = (
            nn.LayerNorm(dim, elementwise_affine=not adaln),
            nn.LayerNorm(dim, elementwise_affine=not adaln),
        )
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=not adaln)
        if adaln:
            self.modulation = nn.Parameter(torch.randn(1, 9, dim) / dim**0.5)
        self.qkv = nn.Linear(dim, 3 * HEADS * HEAD_DIM)
        self.out = nn.Linear(HEADS * HEAD_DIM, dim)
        self.cq = nn.Linear(dim, HEADS * HEAD_DIM)  # cross-attention: queries from the tokens,
        self.ckv = nn.Linear(dim, 2 * HEADS * HEAD_DIM)  # keys and values from the rows
        self.cout = nn.Linear(HEADS * HEAD_DIM, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def terms(self, e: Tensor) -> tuple[Tensor, ...]:
        # e [B, 9, dim] -> shift, scale, gate for attention, MLP, cross-attention, each [B, 1, dim]
        return (self.modulation + e).chunk(9, 1)

    def qkv_of(
        self, x: Tensor, angles: Tensor, e: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        # [B, S, dim], the tokens' RoPE angles [S, HEAD_DIM / 2] -> q, k (turned), v, each [B, HEADS, S, HEAD_DIM]
        h = self.norm1(x)
        if e is not None:
            shift, scale = self.terms(e)[:2]
            h = h * (1 + scale) + shift
        qkv = rearrange(self.qkv(h), 'b s (n h d) -> n b h s d', n=3, h=HEADS)
        return rope(qkv[0], angles), rope(qkv[1], angles), qkv[2]

    def finish(
        self,
        x: Tensor,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        rows: Tensor,
        rmask: Tensor,
        e: Tensor | None = None,
    ) -> Tensor:
        # rows [B, 10, dim] the word rows + the state row, rmask [B, 10] bool, True = may attend
        gate, shift2, scale2, gate2, shift3, scale3, gate3 = (
            self.terms(e)[2:] if e is not None else (0, 0, 0, 1, 0, 0, 1)
        )
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + (1 + gate) * self.out(rearrange(a, 'b h s d -> b s (h d)'))
        cq = rearrange(
            self.cq(self.norm3(x) * (1 + scale3) + shift3), 'b s (h d) -> b h s d', h=HEADS
        )
        ck, cv = rearrange(self.ckv(rows), 'b s (n h d) -> n b h s d', n=2, h=HEADS)
        a = F.scaled_dot_product_attention(cq, ck, cv, attn_mask=rmask[:, None, None])
        x = x + gate3 * self.cout(rearrange(a, 'b h s d -> b s (h d)'))
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


def rows_of(
    table: nn.Embedding, state_row: nn.Module, ins: Tensor, state: Tensor
) -> tuple[Tensor, Tensor]:
    # instruction one-hot [B, 8], state [B, 16] -> 10 rows [B, 10, dim] (9 word rows, then the state row)
    # and the mask [B, 10]: 'no word' + the prompt's words + the state
    # (step 11b tried Verne's way, 6 rows with absent words zero and no mask: 13/13, wrong cube 29/26)
    rows = torch.cat([table.weight.expand(len(ins), -1, -1), state_row(state)[:, None]], 1)
    one = torch.ones_like(ins[:, :1])
    return rows, torch.cat([one, ins, one], 1).bool()


def embedding_mlp(dim: int) -> nn.Module:
    # the state [previous, now] [B, 16] -> one row [B, dim] (Verne's embedding_mlp: linear, GELU, linear)
    return nn.Sequential(nn.Linear(16, dim), nn.GELU(), nn.Linear(dim, dim))


class Verne(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_in = nn.Linear(Z * P * P, D_VIDEO)
        # RoPE angles: buffers (move with the model), not weights, not saved
        self.register_buffer('video_angles', video_angles(), persistent=False)
        self.register_buffer(
            'action_angles', rope_angles(HEAD_DIM, torch.arange(CHUNK)), persistent=False
        )
        self.video_words = nn.Embedding(1 + len(WORDS), D_VIDEO)  # row 0 'no word', then WORDS
        self.video_state = embedding_mlp(D_VIDEO)
        self.edit_out = nn.Linear(D_VIDEO, Z * P * P)  # pretraining only: tokens -> latent change
        self.video = nn.ModuleList([Block(D_VIDEO) for _ in range(LAYERS)])
        self.action_in = nn.Linear(8, D_ACTION)
        self.time_in = nn.Sequential(
            nn.Linear(FREQ, D_ACTION), nn.SiLU(), nn.Linear(D_ACTION, D_ACTION)
        )
        self.time_out = nn.Sequential(nn.SiLU(), nn.Linear(D_ACTION, 9 * D_ACTION))  # -> e
        self.action_words = nn.Embedding(1 + len(WORDS), D_ACTION)
        self.action_state = embedding_mlp(D_ACTION)
        self.action = nn.ModuleList([Block(D_ACTION, adaln=True) for _ in range(LAYERS)])
        self.action_out = nn.Linear(D_ACTION, 8)

    def video_side(
        self, frame: Tensor, state: Tensor, ins: Tensor
    ) -> tuple[list[tuple[Tensor, Tensor]], Tensor]:
        # latent [B, Z, 16, 32] (normalised), state [B, 16] (previous, now), instruction [B, 8] -> each
        # video layer's keys and values. Tokens: the 128 patches; words and state enter by cross-attention
        patches = rearrange(frame, 'b c (h p) (w q) -> b (h w) (p q c)', p=P, q=P)
        # -> each layer's keys and values, and the tokens entering the last layer [B, 128, D_VIDEO]
        x = self.patch_in(patches)
        rows, rmask = rows_of(self.video_words, self.video_state, ins, state)
        kvs = []
        for i, block in enumerate(self.video):
            q, k, v = block.qkv_of(x, self.video_angles)
            kvs.append((k, v))
            if i < LAYERS - 1:  # the last layer's output is never read
                x = block.finish(x, q, k, v, rows, rmask)
        return kvs, x

    def see(self, frame: Tensor, state: Tensor, ins: Tensor) -> list[tuple[Tensor, Tensor]]:
        # the policy's read of a clean frame: each video layer's keys and values
        return self.video_side(frame, state, ins)[0]

    def edit(self, frame: Tensor, state: Tensor, ins: Tensor) -> Tensor:
        # pretraining: a clean latent [B, Z, 16, 32] + a colour word -> the change [B, Z, 16, 32] that
        # erases that cube, read from the tokens entering the last video layer
        out = self.edit_out(self.video_side(frame, state, ins)[1])
        return rearrange(out, 'b (h w) (p q c) -> b c (h p) (w q)', h=16 // P, p=P, q=P)

    def velocity(
        self, kvs: list[tuple[Tensor, Tensor]], x: Tensor, tau: Tensor, ins: Tensor, state: Tensor
    ) -> Tensor:
        # noisy chunk x [B, CHUNK, 8] at flow time tau [B], instruction [B, 8], state [B, 16] ->
        # velocity [B, CHUNK, 8]; the action tokens attend to the video tokens of the same layer and
        # to each other, then to the words and the state
        y = self.action_in(x)
        e = self.time_out(self.time_in(sinusoid(1000 * (1 - tau)))).unflatten(1, (9, D_ACTION))
        rows, rmask = rows_of(self.action_words, self.action_state, ins, state)
        for block, (k_video, v_video) in zip(self.action, kvs):
            q, k, v = block.qkv_of(y, self.action_angles, e)
            k, v = torch.cat([k_video, k], 2), torch.cat([v_video, v], 2)
            y = block.finish(y, q, k, v, rows, rmask, e)
        return self.action_out(y)


def loss(net: Verne, frame: Tensor, state: Tensor, ins: Tensor, acts: Tensor) -> Tensor:
    # flow matching: on the line from noise (tau 0) to the chunk (tau 1), predict the direction
    noise = torch.randn_like(acts)
    tau = torch.rand(len(acts), device=acts.device)
    x = (1 - tau[:, None, None]) * noise + tau[:, None, None] * acts
    return F.mse_loss(net.velocity(net.see(frame, state, ins), x, tau, ins, state), acts - noise)


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
        x = x + net.velocity(kvs, x, tau[i].expand(len(frame)), ins, state) * (tau[i + 1] - tau[i])
    return x
