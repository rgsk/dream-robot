"""The network: observation in, a velocity field over action chunks out.

Diffusion Policy (Chi et al. 2023) with the diffusion replaced by **flow
matching** (Lipman et al. 2023; Liu et al. 2023, "rectified flow"), the
objective pi0 and most 2025-26 action heads train with. BC and ACT regress one
answer per observation; this model learns a *distribution* over answers and
draws from it, which is the only way to say "left bin or right bin" without
saying "between the bins".

The objective, with x0 ~ N(0, I) a noise chunk and x1 a demo chunk (normalised):

    x_t = (1 - t) x0 + t x1              straight line from noise to data, t in [0, 1]
    dx_t/dt = x1 - x0                    its velocity, constant along the line
    loss = || v_θ(x_t, t | obs) - (x1 - x0) ||²

At a given (x_t, t) many lines pass through, so the MSE minimiser is the
*average* velocity of the lines through that point, E[x1 - x0 | x_t]. Near
t = 0 that average points at the middle of the modes; as t grows, x_t already
sits closer to one mode and the average points there. Integrating
dx/dt = v_θ from a random x0 therefore ends on a demo-like chunk, and which one
is decided by where x0 started. Sampling:

    x ← x0 ~ N(0, I)
    for k in 0 .. K-1:   x ← x + (1/K) v_θ(x, k/K | obs)        # Euler, K = 10

No noise schedule, no alphas and betas, no variance-preserving bookkeeping --
that is the whole difference from DDPM, and why few steps suffice: the learned
paths are nearly straight.

Architecture:
- **Observation encoder: BC's.** One spatial-softmax conv stack per camera, plus
  the normalised state, squeezed to one ``dim`` vector. Keeps the comparison
  with BC and ACT about the action head.
- **Velocity net: a small DiT** (Peebles & Xie 2023). One token per chunk step;
  observation and time enter every block through adaLN-zero (a per-block
  scale, shift and gate predicted from the conditioning vector, gates
  initialised to zero so each block starts as the identity). Diffusion
  Policy's other option is a 1-D conv U-Net with FiLM; the transformer is what
  current action experts use and is less code.
- **Optional target words** (T5). ``condition_words`` is a fixed vocabulary;
  each prompt becomes a 0/1 vector over it (``condition_vector``) joined to the
  observation features. With ``red green blue yellow left right`` that is a
  colour one-hot plus a side one-hot: the target handed over as an index, with
  no language model in the way. Empty (the default) is the unconditioned net.
- **``film``: the target inside the encoders.** Joined after the spatial
  softmax, the one-hot arrives too late: the keypoints are already positions,
  and T5's colour probe (experiments/t5, step 11-12) showed that encoder never
  learns which cube is red. With ``film`` every conv block's normalised output
  becomes (1 + γ(c)) h + β(c), so the maps know the wanted colour before the
  softmax picks where they fire. Probe: 52% -> 100%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dream_robot.policies.bc.model import CameraEncoder, NormalizationStats, Normalizer


@dataclass(frozen=True)
class FlowConfig:
    """Everything needed to rebuild the network, stored in the checkpoint.

    The first six fields are read off the dataset (ROADMAP rule 2). ``chunk`` 16
    at 30 Hz is ~0.5 s, Diffusion Policy's horizon; ``policy.py`` executes the
    first half of each plan. ``conv_channels`` / ``keypoints`` / ``groups`` feed
    BC's ``CameraEncoder`` unchanged, so the vision side matches BC exactly.
    """

    state_dim: int
    action_dim: int
    arm_joints: int
    grippers: int
    cameras: tuple[str, ...]
    image_hw: tuple[int, int]
    chunk: int = 16
    dim: int = 256
    heads: int = 4
    layers: int = 4
    feedforward: int = 1024
    conv_channels: tuple[int, ...] = (32, 64, 128, 64)
    keypoints: int = 64
    groups: int = 8
    sample_steps: int = 10
    condition_words: tuple[str, ...] = ()
    film: bool = False

    def to_dict(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "arm_joints": self.arm_joints,
            "grippers": self.grippers,
            "cameras": list(self.cameras),
            "image_hw": list(self.image_hw),
            "chunk": self.chunk,
            "dim": self.dim,
            "heads": self.heads,
            "layers": self.layers,
            "feedforward": self.feedforward,
            "conv_channels": list(self.conv_channels),
            "keypoints": self.keypoints,
            "groups": self.groups,
            "sample_steps": self.sample_steps,
            "condition_words": list(self.condition_words),
            "film": self.film,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> FlowConfig:
        ints = (
            "state_dim",
            "action_dim",
            "arm_joints",
            "grippers",
            "chunk",
            "dim",
            "heads",
            "layers",
            "feedforward",
            "keypoints",
            "groups",
            "sample_steps",
        )
        return cls(
            **{name: int(raw[name]) for name in ints},
            cameras=tuple(raw["cameras"]),
            image_hw=tuple(raw["image_hw"]),
            conv_channels=tuple(raw["conv_channels"]),
            condition_words=tuple(raw.get("condition_words", ())),
            film=bool(raw.get("film", False)),
        )

    @property
    def gripper_slice(self) -> slice:
        return slice(self.arm_joints, self.arm_joints + self.grippers)


def condition_vector(prompt: str, words: tuple[str, ...]) -> np.ndarray:
    """``(len(words),)`` float32: 1 where the word appears in the prompt as a whole word."""
    tokens = set(prompt.lower().replace(",", " ").replace(".", " ").split())
    return np.asarray([word in tokens for word in words], dtype=np.float32)


class FiLMCameraEncoder(nn.Module):
    """BC's ``CameraEncoder`` with a condition vector modulating every conv block.

    Same layers and spatial softmax. After each GroupNorm, h -> (1 + γ) h + β with
    (γ, β) = Linear(condition): a per-channel scale and shift, so which maps light
    up depends on what was asked for. Default (random) init, as in the probe that
    reached 100%: small random γ, β already differ per colour at step 0.
    """

    def __init__(self, config: FlowConfig):
        super().__init__()
        base = CameraEncoder(config)
        layers = list(base.conv)
        # conv, norm, relu per block; then the 1x1 conv to keypoint maps
        self.blocks = nn.ModuleList(
            nn.ModuleList(layers[i : i + 3]) for i in range(0, len(layers) - 1, 3)
        )
        self.keypoint_conv = layers[-1]
        self.film = nn.ModuleList(
            nn.Linear(len(config.condition_words), 2 * block[0].out_channels)
            for block in self.blocks
        )
        self.spatial_softmax = base.spatial_softmax
        self.out_features = base.out_features

    def forward(self, images: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x = images
        for (conv, norm, relu), film in zip(self.blocks, self.film, strict=True):
            gamma, beta = film(condition)[:, :, None, None].chunk(2, dim=1)
            x = relu((1 + gamma) * norm(conv(x)) + beta)
        return self.spatial_softmax(self.keypoint_conv(x))


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """``(B,)`` times in [0, 1] -> ``(B, dim)`` sinusoids.

    Scaled by 1000 first so the fastest frequency still turns over many times
    across [0, 1]; without it every t would look alike to the network.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    angles = 1000.0 * t[:, None] * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    """Pre-norm transformer block whose norms are steered by the conditioning vector."""

    def __init__(self, dim: int, heads: int, feedforward: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, feedforward), nn.GELU(), nn.Linear(feedforward, dim)
        )
        # shift, scale, gate for attention and for the MLP. Zero init: the gates
        # start at 0, so at step 0 the block passes x through untouched.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), shift1, scale1)
        x = x + gate1[:, None] * self.attn(h, h, h, need_weights=False)[0]
        h = _modulate(self.norm2(x), shift2, scale2)
        return x + gate2[:, None] * self.mlp(h)


class FlowPolicyNet(nn.Module):
    """``(noisy chunk, t, observation)`` -> velocity, and a sampler built on it."""

    def __init__(self, config: FlowConfig, stats: NormalizationStats):
        super().__init__()
        self.config = config
        self.cameras = tuple(config.cameras)
        dim = config.dim

        self.state_norm = Normalizer(stats.state_mean, stats.state_std)
        self.action_norm = Normalizer(stats.action_mean, stats.action_std)
        self.image_norm = Normalizer(
            np.asarray(stats.image_mean).reshape(3, 1, 1),
            np.asarray(stats.image_std).reshape(3, 1, 1),
        )

        # --- observation -> one conditioning vector ---
        if config.film and not config.condition_words:
            raise ValueError("film needs condition_words")
        encoder = FiLMCameraEncoder if config.film else CameraEncoder
        self.encoders = nn.ModuleDict({name: encoder(config) for name in self.cameras})
        vision = sum(e.out_features for e in self.encoders.values())
        self.obs_proj = nn.Sequential(
            nn.Linear(vision + config.state_dim + len(config.condition_words), dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        # --- noisy chunk -> velocity ---
        self.action_in = nn.Linear(config.action_dim, dim)
        self.pos = nn.Parameter(torch.randn(config.chunk, dim) * 0.02)
        self.blocks = nn.ModuleList(
            Block(dim, config.heads, config.feedforward) for _ in range(config.layers)
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.action_out = nn.Linear(dim, config.action_dim)
        for layer in (self.ada_out[1], self.action_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def encode(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, dim)`` observation vector, computed once per plan and reused by every step.

        ``condition`` is ``(B, len(condition_words))``; required iff the net has words.
        """
        missing = set(self.cameras) - set(images)
        if missing:
            raise ValueError(f"missing camera(s) {sorted(missing)}; expected {self.cameras}")
        if self.config.condition_words and condition is None:
            raise ValueError(
                f"this net is conditioned on {list(self.config.condition_words)}; pass condition"
            )
        film = (condition,) if self.config.film else ()
        features = [
            self.encoders[name](self.image_norm.normalize(images[name]), *film)
            for name in self.cameras
        ]
        features.append(self.state_norm.normalize(state))
        if self.config.condition_words:
            features.append(condition)
        return self.obs_proj(torch.cat(features, dim=-1))

    def velocity(self, x: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """``v_θ(x_t, t | obs)``: ``(B, chunk, action_dim)`` in normalised action units."""
        cond = obs + self.time_proj(time_embedding(t, self.config.dim))
        h = self.action_in(x) + self.pos
        for block in self.blocks:
            h = block(h, cond)
        shift, scale = self.ada_out(cond).chunk(2, dim=-1)
        return self.action_out(_modulate(self.norm_out(h), shift, scale))

    def loss(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        actions: torch.Tensor,
        pad: torch.Tensor,
        generator: torch.Generator | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Flow matching MSE on one batch of demo chunks (contract units in).

        t is uniform on [0, 1]. pi0 skews it toward the noisy end with a Beta;
        that is a knob to try if sampling proves weak, not a default.
        ``generator`` fixes x0 and t, so validation loss is comparable across epochs.
        """
        x1 = self.action_norm.normalize(actions)
        x0 = torch.randn(x1.shape, generator=generator, device=x1.device)
        t = torch.rand(x1.shape[0], generator=generator, device=x1.device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        predicted = self.velocity(xt, t, self.encode(images, state, condition))
        return masked_mse(predicted, x1 - x0, pad)

    @torch.no_grad()
    def sample(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        *,
        steps: int | None = None,
        generator: torch.Generator | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Draw one ``(B, chunk, action_dim)`` plan per observation, in contract units.

        Euler from t = 0 (noise) to t = 1 (data). ``generator`` picks the starting
        noise, and with it the mode: same generator state, same plan.
        """
        steps = steps or self.config.sample_steps
        obs = self.encode(images, state, condition)
        batch = state.shape[0]
        x = torch.randn(
            (batch, self.config.chunk, self.config.action_dim),
            generator=generator,
            device=state.device,
        )
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((batch,), k * dt, device=state.device)
            x = x + dt * self.velocity(x, t, obs)
        return self.action_norm.denormalize(x)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_mse(predicted: torch.Tensor, target: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Mean squared error over real chunk steps only; padded steps are repeats, not demos.

    MSE, not BC/ACT's L1: the flow objective needs the *mean* velocity at each
    point, and the mean is what squared error's minimiser is. With L1 the
    minimiser is the median velocity, and integrating it no longer carries noise
    onto the data distribution.
    """
    keep = (~pad).unsqueeze(-1).to(predicted.dtype)
    return ((predicted - target).pow(2) * keep).sum() / (
        keep.sum() * predicted.shape[-1]
    ).clamp_min(1)
