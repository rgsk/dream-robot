"""The network: two camera streams and a state vector in, a chunk of future actions out.

ACT (Zhao et al. 2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost
Hardware") in the shape of ``policies/bc``. Three pieces BC does not have:

- **A chunk.** The decoder emits the next ``chunk`` actions at once, one learned
  query per future step. At rollout ``policy.py`` re-plans every step and
  averages the overlapping plans, which is what smooths out a single bad guess.
- **A transformer over image features.** Each camera becomes a grid of feature
  tokens rather than BC's 2 x keypoints coordinates, so the model can attend to
  where things are relative to each other.
- **A CVAE latent.** During training a small encoder reads the true action chunk
  and summarises its "style" into ``latent`` numbers the decoder is conditioned
  on. At rollout there is no true chunk, so the latent is zero -- the mean of
  the prior. It exists to stop two different ways of doing the same thing from
  being averaged into a third way that does neither.

Sized for this repo's data, not the paper's: BC's small from-scratch conv stack
instead of a pretrained ResNet-18 per camera, and dim 256 instead of 512. The
comparison against BC should be about chunking and attention, not about
ImageNet features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dream_robot.policies.bc.model import NormalizationStats, Normalizer


@dataclass(frozen=True)
class ACTConfig:
    """Everything needed to rebuild the network, stored in the checkpoint.

    The first six fields are read off the dataset (ROADMAP rule 2); the rest are
    hyperparameters. Defaults: chunk 50 steps is ~1.7 s at 30 Hz, against the
    paper's 100 steps at 50 Hz (2 s). Latent size, heads and the single decoder
    layer follow LeRobot's ACT defaults. The 8x8 image grid and 2 encoder / 2 CVAE
    layers are measured choices: ~6x faster per step than a 16x16 grid with 4 / 4
    (experiments/act/notes.md), since attention over image tokens dominated.
    """

    state_dim: int
    action_dim: int
    arm_joints: int
    grippers: int
    cameras: tuple[str, ...]
    image_hw: tuple[int, int]
    chunk: int = 50
    dim: int = 256
    heads: int = 8
    feedforward: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 1
    vae_layers: int = 2
    latent: int = 32
    conv_channels: tuple[int, ...] = (32, 64, 128, 128)
    groups: int = 8
    dropout: float = 0.1

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
            "feedforward": self.feedforward,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "vae_layers": self.vae_layers,
            "latent": self.latent,
            "conv_channels": list(self.conv_channels),
            "groups": self.groups,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> ACTConfig:
        ints = (
            "state_dim",
            "action_dim",
            "arm_joints",
            "grippers",
            "chunk",
            "dim",
            "heads",
            "feedforward",
            "encoder_layers",
            "decoder_layers",
            "vae_layers",
            "latent",
            "groups",
        )
        return cls(
            **{name: int(raw[name]) for name in ints},
            cameras=tuple(raw["cameras"]),
            image_hw=tuple(raw["image_hw"]),
            conv_channels=tuple(raw["conv_channels"]),
            dropout=float(raw["dropout"]),
        )

    @property
    def gripper_slice(self) -> slice:
        return slice(self.arm_joints, self.arm_joints + self.grippers)


def sinusoid_2d(height: int, width: int, dim: int) -> torch.Tensor:
    """Fixed ``(height * width, dim)`` position code: half the channels for rows, half for columns.

    Fixed rather than learned because 25 demonstrations are not enough to learn
    a good position code for 256 grid cells, and a transformer with no position
    information at all cannot tell the cube's left edge from its right.
    """
    if dim % 4:
        raise ValueError(f"dim must be divisible by 4 for a 2-D sinusoid, got {dim}")
    quarter = dim // 4
    freqs = torch.exp(-math.log(10000.0) * torch.arange(quarter, dtype=torch.float32) / quarter)
    rows = torch.arange(height, dtype=torch.float32)[:, None] * freqs
    cols = torch.arange(width, dtype=torch.float32)[:, None] * freqs
    row_code = torch.cat([rows.sin(), rows.cos()], dim=-1)  # (H, dim/2)
    col_code = torch.cat([cols.sin(), cols.cos()], dim=-1)  # (W, dim/2)
    grid = torch.cat(
        [
            row_code[:, None, :].expand(height, width, dim // 2),
            col_code[None, :, :].expand(height, width, dim // 2),
        ],
        dim=-1,
    )
    return grid.reshape(height * width, dim)


class Backbone(nn.Module):
    """One camera -> a ``(h * w, dim)`` grid of feature tokens.

    BC's conv stack (GroupNorm, for the same batch-of-one reason) without its
    spatial softmax: the transformer wants the grid, not a summary of it.
    """

    def __init__(self, config: ACTConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for i, out_channels in enumerate(config.conv_channels):
            kernel = 5 if i == 0 else 3
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel, 2, kernel // 2),
                nn.GroupNorm(min(config.groups, out_channels), out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, config.dim, 1))
        self.conv = nn.Sequential(*layers)
        stride = 2 ** len(config.conv_channels)
        self.grid = (config.image_hw[0] // stride, config.image_hw[1] // stride)
        if min(self.grid) < 2:
            raise ValueError(
                f"images of {config.image_hw} collapse to {self.grid}; use fewer blocks"
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv(images).flatten(2).transpose(1, 2)


def _encoder(config: ACTConfig, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        config.dim, config.heads, config.feedforward, config.dropout, batch_first=True
    )
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class ACTPolicyNet(nn.Module):
    """observation -> ``(chunk, action_dim)`` actions, plus ``(mu, logvar)`` in training."""

    def __init__(self, config: ACTConfig, stats: NormalizationStats):
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

        # --- observation side: [latent, state, camera tokens...] -> encoder ---
        self.backbones = nn.ModuleDict({name: Backbone(config) for name in self.cameras})
        height, width = next(iter(self.backbones.values())).grid
        self.register_buffer("image_pos", sinusoid_2d(height, width, dim), persistent=False)
        # Both cameras share one position grid, so a learned per-camera offset is
        # what tells the model which view a token came from.
        self.camera_embed = nn.Parameter(torch.randn(len(self.cameras), dim) * 0.02)
        self.state_in = nn.Linear(config.state_dim, dim)
        self.latent_in = nn.Linear(config.latent, dim)
        self.extra_pos = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.encoder = _encoder(config, config.encoder_layers)

        # --- action side: one learned query per future step -> decoder ---
        self.queries = nn.Parameter(torch.randn(config.chunk, dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            dim, config.heads, config.feedforward, config.dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.action_out = nn.Linear(dim, config.action_dim)

        # --- CVAE encoder, training only: [cls, state, true chunk] -> latent ---
        self.vae_cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.vae_state = nn.Linear(config.state_dim, dim)
        self.vae_action = nn.Linear(config.action_dim, dim)
        self.vae_pos = nn.Parameter(torch.randn(config.chunk + 2, dim) * 0.02)
        self.vae = _encoder(config, config.vae_layers)
        self.latent_out = nn.Linear(dim, 2 * config.latent)

    def forward(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
        pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Normalised chunk prediction ``(B, chunk, action_dim)``, and ``(mu, logvar)``.

        With ``actions`` (training), the latent is sampled from the CVAE encoder's
        reading of the true chunk; ``pad`` marks chunk steps past the episode end,
        which the encoder must not read. Without (rollout), the latent is zero and
        ``mu`` / ``logvar`` are None.
        """
        missing = set(self.cameras) - set(images)
        if missing:
            raise ValueError(f"missing camera(s) {sorted(missing)}; expected {self.cameras}")
        batch = state.shape[0]
        state = self.state_norm.normalize(state)

        if actions is not None:
            tokens = (
                torch.cat(
                    [
                        self.vae_cls.expand(batch, -1, -1),
                        self.vae_state(state)[:, None],
                        self.vae_action(self.action_norm.normalize(actions)),
                    ],
                    dim=1,
                )
                + self.vae_pos
            )
            mask = None
            if pad is not None:
                mask = torch.cat(
                    [torch.zeros(batch, 2, dtype=torch.bool, device=pad.device), pad], dim=1
                )
            summary = self.vae(tokens, src_key_padding_mask=mask)[:, 0]
            mu, logvar = self.latent_out(summary).chunk(2, dim=-1)
            latent = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            mu = logvar = None
            latent = state.new_zeros(batch, self.config.latent)

        tokens = [
            torch.stack([self.latent_in(latent), self.state_in(state)], dim=1) + self.extra_pos
        ]
        for i, name in enumerate(self.cameras):
            grid = self.backbones[name](self.image_norm.normalize(images[name]))
            tokens.append(grid + self.image_pos + self.camera_embed[i])
        memory = self.encoder(torch.cat(tokens, dim=1))
        decoded = self.decoder(self.queries.expand(batch, -1, -1), memory)
        return self.action_out(decoded), mu, logvar

    @torch.no_grad()
    def predict(self, images: dict[str, torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """``(B, chunk, action_dim)`` in contract units, latent at the prior mean."""
        chunk, _, _ = self(images, state)
        return self.action_norm.denormalize(chunk)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z | chunk) || N(0, I)), summed over latent dims and averaged over the batch."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=-1).mean()


def masked_l1(predicted: torch.Tensor, target: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over real chunk steps only; padded steps are repeats, not demos."""
    keep = (~pad).unsqueeze(-1).to(predicted.dtype)
    return ((predicted - target).abs() * keep).sum() / (keep.sum() * predicted.shape[-1]).clamp_min(
        1
    )
