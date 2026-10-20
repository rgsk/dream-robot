"""The network: each camera seen now and k steps ahead, plus the joints now -> k actions.

**Why regression, not a generative head.** A policy must *choose* (which cube,
which bin, which grasp) and its targets are multimodal, which is what ACT's
latent and flow matching are for. An IDM is shown where the arm ends up, so the
motion in between has essentially one answer. Plain L1 on the chunk is the
right objective, and keeping it plain means a failure here is about the frames,
not about the head.

**Spatial softmax is the right encoder here, for once.** On T5 it lost the cube's
colour (probe B, 52%) while keeping its position. An IDM needs no colour: the
future frame already shows which cube the arm went to. What it needs is *where*
things are in both frames, which is exactly what keypoints are. Both frames of a
camera go through the same encoder, and the head sees now, future, and their
difference: keypoint motion is the action, in image space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from dream_robot.policies.bc.model import (
    CameraEncoder,
    ModelConfig,
    NormalizationStats,
    Normalizer,
)


@dataclass(frozen=True)
class IDMConfig:
    state_dim: int
    action_dim: int
    arm_joints: int
    grippers: int
    cameras: tuple[str, ...]
    image_hw: tuple[int, int]
    k: int                                      # frames ahead = actions predicted
    conv_channels: tuple[int, ...] = (32, 64, 128, 64)
    keypoints: int = 64
    hidden: tuple[int, int] = (512, 512)
    dropout: float = 0.1
    groups: int = 8

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}

    @classmethod
    def from_dict(cls, raw: dict) -> IDMConfig:
        return cls(**{
            k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()
        })

    @property
    def gripper_slice(self) -> slice:
        return slice(self.arm_joints, self.arm_joints + self.grippers)

    def encoder_config(self) -> ModelConfig:
        """BC's encoder settings, so the camera encoder is BC's, unchanged."""
        return ModelConfig(
            state_dim=self.state_dim, action_dim=self.action_dim,
            arm_joints=self.arm_joints, grippers=self.grippers,
            cameras=self.cameras, image_hw=self.image_hw,
            conv_channels=self.conv_channels, keypoints=self.keypoints,
            hidden=self.hidden, dropout=self.dropout, groups=self.groups,
        )


class IDMNet(nn.Module):
    def __init__(self, config: IDMConfig, stats: NormalizationStats):
        super().__init__()
        self.config = config
        self.cameras = tuple(config.cameras)
        self.state_norm = Normalizer(stats.state_mean, stats.state_std)
        self.action_norm = Normalizer(stats.action_mean, stats.action_std)
        self.image_norm = Normalizer(
            np.asarray(stats.image_mean).reshape(3, 1, 1),
            np.asarray(stats.image_std).reshape(3, 1, 1),
        )
        enc = config.encoder_config()
        self.encoders = nn.ModuleDict({name: CameraEncoder(enc) for name in self.cameras})
        vision = 3 * sum(e.out_features for e in self.encoders.values())

        sizes = [vision + config.state_dim, *config.hidden]
        head: list[nn.Module] = []
        for a, b in zip(sizes[:-1], sizes[1:], strict=True):
            head += [nn.Linear(a, b), nn.ReLU(inplace=True), nn.Dropout(config.dropout)]
        head.append(nn.Linear(sizes[-1], config.k * config.action_dim))
        self.head = nn.Sequential(*head)

    def forward(
        self,
        now: dict[str, torch.Tensor],
        future: dict[str, torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Normalised (N, k, action_dim)."""
        features = []
        for name in self.cameras:
            both = torch.cat([now[name], future[name]])          # one encoder pass
            kp_now, kp_future = self.encoders[name](self.image_norm.normalize(both)).chunk(2)
            features += [kp_now, kp_future, kp_future - kp_now]
        features.append(self.state_norm.normalize(state))
        out = self.head(torch.cat(features, dim=-1))
        return out.reshape(-1, self.config.k, self.config.action_dim)

    @torch.no_grad()
    def predict(self, now, future, state) -> torch.Tensor:
        """(N, k, action_dim) in contract units."""
        return self.action_norm.denormalize(self(now, future, state))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
