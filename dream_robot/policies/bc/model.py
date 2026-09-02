"""The network: two camera streams and a state vector in, one action out.

Deliberately small (~0.6 M parameters). 25 episodes is about 5.3 k frames, and a
ResNet-18 encoder per camera would be 11 M parameters each -- it would memorise
the training episodes in a few hundred steps and teach nothing about whether the
*data* is sufficient, which is the question this rung actually answers.

Nothing here knows what a simulator is, and nothing here hardcodes a dimension:
shapes come from the dataset metadata that ``core.dataset`` writes (ROADMAP
rule 2). A bimanual task is a larger ``action_dim`` and no code change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

#: Dimensions whose std is below this are treated as constant.
#:
#: Normalising by a true std of ~0 turns quantisation noise into a huge input.
#: Every real joint in the recorded dataset has std >= 0.033, so this only ever
#: fires on a degree of freedom that never moved -- which is exactly the case
#: where dividing by the measured spread is meaningless.
STD_FLOOR = 1e-3


@dataclass(frozen=True)
class ModelConfig:
    """Everything needed to rebuild the network, stored in the checkpoint.

    ``state_dim`` / ``action_dim`` / ``cameras`` / ``image_hw`` are read off the
    dataset, never chosen here. The rest are the actual hyperparameters.
    """

    state_dim: int
    action_dim: int
    arm_joints: int
    grippers: int
    cameras: tuple[str, ...]
    image_hw: tuple[int, int]
    conv_channels: tuple[int, ...] = (32, 64, 128, 64)
    keypoints: int = 64
    hidden: tuple[int, int] = (512, 512)
    dropout: float = 0.1
    groups: int = 8

    def to_dict(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "arm_joints": self.arm_joints,
            "grippers": self.grippers,
            "cameras": list(self.cameras),
            "image_hw": list(self.image_hw),
            "conv_channels": list(self.conv_channels),
            "keypoints": self.keypoints,
            "hidden": list(self.hidden),
            "dropout": self.dropout,
            "groups": self.groups,
        }

    @property
    def gripper_slice(self) -> slice:
        """Which action entries are gripper commands. Read, never assumed."""
        return slice(self.arm_joints, self.arm_joints + self.grippers)

    @classmethod
    def from_dict(cls, raw: dict) -> ModelConfig:
        return cls(
            state_dim=int(raw["state_dim"]),
            action_dim=int(raw["action_dim"]),
            arm_joints=int(raw["arm_joints"]),
            grippers=int(raw["grippers"]),
            cameras=tuple(raw["cameras"]),
            image_hw=tuple(raw["image_hw"]),
            conv_channels=tuple(raw["conv_channels"]),
            keypoints=int(raw["keypoints"]),
            hidden=tuple(raw["hidden"]),
            dropout=float(raw["dropout"]),
            groups=int(raw["groups"]),
        )


class Normalizer(nn.Module):
    """Per-dimension ``(x - mean) / std``, and its inverse.

    An ``nn.Module`` with buffers rather than a plain dataclass so the constants
    travel inside ``state_dict`` and move with ``.to(device)``. A checkpoint that
    stored weights but not the normalisation it was fitted under would load
    cleanly and predict nonsense, which is the worst kind of broken.
    """

    def __init__(self, mean, std, *, floor: float = STD_FLOOR):
        super().__init__()
        mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        std = torch.as_tensor(np.asarray(std), dtype=torch.float32).clamp_min(floor)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


class SpatialSoftmax(nn.Module):
    '''Collapse a feature map to the expected image coordinate of each channel.

    The standard visuomotor encoder bottleneck (Levine et al. 2016). The
    alternative -- global average pooling -- answers "how much of feature c is
    in this image", which throws away exactly what manipulation needs. Spatial
    softmax answers "*where* is feature c", and position is what an action is a
    function of.

    Given a feature map f_c(i, j) for channel c over an h x w grid, treat each
    channel as an unnormalised log-density over pixels and take its mean:

        a_c(i, j) = exp(f_c(i, j) / T) / SUM_{i',j'} exp(f_c(i', j') / T)

        x_c = SUM_{i,j} a_c(i, j) * u_j        u_j = -1 + 2j/(w-1)   in [-1, 1]
        y_c = SUM_{i,j} a_c(i, j) * v_i        v_i = -1 + 2i/(h-1)   in [-1, 1]

    Output is 2C numbers: a (x, y) keypoint per channel, in normalised image
    coordinates. C = 64 channels over a 16x16 grid goes from 16384 numbers to
    128 -- a bottleneck the shape of the answer, which is why it needs so little
    data to train.

    Two properties worth knowing:

    - It is **differentiable in position**, unlike an argmax. Moving the cube one
      pixel moves x_c smoothly, so the gradient tells the encoder which way to
      look. An argmax would give the same bottleneck with no gradient.
    - The expectation is over the *whole* map, so a bimodal channel returns the
      midpoint between two blobs -- a coordinate where nothing is. Learned
      temperature T is what lets the network sharpen a channel until it commits
      to one, and it is why T is a parameter here rather than fixed at 1.
    '''

    def __init__(self, height: int, width: int):
        super().__init__()
        # log T rather than T so the parameter is unconstrained and T stays > 0.
        self.log_temperature = nn.Parameter(torch.zeros(()))
        v, u = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        # Flattened to (h*w,) so the expectation is one matmul over pixels.
        self.register_buffer("grid_u", u.reshape(-1))
        self.register_buffer("grid_v", v.reshape(-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        n, c, h, w = features.shape
        flat = features.reshape(n * c, h * w) / self.log_temperature.exp()
        attention = torch.softmax(flat, dim=-1)
        x = attention @ self.grid_u
        y = attention @ self.grid_v
        # (N, C, 2) -> (N, 2C): keypoints stay grouped per channel.
        return torch.stack([x, y], dim=-1).reshape(n, c * 2)


class CameraEncoder(nn.Module):
    """One camera -> 2 * keypoints numbers.

    **GroupNorm, not BatchNorm.** A policy runs one observation at a time in a
    rollout, and BatchNorm's train/eval asymmetry means the network being
    evaluated is not quite the network that was trained -- with a batch of one
    it is either degenerate or running on statistics from a different
    distribution. GroupNorm computes the same thing at batch size 1 and 64, so
    the rollout sees exactly the trained function.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for i, out_channels in enumerate(config.conv_channels):
            stride = 2 if i < len(config.conv_channels) - 1 else 1
            kernel = 5 if i == 0 else 3
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2),
                nn.GroupNorm(min(config.groups, out_channels), out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, config.keypoints, 1))
        self.conv = nn.Sequential(*layers)

        strides = 2 ** (len(config.conv_channels) - 1)
        height = config.image_hw[0] // strides
        width = config.image_hw[1] // strides
        if height < 2 or width < 2:
            raise ValueError(
                f"images of {config.image_hw} collapse to {height}x{width} after "
                f"{len(config.conv_channels)} conv blocks; use fewer blocks"
            )
        self.spatial_softmax = SpatialSoftmax(height, width)
        self.out_features = config.keypoints * 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.spatial_softmax(self.conv(images))


STAT_FIELDS = (
    "state_mean", "state_std", "action_mean", "action_std", "image_mean", "image_std",
)


@dataclass(frozen=True, eq=False)
class NormalizationStats:
    """The constants a checkpoint needs to interpret its own inputs and outputs."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    image_mean: np.ndarray                       # per channel, over [0, 1] pixels
    image_std: np.ndarray

    def to_dict(self) -> dict:
        return {name: np.asarray(getattr(self, name)).tolist() for name in STAT_FIELDS}

    @classmethod
    def from_dict(cls, raw: dict) -> NormalizationStats:
        return cls(**{
            name: np.asarray(raw[name], dtype=np.float32) for name in STAT_FIELDS
        })


class BCPolicyNet(nn.Module):
    """observation -> action. One step, no history, no chunk.

    Everything this class does not have is the point of the rungs above it:
    no action chunking (ACT), no generative head over multimodal actions
    (Diffusion Policy), no temporal context at all. When BC fails on T2 by
    steering the cube into the wall between two valid bins, it fails because of
    the missing piece, not because of a bug.
    """

    def __init__(self, config: ModelConfig, stats: NormalizationStats):
        super().__init__()
        self.config = config
        self.cameras = tuple(config.cameras)

        self.state_norm = Normalizer(stats.state_mean, stats.state_std)
        self.action_norm = Normalizer(stats.action_mean, stats.action_std)
        # Image constants are shaped (3, 1, 1) so they broadcast over (N, 3, H, W).
        self.image_norm = Normalizer(
            np.asarray(stats.image_mean).reshape(3, 1, 1),
            np.asarray(stats.image_std).reshape(3, 1, 1),
        )

        # One encoder per camera, not a shared one. A wrist view and a
        # third-person view have almost nothing in common: the wrist camera's
        # frame moves with the arm, so the same pixel means something different
        # in each. Sharing weights would force one filter bank to serve both.
        self.encoders = nn.ModuleDict(
            {name: CameraEncoder(config) for name in self.cameras}
        )
        vision_features = sum(e.out_features for e in self.encoders.values())

        sizes = [vision_features + config.state_dim, *config.hidden]
        head: list[nn.Module] = []
        for in_features, out_features in zip(sizes[:-1], sizes[1:], strict=True):
            head += [
                nn.Linear(in_features, out_features),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ]
        head.append(nn.Linear(sizes[-1], config.action_dim))
        self.head = nn.Sequential(*head)

    def forward(
        self, images: dict[str, torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        """Normalised action prediction. Raw units come from ``predict``."""
        missing = set(self.cameras) - set(images)
        if missing:
            raise ValueError(f"missing camera(s) {sorted(missing)}; expected {self.cameras}")
        features = [
            self.encoders[name](self.image_norm.normalize(images[name]))
            for name in self.cameras
        ]
        features.append(self.state_norm.normalize(state))
        return self.head(torch.cat(features, dim=-1))

    @torch.no_grad()
    def predict(
        self, images: dict[str, torch.Tensor], state: torch.Tensor
    ) -> torch.Tensor:
        """Action in contract units -- radians and a [0, 1] gripper."""
        return self.action_norm.denormalize(self(images, state))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
