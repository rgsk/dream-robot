"""A trained ACT checkpoint, wrapped as something that can be rolled out.

Satisfies ``core.record.Policy`` like BC does. The difference is that it is
**stateful**: it re-plans a whole chunk every step and keeps the recent plans,
so ``reset()`` must be called between episodes -- the harness already does.

That internal step counter is the policy's own bookkeeping, not an observation
feature. spec.md bans time in ``observation.*`` because a recorded clock can
predict the expert's actions; here the counter only indexes which of the
policy's own past plans refer to the current step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dream_robot.core.dataset import frame_to_float_chw
from dream_robot.core.schema import Observation
from dream_robot.policies.act.model import ACTConfig, ACTPolicyNet
from dream_robot.policies.bc.model import NormalizationStats

#: Temporal-ensembling decay, the paper's m. Weight exp(-m * i) on the i-th
#: oldest plan still covering this step. 0.01 over a 50-step chunk is nearly a
#: plain average, tilted slightly toward the older, already-committed plans.
ENSEMBLE_DECAY = 0.01


class ACTPolicy:
    """Load once, call every control step, reset every episode."""

    def __init__(
        self,
        net: ACTPolicyNet,
        *,
        device: torch.device,
        metadata: dict,
        ensemble_decay: float = ENSEMBLE_DECAY,
    ):
        self._net = net.to(device).eval()
        self._device = device
        self._decay = ensemble_decay
        self.metadata = metadata
        self.reset()

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: torch.device | str | None = None,
        ensemble_decay: float = ENSEMBLE_DECAY,
    ) -> ACTPolicy:
        resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        blob = torch.load(path, map_location=resolved, weights_only=False)
        net = ACTPolicyNet(
            ACTConfig.from_dict(blob["model_config"]),
            NormalizationStats.from_dict(blob["normalization"]),
        )
        net.load_state_dict(blob["state_dict"])
        return cls(
            net,
            device=resolved,
            metadata={
                k: blob[k]
                for k in ("dataset", "train_config", "epoch", "val_l1", "val_l1_rad")
                if k in blob
            },
            ensemble_decay=ensemble_decay,
        )

    @property
    def cameras(self) -> tuple[str, ...]:
        return self._net.cameras

    def reset(self) -> None:
        self._step = 0
        self._plans: list[tuple[int, np.ndarray]] = []  # (step planned at, chunk)

    def __call__(self, observation: Observation) -> np.ndarray:
        images = {}
        for name in self._net.cameras:
            if name not in observation.images:
                raise ValueError(
                    f"policy was trained on camera {name!r}, which this environment "
                    f"does not provide; it has {sorted(observation.images)}"
                )
            images[name] = (
                torch.from_numpy(frame_to_float_chw(observation.images[name]))
                .unsqueeze(0)
                .to(self._device)
            )
        state = (
            torch.from_numpy(np.asarray(observation.state, dtype=np.float32))
            .unsqueeze(0)
            .to(self._device)
        )

        plan = self._net.predict(images, state)[0].cpu().numpy()
        self._plans.append((self._step, plan))
        horizon = plan.shape[0]
        self._plans = [(start, p) for start, p in self._plans if self._step - start < horizon]

        # Oldest first, so weight exp(-m * i) favours plans already under way.
        predictions = np.stack([p[self._step - start] for start, p in self._plans])
        weights = np.exp(-self._decay * np.arange(len(predictions)))
        action = (weights[:, None] * predictions).sum(axis=0) / weights.sum()
        self._step += 1

        gripper = self._net.config.gripper_slice
        action[gripper] = np.clip(action[gripper], 0.0, 1.0)
        return action.astype(np.float32)
