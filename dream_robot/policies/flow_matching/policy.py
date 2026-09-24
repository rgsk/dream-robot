"""A trained flow matching checkpoint, wrapped as something that can be rolled out.

Satisfies ``core.record.Policy``. Like ACT it is **stateful** -- it keeps the
unexecuted tail of its current plan -- so ``reset()`` runs between episodes.

**Receding horizon, not temporal ensembling.** ACT re-plans every step and
averages the overlapping plans. For a sampled policy that average is exactly
the failure this rung exists to avoid: one plan heading left and the next
heading right average to straight ahead. So a plan is sampled, its first
``execute`` actions are run open loop, then a fresh plan is sampled (Diffusion
Policy's scheme: horizon 16, execute 8). Consecutive plans usually agree
because the arm is already moving toward one bin, and that is what keeps the
choice committed.

The policy owns its noise: a ``torch.Generator`` seeded at construction, or by
``seed()``. The eval harness seeds the env, not torch, so without this a rerun
of the same eval seeds would roll out different plans.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from dream_robot.core.dataset import frame_to_float_chw
from dream_robot.core.schema import Observation
from dream_robot.policies.bc.model import NormalizationStats
from dream_robot.policies.flow_matching.model import FlowConfig, FlowPolicyNet


class FlowPolicy:
    """Load once, call every control step, reset every episode."""

    def __init__(
        self,
        net: FlowPolicyNet,
        *,
        device: torch.device,
        metadata: dict,
        execute: int | None = None,
        steps: int | None = None,
        seed: int = 0,
    ):
        # Actions run from each plan before re-planning; default half the chunk (8 of 16).
        execute = execute or max(1, net.config.chunk // 2)
        if not 1 <= execute <= net.config.chunk:
            raise ValueError(f"execute must be in [1, {net.config.chunk}], got {execute}")
        self._net = net.to(device).eval()
        self._device = device
        self._execute = execute
        self._steps = steps
        self._generator = torch.Generator(device=device)
        self.metadata = metadata
        self.seed(seed)
        self.reset()

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        device: torch.device | str | None = None,
        execute: int | None = None,
        steps: int | None = None,
        seed: int = 0,
    ) -> FlowPolicy:
        resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        blob = torch.load(path, map_location=resolved, weights_only=False)
        net = FlowPolicyNet(
            FlowConfig.from_dict(blob["model_config"]),
            NormalizationStats.from_dict(blob["normalization"]),
        )
        net.load_state_dict(blob["state_dict"])
        return cls(
            net,
            device=resolved,
            metadata={
                k: blob[k]
                for k in ("dataset", "train_config", "epoch", "val_loss", "val_l1_rad")
                if k in blob
            },
            execute=execute,
            steps=steps,
            seed=seed,
        )

    @property
    def cameras(self) -> tuple[str, ...]:
        return self._net.cameras

    def seed(self, seed: int) -> None:
        """Fix the noise the next plans start from; the sampling test sweeps this."""
        self._generator.manual_seed(seed)

    def reset(self) -> None:
        self._queue: deque[np.ndarray] = deque()

    def __call__(self, observation: Observation) -> np.ndarray:
        if not self._queue:
            self._queue.extend(self._plan(observation)[: self._execute])
        action = self._queue.popleft().copy()
        gripper = self._net.config.gripper_slice
        action[gripper] = np.clip(action[gripper], 0.0, 1.0)
        return action.astype(np.float32)

    def _plan(self, observation: Observation) -> np.ndarray:
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
        plan = self._net.sample(images, state, steps=self._steps, generator=self._generator)
        return plan[0].cpu().numpy()
