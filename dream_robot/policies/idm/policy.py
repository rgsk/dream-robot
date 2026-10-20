"""An IDM checkpoint plus a source of future frames, rolled out as a policy.

Satisfies ``core.record.Policy``. The future source is anything with
``reset()`` and ``observe(observation, imagine=bool) -> Observation | None``:
``sims.robosuite.oracle.ExpertFuture`` today, a video model later. This file
never learns which -- the swap in ROADMAP's step 2 is a different object here.

Every ``execute`` steps it asks for a new future and plans k actions toward it,
then plays the first ``execute`` of them. ``execute = k`` (the default) plays
each plan to the frame it was aimed at.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dream_robot.core.dataset import frame_to_float_chw
from dream_robot.core.schema import Observation
from dream_robot.policies.bc.model import NormalizationStats
from dream_robot.policies.idm.model import IDMConfig, IDMNet


class IDMPolicy:
    def __init__(self, net: IDMNet, future, *, device, execute: int | None = None,
                 metadata: dict | None = None):
        self._net = net.to(device).eval()
        self._future = future
        self._device = device
        self.k = net.config.k
        self.execute = execute or self.k
        if not 1 <= self.execute <= self.k:
            raise ValueError(f"execute must be in 1..{self.k}, got {self.execute}")
        self.metadata = metadata or {}
        self.reset_queue()

    @staticmethod
    def load_net(path: Path, device) -> tuple[IDMNet, dict]:
        blob = torch.load(path, map_location=device, weights_only=False)
        net = IDMNet(IDMConfig.from_dict(blob["model_config"]),
                     NormalizationStats.from_dict(blob["normalization"]))
        net.load_state_dict(blob["state_dict"])
        meta = {k: blob[k] for k in ("dataset", "train_config", "epoch", "val_l1_rad",
                                     "val_last_rad") if k in blob}
        return net, meta

    def reset_queue(self) -> None:
        self._queue: list[np.ndarray] = []
        #: The future frames the current plan is aimed at; core.eval films them
        #: as a third column. Kept until the next plan replaces them.
        self.aim_images: dict[str, np.ndarray] | None = None

    def reset(self) -> None:
        self._future.reset()
        self.reset_queue()

    def _images(self, observation: Observation) -> dict[str, torch.Tensor]:
        return {
            name: torch.from_numpy(frame_to_float_chw(observation.images[name]))
            .unsqueeze(0).to(self._device)
            for name in self._net.cameras
        }

    def __call__(self, observation: Observation) -> np.ndarray:
        future = self._future.observe(observation, imagine=not self._queue)
        if future is not None:
            state = torch.from_numpy(
                np.asarray(observation.state, dtype=np.float32)
            ).unsqueeze(0).to(self._device)
            plan = self._net.predict(self._images(observation), self._images(future), state)
            self._queue = list(plan[0, : self.execute].cpu().numpy())
            self.aim_images = {name: future.images[name] for name in self._net.cameras}
        action = self._queue.pop(0).astype(np.float32)
        gripper = self._net.config.gripper_slice
        action[gripper] = np.clip(action[gripper], 0.0, 1.0)
        return action
