"""A trained checkpoint, wrapped as something that can be rolled out.

Satisfies ``core.record.Policy`` -- ``reset()`` and ``__call__(Observation) ->
action`` -- which is the same protocol the scripted expert is recorded through.
That is on purpose: the eval harness rolls out a trained policy exactly the way
the recorder rolled out the expert, so a difference in the numbers is a
difference in the policy and not in the loop around it.

The checkpoint is self-contained. It carries the model config, the normalisation
constants, and the camera names, so loading one never needs the dataset it was
trained on -- which matters because that dataset may have been produced in a
different venv on a different machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dream_robot.core.dataset import frame_to_float_chw
from dream_robot.core.schema import Observation
from dream_robot.policies.bc.model import BCPolicyNet, ModelConfig, NormalizationStats


class BCPolicy:
    """Load once, call every control step.

    Stateless between steps by construction: plain BC has no history, so
    ``reset()`` has nothing to clear. It exists to satisfy the protocol, and
    keeping it here rather than making the protocol optional is what lets ACT --
    which very much does carry state between steps -- drop into the same
    harness later without changing the harness.
    """

    def __init__(self, net: BCPolicyNet, *, device: torch.device, metadata: dict):
        self._net = net.to(device).eval()
        self._device = device
        self.metadata = metadata

    @classmethod
    def load(cls, path: Path, *, device: torch.device | str | None = None) -> BCPolicy:
        resolved = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # weights_only=False: the checkpoint carries its own config and stats
        # dicts, not just tensors. It is a file this repo wrote.
        blob = torch.load(path, map_location=resolved, weights_only=False)
        net = BCPolicyNet(
            ModelConfig.from_dict(blob["model_config"]),
            NormalizationStats.from_dict(blob["normalization"]),
        )
        net.load_state_dict(blob["state_dict"])
        return cls(
            net,
            device=resolved,
            metadata={
                key: blob[key]
                for key in ("dataset", "train_config", "epoch", "val_l1", "val_l1_rad")
                if key in blob
            },
        )

    @property
    def cameras(self) -> tuple[str, ...]:
        return self._net.cameras

    def reset(self) -> None:
        """No history to clear. See the class docstring."""

    def __call__(self, observation: Observation) -> np.ndarray:
        """One absolute joint-target command, in contract units."""
        images = {}
        for name in self._net.cameras:
            if name not in observation.images:
                raise ValueError(
                    f"policy was trained on camera {name!r}, which this environment "
                    f"does not provide; it has {sorted(observation.images)}"
                )
            # Through core.dataset, the same function the training frames came
            # out of, inverted. Doing the divide-by-255 and the transpose by
            # hand here is how a policy ends up being fed pixels it never
            # trained on, with no error and a mysteriously bad success rate.
            images[name] = torch.from_numpy(
                frame_to_float_chw(observation.images[name])
            ).unsqueeze(0).to(self._device)

        state = torch.from_numpy(
            np.asarray(observation.state, dtype=np.float32)
        ).unsqueeze(0).to(self._device)

        action = self._net.predict(images, state)[0].cpu().numpy().astype(np.float32)

        # The gripper is a normalised command and the contract refuses anything
        # outside [0, 1]; a regression head has no idea about that and will
        # overshoot slightly near the ends. The arm joints are deliberately left
        # alone -- an out-of-range joint target is the controller's business,
        # and silently clipping one would hide a policy that has gone somewhere
        # the robot cannot follow.
        gripper = self._net.config.gripper_slice
        action[gripper] = np.clip(action[gripper], 0.0, 1.0)
        return action
