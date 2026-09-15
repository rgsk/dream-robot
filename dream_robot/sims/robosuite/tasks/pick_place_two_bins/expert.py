"""Scripted expert for pick_place_two_bins: T1's expert plus a coin flip.

ROADMAP: "same scene, same expert code -- except the scripted expert flips a
coin on which box". So the phase machine, IK and every waypoint are T1's; the
only new decision is ``target_bin``, drawn once per episode at ``reset()``.

**The coin is not observable, on purpose.** Nothing in the scene says which bin
this episode will use, so from the first frame the demonstrations contain two
different futures for identical-looking states. That is the whole test.

**The coin uses NumPy's global RNG**, which ``env.reset(seed=...)`` has just
seeded (robosuite's placement sampler forces that; see T1's env.py). So the
choice is a pure function of the seed: reproducible, and recoverable later by
resetting the same seed, without being written into the dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import (
    ExpertActor,
    ExpertCommand,
    ExpertConfig,
    ExpertContext,
)
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import (
    ExpertPolicy as _OneBinExpertPolicy,
)

TASK_YAML = Path(__file__).with_name("task.yaml")

__all__ = ["ExpertActor", "ExpertPolicy"]


class ExpertPolicy(_OneBinExpertPolicy):
    """T1's ``ExpertPolicy``, aimed at a bin chosen by coin flip each episode."""

    def __init__(self, env, config: ExpertConfig | None = None):
        super().__init__(env, config or ExpertConfig.load(TASK_YAML))
        self.target_bin: int | None = None

    def reset(self) -> None:
        """Call after env.reset(seed=...). Flips the coin for this episode."""
        super().reset()
        self.target_bin = int(np.random.randint(len(self._env.bin_centers)))

    def __call__(self, observation) -> tuple[np.ndarray, ExpertCommand]:
        if self._ik is None or self.target_bin is None:
            raise RuntimeError("call reset() after env.reset() before stepping")

        opening = float(observation.state[-1])
        ctx = ExpertContext(
            eef_pos=self._ik.site_pos,
            cube_pos=self._cube0,
            bin_center=self._env.bin_centers[self.target_bin],
            gripper_opening=opening,
            grip_settled=self._machine.grip_settled(opening),
        )
        command = self._machine(ctx)
        joints = self._ik.solve(command.waypoint, self._hold_rotation)
        action = np.concatenate([joints, [command.gripper]]).astype(np.float32)
        return action, command
