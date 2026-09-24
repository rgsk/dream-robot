"""Scripted expert for instructed_pick_place: T2's expert, obeying the instruction.

T2's expert flipped a coin for the bin. Here the env has already drawn the
instruction at reset, so the expert reads both halves of it from privileged
state and makes no choice of its own:

- the cube: ``env.cube_pos`` is the *named* cube, so T1's reset latches the
  right one without a line of change;
- the bin: ``env.target_bin``, read where T2 flipped its coin.

The phase machine, IK and every waypoint are T1's.
"""

from __future__ import annotations

from pathlib import Path

from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertConfig
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import (
    ExpertPolicy as _OneBinExpertPolicy,
)
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.expert import (
    ExpertPolicy as _CoinExpertPolicy,
)

TASK_YAML = Path(__file__).with_name("task.yaml")

__all__ = ["ExpertActor", "ExpertPolicy"]


class ExpertPolicy(_CoinExpertPolicy):
    """T2's ``ExpertPolicy`` with the coin replaced by the instruction."""

    def __init__(self, env, config: ExpertConfig | None = None):
        super().__init__(env, config or ExpertConfig.load(TASK_YAML))

    def reset(self) -> None:
        """Call after env.reset(seed=...). Reads this episode's target bin."""
        _OneBinExpertPolicy.reset(self)            # T1's: IK, and latch env.cube_pos
        self.target_bin = self._env.target_bin
