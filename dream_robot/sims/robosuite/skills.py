"""pick(point) and place(bin): the scripted expert's two halves, aimed by perception.

The expert reads the cube's true pose; these do not. ``pick`` is aimed at a
point that ``find_by_colour`` measured from the camera, and whether it worked is
read from the gripper's own width, the way a real gripper reports it. What is
still known rather than sensed: the robot's own kinematics (the IK) and the
bins' positions -- fixtures, like a calibrated workcell's.

Phase machine, IK and every waypoint are the expert's (pick_place_cube/expert.py),
not copies: ``ScriptedExpert`` runs ``PICK_PHASES`` or ``PLACE_PHASES``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dream_robot.core.env_api import FailureMode, StepResult
from dream_robot.sims.robosuite.ik import DifferentialIK
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import (
    EEF_SITE,
    PICK_PHASES,
    PLACE_PHASES,
    ExpertConfig,
    ExpertContext,
    Phase,
    ScriptedExpert,
)

# find returns the centre of the visible top face. The expert's grasp reference
# is the cube's centre, half a cube lower (Lift's cubes: 2.0-2.2 cm half-size).
TOP_TO_GRASP_REFERENCE = 0.021   # metres

# Gripper opening (0 shut .. 1 open) above which *closed* fingers are on something.
# Shut on nothing reads ~0; shut on a 4 cm cube, about half the 8 cm span. Width
# alone means nothing: an open, empty gripper is the widest of all. Holding is
# "commanded closed, and still wide".
HOLDING_MIN_OPENING = 0.15

Step = Callable[[np.ndarray], StepResult]


@dataclass(frozen=True)
class SkillResult:
    ok: bool
    failure: FailureMode          # NONE when ok
    phase: str                    # the phase it finished in, or failed in
    steps: int
    holding: bool                 # read from the gripper's width at the end
    episode_over: bool            # the env ended the episode mid-skill

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "failure": str(self.failure), "phase": self.phase,
            "steps": self.steps, "holding": self.holding, "episode_over": self.episode_over,
        }


class ArmSkills:
    """pick and place for a robosuite single-Panda task. Call ``on_reset`` after every reset."""

    def __init__(self, env, config: ExpertConfig | None = None):
        self._env = env
        self._cfg = config or ExpertConfig.load()
        self._ik: DifferentialIK | None = None
        self._hold_rotation: np.ndarray | None = None
        self._closed = False                     # last gripper command was "close"

    @property
    def bins(self) -> tuple[str, ...]:
        return tuple(self._env.bin_names)

    def on_reset(self) -> None:
        """Rebuild the IK against the fresh MjData; latch gripper-down from the reset pose."""
        model, data = self._env.mujoco
        self._ik = DifferentialIK(
            model, data, site_name=EEF_SITE, qpos_idx=self._env.arm_qpos_idx,
            damping=self._cfg.ik.damping, max_joint_step=self._cfg.ik.max_joint_step,
        )
        self._hold_rotation = self._ik.site_mat
        self._closed = False                     # every reset pose has the gripper open

    def holding(self, obs) -> bool:
        return self._closed and float(obs.state[-1]) > HOLDING_MIN_OPENING

    def pick(self, point: np.ndarray, obs, step: Step) -> SkillResult:
        """Approach, descend, close, lift, at a top-face point from perception."""
        if self.holding(obs):
            raise ValueError("already holding something: place it before picking again")
        reference = np.asarray(point, dtype=float) - [0.0, 0.0, TOP_TO_GRASP_REFERENCE]
        result = self._run(PICK_PHASES, reference, np.zeros(3), obs, step)
        if result.ok and not result.holding:      # lifted, fingers shut on nothing
            return SkillResult(False, FailureMode.NO_GRASP, result.phase, result.steps,
                               False, result.episode_over)
        return result

    def place(self, bin_name: str, obs, step: Step) -> SkillResult:
        """Carry over a bin, lower, open, retreat."""
        if not self.holding(obs):
            return SkillResult(False, FailureMode.NO_GRASP, "start", 0, False, False)
        center = self._env.bin_centers[self.bins.index(bin_name)]
        return self._run(PLACE_PHASES, np.zeros(3), center, obs, step)

    def _run(self, phases: tuple[Phase, ...], cube_ref, bin_center, obs, step) -> SkillResult:
        if self._ik is None:
            raise RuntimeError("call on_reset() after env.reset()")
        machine = ScriptedExpert(self._cfg, control_hz=self._env.control_hz, phases=phases)
        carrying = {p.name for p in PLACE_PHASES[:2]}     # carry, lower: must stay shut on it
        steps = 0
        while True:
            opening = float(obs.state[-1])
            command = machine(ExpertContext(
                eef_pos=self._ik.site_pos, cube_pos=cube_ref, bin_center=bin_center,
                gripper_opening=opening, grip_settled=machine.grip_settled(opening),
            ))
            if command.finished:
                return SkillResult(True, FailureMode.NONE, command.phase, steps,
                                   self.holding(obs), False)
            if command.timed_out:
                return SkillResult(False, FailureMode.TIMEOUT, command.phase, steps,
                                   self.holding(obs), False)
            if command.phase in carrying and not self.holding(obs):
                return SkillResult(False, FailureMode.DROPPED, command.phase, steps,
                                   False, False)
            self._closed = command.gripper < 0.5
            joints = self._ik.solve(command.waypoint, self._hold_rotation)
            result = step(np.concatenate([joints, [command.gripper]]).astype(np.float32))
            obs, steps = result.observation, steps + 1
            if result.terminated or result.truncated:
                return SkillResult(result.success, result.failure_mode, command.phase, steps,
                                   self.holding(obs), True)
