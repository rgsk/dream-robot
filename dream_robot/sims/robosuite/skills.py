"""pick(grasp) and place_at(point): the scripted expert's two halves, aimed by perception.

The expert reads the cube's true pose; these do not. ``pick`` is aimed by a
grasp that perception measured from the camera -- where the finger pads close,
along which direction, and how high that is above the table -- and whether it
worked is read from the gripper's own width, the way a real gripper reports it.
``place_at`` goes to a point perception found (a container's opening), not to a
named bin. What is still known rather than sensed: the robot's own kinematics.

Phase machine, IK and every waypoint are the expert's (pick_place_cube/expert.py),
not copies: ``ScriptedExpert`` runs ``PICK_PHASES`` or ``PLACE_PHASES``. What is
new is the wrist: the expert always held the reset orientation, which only
suits a cube; pick turns it about vertical so the fingers close across the
object's narrow side, and carries with that orientation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
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

# The Panda's finger bodies: the line between them is the closing direction.
FINGERS = ("gripper0_right_leftfinger", "gripper0_right_rightfinger")

# place_at lets go when the held object's bottom is this far above the drop
# point: it falls the rest, rather than being pushed into the rim.
RELEASE_GAP = 0.02

# The finger pads' centre cannot go lower than this over the table: the fingertips
# reach past it and touch down at ~2.0 cm (measured: descend stalls at 0.820 on
# the 0.800 table). Lower grasps -- a 3 cm box's mid-height -- are raised to it.
MIN_ABOVE_TABLE = 0.025

# A grasp with no height above the table (find_by_colour's) is assumed to hang
# like a T5 cube held at its centre.
DEFAULT_ABOVE_TABLE = 0.021

# Gripper opening (0 shut .. 1 open) above which *closed* fingers are on something.
# Shut on nothing reads ~0; shut on a 4 cm cube, about half the 8 cm span. Width
# alone means nothing: an open, empty gripper is the widest of all. Holding is
# "commanded closed, and still wide".
HOLDING_MIN_OPENING = 0.15

Step = Callable[[np.ndarray], StepResult]


@dataclass(frozen=True)
class Grasp:
    point: np.ndarray             # world xyz where the finger pads should close
    yaw: float | None             # world direction the fingers close along, radians; None: as reset
    above_table: float | None     # point's height over the table: how far the object hangs


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
        self._down: np.ndarray | None = None     # reset orientation: gripper pointing down
        self._close_yaw0 = 0.0                   # its closing direction, world yaw
        self._hold_rotation: np.ndarray | None = None
        self._hang = DEFAULT_ABOVE_TABLE         # how far the held object hangs below the pads
        self._closed = False                     # last gripper command was "close"

    def on_reset(self) -> None:
        """Rebuild the IK against the fresh MjData; latch gripper-down from the reset pose."""
        model, data = self._env.mujoco
        self._ik = DifferentialIK(
            model, data, site_name=EEF_SITE, qpos_idx=self._env.arm_qpos_idx,
            damping=self._cfg.ik.damping, max_joint_step=self._cfg.ik.max_joint_step,
        )
        self._down = self._ik.site_mat
        self._hold_rotation = self._down
        left, right = (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f)]
                       for f in FINGERS)
        self._close_yaw0 = float(np.arctan2(*(left - right)[1::-1]))
        self._closed = False                     # every reset pose has the gripper open

    def holding(self, obs) -> bool:
        return self._closed and float(obs.state[-1]) > HOLDING_MIN_OPENING

    def pick(self, grasp: Grasp, obs, step: Step) -> SkillResult:
        """Turn the wrist, approach, descend, close, lift, as the grasp says."""
        if self.holding(obs):
            raise ValueError("already holding something: place it before picking again")
        # A closing direction is an axis: turn the short way, never more than 90 degrees.
        # No yaw: keep the reset wrist, as the expert always did.
        turn = 0.0 if grasp.yaw is None else (grasp.yaw - self._close_yaw0 + np.pi / 2) % np.pi - np.pi / 2
        c, s = np.cos(turn), np.sin(turn)
        self._hold_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self._down
        point = np.asarray(grasp.point, dtype=float).copy()
        self._hang = DEFAULT_ABOVE_TABLE
        if grasp.above_table is not None:
            raise_by = max(0.0, MIN_ABOVE_TABLE - grasp.above_table)
            point[2] += raise_by
            self._hang = grasp.above_table + raise_by
        # The phase machine descends to reference + grasp_dz; land the pads on the point.
        reference = point - [0.0, 0.0, self._cfg.grasp_dz]
        result = self._run(PICK_PHASES, reference, np.zeros(3), obs, step)
        if result.ok and not result.holding:      # lifted, fingers shut on nothing
            return SkillResult(False, FailureMode.NO_GRASP, result.phase, result.steps,
                               False, result.episode_over)
        return result

    def place_at(self, point: np.ndarray, obs, step: Step) -> SkillResult:
        """Carry above ``point``, lower until the held object's bottom is just over it,
        open, retreat. ``point`` is where the object should land: a container's
        opening at rim height, or a spot on the table."""
        if not self.holding(obs):
            return SkillResult(False, FailureMode.NO_GRASP, "start", 0, False, False)
        release_z = float(point[2]) + self._hang + RELEASE_GAP
        # The phase machine lowers to bin_center + drop_height; aim that at release_z.
        target = np.array([point[0], point[1], release_z - self._cfg.drop_height])
        return self._run(PLACE_PHASES, np.zeros(3), target, obs, step)

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
