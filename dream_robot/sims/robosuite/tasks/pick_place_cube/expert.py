"""Scripted expert for pick_place_cube -- the data generator.

Two properties this file exists to have.

**It advances on predicates, never on step counts.** ../rl-prac gave each phase
a fixed step budget, which made episodes a rigid 740 steps with phase boundaries
at fixed offsets. "Where am I in the episode" then genuinely predicted the
action, phase and time features got fed to the policy, and the resulting success
rate measured timing recall rather than manipulation. Fixing it here, upstream,
is what makes spec.md's ban on those features cost nothing to hold.

**The phase machine is pure.** ``ScriptedExpert`` maps a context of numbers to a
waypoint and a gripper command. It never touches a simulator, so the whole
decision structure is unit-testable without physics -- see tests/test_expert.py.
Turning the waypoint into joint targets is the IK's job, on the sim side.

The expert reads privileged state (cube pose, bin pose) on purpose. spec.md
permits exactly that: it is a data generator, and privileged access is what
makes it cheap. The prohibition is on ``observation.*``, which the recorder
builds separately and which never sees any of this.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from dream_robot.sims.robosuite.ik import DifferentialIK

TASK_YAML = Path(__file__).with_name("task.yaml")


@dataclass(frozen=True)
class IKConfig:
    damping: float
    max_joint_step: float


@dataclass(frozen=True)
class ExpertConfig:
    approach_height: float
    grasp_dz: float
    lift_height: float
    drop_height: float
    reach_tolerance: float
    fine_tolerance: float
    carry_tolerance: float
    lift_clearance: float
    release_opening: float
    grip_settle_window: int
    grip_settle_epsilon: float
    phase_timeout_seconds: float
    ik: IKConfig

    @classmethod
    def load(cls, path: Path = TASK_YAML) -> ExpertConfig:
        raw = yaml.safe_load(path.read_text())["expert"]
        return cls(
            approach_height=float(raw["approach_height"]),
            grasp_dz=float(raw["grasp_dz"]),
            lift_height=float(raw["lift_height"]),
            drop_height=float(raw["drop_height"]),
            reach_tolerance=float(raw["reach_tolerance"]),
            fine_tolerance=float(raw["fine_tolerance"]),
            carry_tolerance=float(raw["carry_tolerance"]),
            lift_clearance=float(raw["lift_clearance"]),
            release_opening=float(raw["release_opening"]),
            grip_settle_window=int(raw["grip_settle_window"]),
            grip_settle_epsilon=float(raw["grip_settle_epsilon"]),
            phase_timeout_seconds=float(raw["phase_timeout_seconds"]),
            ik=IKConfig(
                damping=float(raw["ik"]["damping"]),
                max_joint_step=float(raw["ik"]["max_joint_step"]),
            ),
        )


@dataclass(frozen=True)
class ExpertContext:
    """The privileged view. Numbers only -- no simulator handle."""

    eef_pos: np.ndarray
    cube_pos: np.ndarray        # grasp reference, latched at reset
    bin_center: np.ndarray
    gripper_opening: float      # normalised [0, 1], 1 open
    grip_settled: bool          # fingers have stopped moving


# Gripper commands, in the contract's units (schema.py: 0 closed .. 1 open).
OPEN, CLOSED = 1.0, 0.0

Waypoint = Callable[[ExpertContext, ExpertConfig], np.ndarray]
Predicate = Callable[[ExpertContext, ExpertConfig, np.ndarray], bool]


@dataclass(frozen=True)
class Phase:
    name: str
    waypoint: Waypoint
    gripper: float
    reached: Predicate


def _near(ctx: ExpertContext, target: np.ndarray, tol: float) -> bool:
    return bool(np.linalg.norm(ctx.eef_pos - target) < tol)


PHASES: tuple[Phase, ...] = (
    Phase(
        "approach",
        lambda c, k: c.cube_pos + [0.0, 0.0, k.approach_height],
        OPEN,
        lambda c, k, t: _near(c, t, k.reach_tolerance),
    ),
    Phase(
        "descend",
        lambda c, k: c.cube_pos + [0.0, 0.0, k.grasp_dz],
        OPEN,
        lambda c, k, t: _near(c, t, k.fine_tolerance),
    ),
    Phase(
        # Predicate, not a wait: the fingers have stopped closing.
        "grasp",
        lambda c, k: c.cube_pos + [0.0, 0.0, k.grasp_dz],
        CLOSED,
        lambda c, k, t: c.grip_settled,
    ),
    Phase(
        "lift",
        lambda c, k: c.cube_pos + [0.0, 0.0, k.grasp_dz + k.lift_height],
        CLOSED,
        lambda c, k, t: bool(c.eef_pos[2] > t[2] - k.lift_clearance),
    ),
    Phase(
        "carry",
        lambda c, k: np.array([c.bin_center[0], c.bin_center[1],
                               c.bin_center[2] + k.lift_height]),
        CLOSED,
        lambda c, k, t: _near(c, t, k.carry_tolerance),
    ),
    Phase(
        "lower",
        lambda c, k: np.array([c.bin_center[0], c.bin_center[1],
                               c.bin_center[2] + k.drop_height]),
        CLOSED,
        lambda c, k, t: _near(c, t, k.reach_tolerance),
    ),
    Phase(
        "release",
        lambda c, k: np.array([c.bin_center[0], c.bin_center[1],
                               c.bin_center[2] + k.drop_height]),
        OPEN,
        lambda c, k, t: c.gripper_opening > k.release_opening,
    ),
    Phase(
        "retreat",
        lambda c, k: np.array([c.bin_center[0], c.bin_center[1],
                               c.bin_center[2] + k.lift_height]),
        OPEN,
        lambda c, k, t: bool(c.eef_pos[2] > t[2] - k.lift_clearance),
    ),
)


@dataclass
class ExpertCommand:
    waypoint: np.ndarray
    gripper: float
    phase: str
    finished: bool = False
    timed_out: bool = False


class ScriptedExpert:
    """The phase machine. Pure: numbers in, waypoint out."""

    def __init__(self, config: ExpertConfig | None = None, *, control_hz: float = 30.0):
        self._cfg = config or ExpertConfig.load()
        self._timeout_steps = int(round(self._cfg.phase_timeout_seconds * control_hz))
        self._widths: list[float] = []
        self.reset()

    def reset(self) -> None:
        self._i = 0
        self._steps_in_phase = 0
        self._widths = []

    @property
    def config(self) -> ExpertConfig:
        return self._cfg

    @property
    def phase(self) -> str:
        return PHASES[self._i].name if self._i < len(PHASES) else "done"

    def grip_settled(self, opening: float) -> bool:
        """True once the recorded opening has stopped changing.

        Kept here rather than in the caller so the history lives with the state
        machine that consumes it, and so the test suite can drive it directly.
        """
        self._widths.append(float(opening))
        window = self._cfg.grip_settle_window
        if len(self._widths) <= window:
            return False
        return bool(np.ptp(self._widths[-window:]) < self._cfg.grip_settle_epsilon)

    def __call__(self, ctx: ExpertContext) -> ExpertCommand:
        if self._i >= len(PHASES):
            last = PHASES[-1]
            return ExpertCommand(
                waypoint=np.asarray(last.waypoint(ctx, self._cfg), dtype=float),
                gripper=last.gripper,
                phase="done",
                finished=True,
            )

        phase = PHASES[self._i]
        target = np.asarray(phase.waypoint(ctx, self._cfg), dtype=float)
        self._steps_in_phase += 1

        if phase.reached(ctx, self._cfg, target):
            self._i += 1
            self._steps_in_phase = 0
            self._widths = []
            return ExpertCommand(target, phase.gripper, phase.name,
                                 finished=self._i >= len(PHASES))

        if self._steps_in_phase >= self._timeout_steps:
            return ExpertCommand(target, phase.gripper, phase.name, timed_out=True)

        return ExpertCommand(target, phase.gripper, phase.name)


# The end-effector site the IK drives. robosuite/Panda name; it stays on the sim
# side of the seam and nothing downstream ever sees it.
EEF_SITE = "gripper0_right_grip_site"


class ExpertPolicy:
    """Wires the pure phase machine to the robosuite task.

    Privileged state and MuJoCo handles in, contract-shaped actions out. This is
    the only place the two halves meet, and it is on the sim side of the seam:
    ``core`` never learns that an IK solver exists.
    """

    def __init__(self, env, config: ExpertConfig | None = None):
        self._env = env
        self._cfg = config or ExpertConfig.load()
        self._machine = ScriptedExpert(self._cfg, control_hz=env.control_hz)
        self._ik: DifferentialIK | None = None

    def reset(self) -> None:
        """Call after env.reset(). Rebuilds the IK against the fresh MjData."""
        model, data = self._env.mujoco
        self._ik = DifferentialIK(
            model,
            data,
            site_name=EEF_SITE,
            qpos_idx=self._env.arm_qpos_idx,
            damping=self._cfg.ik.damping,
            max_joint_step=self._cfg.ik.max_joint_step,
        )
        # Latched once: the grasp reference is where the cube was at reset, not
        # wherever it has been nudged to since. Also the orientation to hold --
        # the reset pose already points the gripper down.
        self._cube0 = self._env.cube_pos
        self._hold_rotation = self._ik.site_mat
        self._machine.reset()

    @property
    def phase(self) -> str:
        return self._machine.phase

    def __call__(self, observation) -> tuple[np.ndarray, ExpertCommand]:
        """Absolute joint targets for one control step, plus what the expert is doing.

        ``observation`` is the contract observation -- the expert reads the
        gripper opening from it in contract units rather than reaching into the
        simulator for finger joints in metres.
        """
        if self._ik is None:
            raise RuntimeError("call reset() after env.reset() before stepping")

        opening = float(observation.state[-1])
        ctx = ExpertContext(
            eef_pos=self._ik.site_pos,
            cube_pos=self._cube0,
            bin_center=self._env.bin_center,
            gripper_opening=opening,
            grip_settled=self._machine.grip_settled(opening),
        )
        command = self._machine(ctx)
        joints = self._ik.solve(command.waypoint, self._hold_rotation)
        action = np.concatenate([joints, [command.gripper]]).astype(np.float32)
        return action, command


class ExpertActor:
    """``ExpertPolicy`` narrowed to ``core.record.Policy``: an action, nothing else.

    The recorder is not allowed to know what a phase is. ``ExpertPolicy``
    returns ``(action, ExpertCommand)`` because ``demo.py`` wants to print which
    phase an episode got stuck in, and that is a debugging affordance of this
    task, not part of the interface a dataset is recorded through -- teleop and
    a trained policy have no phase to report and must not have to invent one.

    Note what this deliberately drops: ``command.timed_out``. The expert giving
    up is a faster way to end an episode the environment would have truncated
    anyway, and honouring it would give the recorder a second opinion about when
    an episode ended, competing with the environment's. It is worth about ten
    seconds across a fifty-episode recording, which is not worth two authorities
    on the same question.
    """

    def __init__(self, policy: ExpertPolicy):
        self._policy = policy

    def reset(self) -> None:
        self._policy.reset()

    def __call__(self, observation) -> np.ndarray:
        action, _command = self._policy(observation)
        return action
