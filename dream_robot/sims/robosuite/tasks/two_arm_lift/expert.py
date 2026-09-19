"""Scripted expert for two_arm_lift -- two phase machines and one barrier.

T1's expert is the template: predicates rather than step budgets, and a pure
phase machine that maps numbers to a waypoint so the decision structure is
testable with no simulator running. Two things are genuinely new here, and both
are the task.

**The barrier.** Each arm has its own phase index. Neither leaves a phase until
*both* arms have satisfied it -- in particular neither lifts until both grippers
have settled closed. Without it, the arm that grasps first starts lifting while
the other is still descending, which tips the tray and spills the marbles: the
exact ``DESYNCHRONISED`` failure the learned policies are supposed to be
measured on. Demonstrating it would be teaching it.

The barrier is a config flag (``expert.sync_barrier``), not a hard-wired one,
because "unsynchronised demos produce desynchronised policies" is a claim this
repo can now measure instead of assert -- turn it off and record a second
dataset.

**The grasp frame is not the reset frame.** T1 held the gripper's reset rotation
all episode: a cube is symmetric about z, so any yaw grasps it. A handle bar is
not, and the tray's yaw is the scene's randomised variable (measured over 12
seeds: 120 degrees of spread). The fingers have to close *across* the bar, so
each arm's wrist yaw is a function of where its own handle ended up. See
``grasp_frame``.

Privileged state -- handle positions, the tray's pose -- is read on purpose, as
in T1: spec.md permits it for a data generator and forbids it in
``observation.*``, which the recorder builds separately and which never sees any
of this.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from dream_robot.sims.robosuite.ik import DifferentialIK
from dream_robot.sims.robosuite.tasks.two_arm_lift.env import EEF_SITES, N_ARMS

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
    reach_tolerance: float
    fine_tolerance: float
    lift_clearance: float
    sync_barrier: bool
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
            reach_tolerance=float(raw["reach_tolerance"]),
            fine_tolerance=float(raw["fine_tolerance"]),
            lift_clearance=float(raw["lift_clearance"]),
            sync_barrier=bool(raw["sync_barrier"]),
            grip_settle_window=int(raw["grip_settle_window"]),
            grip_settle_epsilon=float(raw["grip_settle_epsilon"]),
            phase_timeout_seconds=float(raw["phase_timeout_seconds"]),
            ik=IKConfig(
                damping=float(raw["ik"]["damping"]),
                max_joint_step=float(raw["ik"]["max_joint_step"]),
            ),
        )


@dataclass(frozen=True)
class ArmContext:
    """One arm's privileged view. Numbers only -- no simulator handle."""

    eef_pos: np.ndarray
    handle_pos: np.ndarray      # this arm's own handle, latched at reset
    gripper_opening: float      # normalised [0, 1], 1 open
    grip_settled: bool          # fingers have stopped moving


# Gripper commands, in the contract's units (schema.py: 0 closed .. 1 open).
OPEN, CLOSED = 1.0, 0.0

Waypoint = Callable[[ArmContext, ExpertConfig], np.ndarray]
Predicate = Callable[[ArmContext, ExpertConfig, np.ndarray], bool]


@dataclass(frozen=True)
class Phase:
    name: str
    waypoint: Waypoint
    gripper: float
    reached: Predicate


def _near(ctx: ArmContext, target: np.ndarray, tol: float) -> bool:
    return bool(np.linalg.norm(ctx.eef_pos - target) < tol)


#: Four phases, and every one of them is run by both arms at once. There is no
#: carry or release: success is the tray off the table with its marbles aboard,
#: so the episode ends at the top of the lift.
PHASES: tuple[Phase, ...] = (
    Phase(
        "approach",
        lambda c, k: c.handle_pos + [0.0, 0.0, k.approach_height],
        OPEN,
        lambda c, k, t: _near(c, t, k.reach_tolerance),
    ),
    Phase(
        "descend",
        lambda c, k: c.handle_pos + [0.0, 0.0, k.grasp_dz],
        OPEN,
        lambda c, k, t: _near(c, t, k.fine_tolerance),
    ),
    Phase(
        # Predicate, not a wait: the fingers have stopped closing. Under the
        # barrier this is the phase that matters -- BOTH arms must satisfy it
        # before either leaves, which is what "neither lifts until both have
        # settled closed" means operationally.
        "grasp",
        lambda c, k: c.handle_pos + [0.0, 0.0, k.grasp_dz],
        CLOSED,
        lambda c, k, t: c.grip_settled,
    ),
    Phase(
        "lift",
        lambda c, k: c.handle_pos + [0.0, 0.0, k.grasp_dz + k.lift_height],
        CLOSED,
        lambda c, k, t: bool(c.eef_pos[2] > t[2] - k.lift_clearance),
    ),
)


@dataclass
class ArmCommand:
    waypoint: np.ndarray
    gripper: float
    phase: str
    waiting: bool = False       # own predicate met, held at the barrier
    timed_out: bool = False


@dataclass
class BimanualCommand:
    arms: tuple[ArmCommand, ArmCommand]
    finished: bool = False

    @property
    def timed_out(self) -> bool:
        return any(a.timed_out for a in self.arms)

    @property
    def phase(self) -> str:
        """One label for both arms -- ``"grasp"`` when they agree, else a pair.

        Under the barrier they always agree, and a demo script printing "stuck
        in descend|grasp" is then a visible sign that the barrier is off.
        """
        names = [a.phase for a in self.arms]
        return names[0] if names[0] == names[1] else "|".join(names)


class TwoArmExpert:
    """Two phase machines, coupled by the barrier. Pure: numbers in, waypoints out.

    Two indices rather than one shared index, even though the barrier keeps them
    equal. A single index would make desynchronisation unrepresentable, which
    sounds like a safety property and is actually a measurement loss: the
    ablation (``sync_barrier: false``) is how this repo shows what the barrier
    buys, and the lockstep invariant is worth testing rather than assuming.
    """

    def __init__(self, config: ExpertConfig | None = None, *, control_hz: float = 30.0):
        self._cfg = config or ExpertConfig.load()
        self._timeout_steps = int(round(self._cfg.phase_timeout_seconds * control_hz))
        self.reset()

    def reset(self) -> None:
        self._i = [0] * N_ARMS
        self._steps_in_phase = [0] * N_ARMS
        self._widths: list[list[float]] = [[] for _ in range(N_ARMS)]

    @property
    def config(self) -> ExpertConfig:
        return self._cfg

    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(
            PHASES[i].name if i < len(PHASES) else "done" for i in self._i
        )

    @property
    def finished(self) -> bool:
        return all(i >= len(PHASES) for i in self._i)

    def grip_settled(self, arm: int, opening: float) -> bool:
        """True once this arm's recorded opening has stopped changing.

        Per arm, and cleared on that arm's phase advance: a window shared
        between the arms would let one arm's motion mask the other's.
        """
        widths = self._widths[arm]
        widths.append(float(opening))
        window = self._cfg.grip_settle_window
        if len(widths) <= window:
            return False
        return bool(np.ptp(widths[-window:]) < self._cfg.grip_settle_epsilon)

    def __call__(self, contexts) -> BimanualCommand:
        phases = [PHASES[min(i, len(PHASES) - 1)] for i in self._i]
        targets = [
            np.asarray(p.waypoint(c, self._cfg), dtype=float)
            for p, c in zip(phases, contexts)
        ]
        done = [i >= len(PHASES) for i in self._i]
        reached = [
            True if d else bool(p.reached(c, self._cfg, t))
            for d, p, c, t in zip(done, phases, contexts, targets)
        ]
        # The barrier. An arm advances when it has satisfied its own phase and
        # -- if synchronised -- when the other arm has satisfied its own too.
        clear = all(reached) if self._cfg.sync_barrier else None

        commands = []
        for arm in range(N_ARMS):
            advance = reached[arm] and (clear if clear is not None else True)
            waiting = reached[arm] and not advance
            timed_out = False
            if advance:
                if not done[arm]:
                    self._i[arm] += 1
                self._steps_in_phase[arm] = 0
                self._widths[arm] = []
            elif waiting:
                # An arm held at the barrier is not an arm that is stuck: its
                # own predicate is met and its clock stops. Otherwise a slow
                # partner would report the waiting arm as the failure.
                pass
            else:
                self._steps_in_phase[arm] += 1
                timed_out = self._steps_in_phase[arm] >= self._timeout_steps
            commands.append(
                ArmCommand(
                    waypoint=targets[arm],
                    gripper=phases[arm].gripper,
                    phase="done" if done[arm] else phases[arm].name,
                    waiting=waiting,
                    timed_out=timed_out,
                )
            )
        return BimanualCommand(arms=tuple(commands), finished=self.finished)


def grasp_frame(handle_pos: np.ndarray, tray_center: np.ndarray,
                current_x: np.ndarray | None = None) -> np.ndarray:
    '''Rotation the gripper must hold to close across a handle bar.

    The fingers of a robosuite Panda separate along the grip site's own **x**
    axis (measured, not assumed: experiments/t6/scripts/grasp_geometry.py takes
    the vector between the two finger bodies and expresses it in the site frame,
    getting (1, 0, 0) for both arms). So a grasp needs that axis perpendicular
    to the bar, and the bar runs across the tray -- its long axis is the tray's
    local x, while the handle hangs off the tray's local ±y.

    Take d̂ the horizontal unit vector from the tray centre out to the handle,

        d = handle − centre,   d̂ = (dx, dy, 0) / ‖(dx, dy)‖

    which is ±(the tray's local y), i.e. perpendicular to the bar. Build a
    right-handed frame that also points the gripper straight down:

        x̂ = d̂                    fingers close across the bar
        ẑ = (0, 0, −1)           approach from above
        ŷ = ẑ × x̂                and the frame closes: x̂ × ŷ = ẑ

    R = [x̂ ŷ ẑ] as columns. Both d̂ and −d̂ grasp the same bar, so the sign is
    free; pick the one nearer the wrist's current x axis, which halves the worst
    case yaw travel from 180° to 90° and keeps the wrist away from its limit.
    '''
    d = np.asarray(handle_pos, dtype=float) - np.asarray(tray_center, dtype=float)
    d[2] = 0.0
    norm = np.linalg.norm(d)
    if norm < 1e-9:
        raise ValueError("handle sits on the tray's own axis; no grasp direction")
    x_axis = d / norm
    if current_x is not None and float(np.dot(x_axis, current_x)) < 0.0:
        x_axis = -x_axis
    z_axis = np.array([0.0, 0.0, -1.0])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


class ExpertPolicy:
    """Wires the pure machine to the robosuite task: two IK solvers, one action.

    The only place privileged state, MuJoCo handles and the contract meet, and
    it is on the sim side of the seam -- ``core`` never learns that an IK solver
    exists, let alone two.
    """

    def __init__(self, env, config: ExpertConfig | None = None):
        self._env = env
        self._cfg = config or ExpertConfig.load()
        self._machine = TwoArmExpert(self._cfg, control_hz=env.control_hz)
        self._ik: list[DifferentialIK] | None = None

    def reset(self) -> None:
        """Call after env.reset(). Rebuilds both IKs against the fresh MjData.

        robosuite reloads the model on every reset, so the model and data
        handles from the previous episode are stale -- a solver built once at
        construction silently plans against last episode's scene.
        """
        model, data = self._env.mujoco
        qpos_idx = self._env.arm_qpos_idx
        self._ik = [
            DifferentialIK(
                model, data,
                site_name=EEF_SITES[arm],
                qpos_idx=qpos_idx[arm],
                damping=self._cfg.ik.damping,
                max_joint_step=self._cfg.ik.max_joint_step,
            )
            for arm in range(N_ARMS)
        ]
        # Latched once, like T1's cube: the grasp reference is where the handle
        # was at reset, not wherever the tray has since been nudged to. Chasing
        # a live handle also feeds the tray's own motion back into the waypoint
        # once the arms start lifting it.
        centre = self._env.pot_center
        self._handles = [np.array(h, dtype=float) for h in self._env.handle_pos]
        self._hold_rotation = [
            grasp_frame(self._handles[arm], centre, current_x=self._ik[arm].site_mat[:, 0])
            for arm in range(N_ARMS)
        ]
        self._machine.reset()

    @property
    def phases(self) -> tuple[str, ...]:
        return self._machine.phases

    def __call__(self, observation) -> tuple[np.ndarray, BimanualCommand]:
        """Absolute joint targets for one control step, plus what the arms are doing."""
        if self._ik is None:
            raise RuntimeError("call reset() after env.reset() before stepping")

        openings = np.asarray(observation.state[-N_ARMS:], dtype=float)
        contexts = tuple(
            ArmContext(
                eef_pos=self._ik[arm].site_pos,
                handle_pos=self._handles[arm],
                gripper_opening=float(openings[arm]),
                grip_settled=self._machine.grip_settled(arm, float(openings[arm])),
            )
            for arm in range(N_ARMS)
        )
        command = self._machine(contexts)
        joints = np.concatenate([
            self._ik[arm].solve(command.arms[arm].waypoint, self._hold_rotation[arm])
            for arm in range(N_ARMS)
        ])
        grippers = [command.arms[arm].gripper for arm in range(N_ARMS)]
        return np.concatenate([joints, grippers]).astype(np.float32), command


class ExpertActor:
    """``ExpertPolicy`` narrowed to ``core.record.Policy``: an action, nothing else.

    Same reasoning as T1's: the recorder is not allowed to know what a phase is,
    and ``command.timed_out`` is deliberately dropped so the environment stays
    the single authority on when an episode ended.
    """

    def __init__(self, policy: ExpertPolicy):
        self._policy = policy

    def reset(self) -> None:
        self._policy.reset()

    def __call__(self, observation) -> np.ndarray:
        action, _command = self._policy(observation)
        return action
