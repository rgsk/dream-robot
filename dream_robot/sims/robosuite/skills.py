"""The arm's primitives: move_to, gripper, and the hand's own reading.

These are what a driver (an LLM, a person) composes into picking, placing, nudging,
recovering. There are no fixed pick/place sequences any more: they were special cases
of these, and their know-how (go down from above, check holding, travel high, release
just over the rim, turn to fit) lives in the driver prompt as recipes.

Two reflexes stay in code because they cannot wait for the next tool call: a move
stops itself when the hand stalls ("blocked"), and the gripper waits until the fingers
settle. Holding is read from the gripper's own width, as a real gripper reports it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from dream_robot.core.env_api import FailureMode, Observation
from dream_robot.sims.robosuite.ik import DifferentialIK
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import EEF_SITE, ExpertConfig

# The Panda's finger bodies: the line between them is the closing direction.
FINGERS = ("gripper0_right_leftfinger", "gripper0_right_rightfinger")



# Primitives: a move counts as blocked when the hand has gone less than this in
# STALL_SECONDS while still short of its target -- it is pressing on something.
STALL_DISTANCE = 0.002
STALL_SECONDS = 0.5
STALL_DEGREES = 1.0
WRIST_MARGIN_DEGREES = 10.0   # keep the last joint this far from its stops when turning
TURN_CHUNK = np.deg2rad(45)   # turn in steps no bigger than this, so the IK keeps the direction


# Gripper opening (0 shut .. 1 open) above which *closed* fingers are on something.
# Shut on nothing reads 1.4-2.0 mm (kitchen, T7); shut on a mug's 3.5 mm rim wall,
# 6.6 mm -- that mug then lifted 14 cm. 0.15 (12 mm) called the rim "nothing", so 0.05
# (4 mm): a wall thinner than ~2 mm still cannot be told from air. Width alone means
# nothing: an open, empty gripper is the widest of all. Holding is "commanded closed,
# and still wider than shut".
HOLDING_MIN_OPENING = 0.05

# One control step: the new observation, and whether the episode's time is up. Nothing
# else -- in particular not the sim's success flag: a real arm does not know whether
# the task is done, the driver judges that from the cameras.
Step = Callable[[np.ndarray], tuple[Observation, bool]]


@dataclass(frozen=True)
class SkillResult:
    ok: bool
    failure: FailureMode          # NONE when ok
    phase: str                    # the phase it finished in, or failed in
    steps: int
    holding: bool                 # read from the gripper's width at the end
    episode_over: bool            # time ran out mid-skill

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "failure": str(self.failure), "phase": self.phase,
            "steps": self.steps, "holding": self.holding, "episode_over": self.episode_over,
        }


class ArmSkills:
    """Primitives for a robosuite single-Panda task. Call ``on_reset`` after every reset."""

    def __init__(self, env, config: ExpertConfig | None = None):
        self._env = env
        self._cfg = config or ExpertConfig.load()
        self._ik: DifferentialIK | None = None
        self._down: np.ndarray | None = None     # reset orientation: gripper pointing down
        self._hold_rotation: np.ndarray | None = None
        self._yaw_rotation: np.ndarray | None = None   # the hold rotation before any tilt
        self._tilt = 0.0                         # commanded tilt, radians: 0 = pointing down
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
        self._yaw_rotation = self._down
        self._tilt = 0.0
        self._finger_direction0 = self._finger_direction()
        self._turned = 0.0                       # total commanded turn since reset, radians
        self._commanded_pos = self._ik.site_pos  # where the pads were last sent
        self._closed = False                     # every reset pose has the gripper open

    def holding(self, obs) -> bool:
        return self._closed and float(obs.state[-1]) > HOLDING_MIN_OPENING

    # --- primitives: the LLM composes these; pick/place are fixed compositions ---

    def hand(self, obs) -> dict:
        """The robot's own reading of its hand: where the finger pads are, the world
        direction from the right finger to the left one, how much the wrist can still turn
        each way, how wide the fingers are open, and whether it holds something."""
        ccw, cw = self._wrist_room()
        return {
            "point": [round(float(c), 4) for c in self._ik.site_pos],
            "yaw_deg": round(float(np.rad2deg(self._finger_direction())), 1),
            "tilt_deg": round(float(np.rad2deg(self._measured_tilt())), 1),
            "wrist_room_deg": {"ccw": round(float(np.rad2deg(ccw)), 1),
                               "cw": round(float(np.rad2deg(cw)), 1)},
            "opening_m": round(float(obs.state[-1]) * self._env._gripper_span, 4),
            "holding": self.holding(obs),
        }

    def move_to(self, point: np.ndarray | None, obs, step: Step,
                yaw: float | None = None, tilt: float | None = None) -> SkillResult:
        """Move the finger pads to ``point`` (None: stay) with the fingers pointing along
        world direction ``yaw`` (None: keep the wrist), the hand tilted by ``tilt``
        (radians, None: keep it).

        ``tilt`` turns the hand about its closing direction: 0 points the fingers straight
        down, +pi/2 horizontal toward yaw + 90 deg (-pi/2: toward yaw - 90). The closing
        direction stays horizontal, so ``yaw`` means the same at any tilt.

        ``yaw`` is a direction, not an axis: 30 and 210 deg are different poses. The arm
        turns whichever way keeps its last wrist joint inside its stops; if no way does,
        nothing moves and the result is "wrist_limit" (``hand.wrist_room_deg`` says what
        is reachable). The gripper stays as it is. Stops by itself when the hand stalls
        short of the target -- pressing on the table or an object -- and reports
        ``blocked``; that reflex cannot wait for the next tool call.
        """
        self._check_ready()
        turn = 0.0
        if yaw is not None:
            d = _wrap_full_turn(yaw - self._commanded_direction())
            ccw, cw = self._wrist_room()
            # The way round with room for it (the room is a lower bound, see _wrist_room);
            # if neither looks clear, the one with more room -- it may still stop short.
            options = (d, d - 2 * np.pi * np.sign(d)) if d else (0.0,)
            fits = [t for t in options if -cw <= t <= ccw]
            turn = min(fits, key=abs) if fits else max(
                options, key=lambda t: (ccw if t > 0 else cw) - abs(t))
        dtilt = 0.0 if tilt is None else float(tilt) - self._tilt
        return self._turn_and_move(point, turn, obs, step, dtilt=dtilt)

    def move_by(self, delta: np.ndarray, obs, step: Step, dyaw: float = 0.0) -> SkillResult:
        """Move the pads by ``delta`` and turn the wrist by exactly ``dyaw`` (radians,
        positive = counter-clockwise seen from above) -- always that way round. A turn
        past the wrist's stops is cut at the stop: the motion completes and the result
        is "wrist_limit"."""
        self._check_ready()
        target = self._commanded_pos + np.asarray(delta, dtype=float)
        return self._turn_and_move(target, dyaw, obs, step)

    def _turn_and_move(self, point, turn: float, obs, step: Step, dtilt: float = 0.0) -> SkillResult:
        """Turn by ``turn``, always that way round, moving the pads to ``point`` meanwhile.

        In chunks of at most TURN_CHUNK: the IK alone would take the short way round. Each
        chunk is sized from the wrist room read *now*, so the turn runs up to the stop and
        no further; if it cannot finish, the result is "wrist_limit" (the move completes).
        """
        target = self._commanded_pos if point is None else np.asarray(point, dtype=float)
        remaining, steps, limited = turn, 0, False
        tilt_left = dtilt
        while True:
            # Tilt in chunks too, alongside the yaw ones: the IK follows small changes.
            tilt_chunk = float(np.sign(tilt_left) * min(abs(tilt_left), TURN_CHUNK))
            tilt_left -= tilt_chunk
            self._tilt += tilt_chunk
            chunk = 0.0
            if abs(remaining) > 1e-6:
                ccw, cw = self._wrist_room()
                room = ccw if remaining > 0 else cw
                chunk = float(np.sign(remaining) * min(abs(remaining), TURN_CHUNK, room))
                if abs(chunk) < np.deg2rad(0.5):
                    limited, chunk = True, 0.0
            self._yaw_rotation = _rot_z(chunk) @ self._yaw_rotation
            self._turned += chunk
            remaining -= chunk
            self._hold_rotation = self._tilted(self._yaw_rotation)
            result = self._servo(target, obs, step)
            steps += result.steps
            obs = self._last_obs
            if not result.ok:
                self._commanded_pos = self._ik.site_pos      # where it stopped
                return SkillResult(False, result.failure, result.phase, steps,
                                   result.holding, result.episode_over)
            if chunk == 0.0 and abs(tilt_left) < 1e-6:
                break
        self._commanded_pos = target
        if limited:
            return SkillResult(False, FailureMode.TIMEOUT, "wrist_limit", steps,
                               self.holding(obs), False)
        return SkillResult(True, FailureMode.NONE, "reached", steps, self.holding(obs), False)

    def _servo(self, target: np.ndarray, obs, step: Step) -> SkillResult:
        """Drive the pads to ``target`` and the wrist to the commanded orientation."""
        gripper = 0.0 if self._closed else 1.0
        hz = self._env.control_hz
        window = int(round(STALL_SECONDS * hz))
        timeout = int(round(self._cfg.phase_timeout_seconds * hz))
        trail = [(self._ik.site_pos, self._ik.site_mat)]
        steps = 0
        self._last_obs = obs
        while True:
            pos_err = float(np.linalg.norm(self._ik.site_pos - target))
            rot_err = _rotation_angle(self._ik.site_mat, self._hold_rotation)
            if pos_err < self._cfg.fine_tolerance and rot_err < np.deg2rad(3):
                return SkillResult(True, FailureMode.NONE, "reached", steps,
                                   self.holding(obs), False)
            if steps >= timeout:
                return SkillResult(False, FailureMode.TIMEOUT, "move", steps,
                                   self.holding(obs), False)
            if len(trail) > window:
                (p1, r1), (p0, r0) = trail[-1], trail[-1 - window]
                # Stalled = neither moving nor turning. Position alone said "blocked" while
                # the wrist was still turning in free air.
                if (np.linalg.norm(p1 - p0) < STALL_DISTANCE
                        and _rotation_angle(r1, r0) < np.deg2rad(STALL_DEGREES)):
                    return SkillResult(False, FailureMode.TIMEOUT, "blocked", steps,
                                       self.holding(obs), False)
            joints = self._ik.solve(target, self._hold_rotation)
            obs, time_up = step(np.concatenate([joints, [gripper]]).astype(np.float32))
            self._last_obs = obs
            steps += 1
            trail.append((self._ik.site_pos, self._ik.site_mat))
            if time_up:
                return SkillResult(False, FailureMode.TIMEOUT, "move", steps,
                                   self.holding(obs), True)

    def _wrist_room(self) -> tuple[float, float]:
        """(ccw, cw): how far the hand may still turn each way, radians, keeping the last
        joint WRIST_MARGIN off its stops (+-166 deg on the Panda). Read from the joint now.

        The joint moves opposite to the hand's yaw, 0.7-1 deg per deg (measured; the
        other joints take up the rest), so assuming 1 is a lower bound: the true room is
        a bit more, and turning in chunks re-reads it, so a turn still reaches the stop.
        """
        model, data = self._env.mujoco
        q_adr = int(self._env.arm_qpos_idx[-1])
        jid = int(np.flatnonzero(model.jnt_qposadr == q_adr)[0])
        lo, hi = model.jnt_range[jid]
        margin = np.deg2rad(WRIST_MARGIN_DEGREES)
        q = float(data.qpos[int(self._env.arm_qpos_idx[-1])])
        return max(0.0, q - lo - margin), max(0.0, hi - margin - q)

    def _commanded_direction(self) -> float:
        """World direction right finger -> left finger once the commanded wrist is reached."""
        return self._finger_direction0 + self._turned

    def _tilted(self, yaw_rotation: np.ndarray) -> np.ndarray:
        """``yaw_rotation`` turned by the commanded tilt about the closing direction."""
        d = self._commanded_direction()
        return _rot_axis(np.array([np.cos(d), np.sin(d), 0.0]), self._tilt) @ yaw_rotation

    def _measured_tilt(self) -> float:
        """Angle between where the fingers point (the grip site's z axis) and straight down."""
        approach = self._ik.site_mat[:, 2]
        return float(np.arccos(np.clip(-approach[2], -1.0, 1.0)))

    def _finger_direction(self) -> float:
        model, data = self._env.mujoco
        left, right = (data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f)]
                       for f in FINGERS)
        return _wrap_full_turn(float(np.arctan2(*(left - right)[1::-1])))

    def _check_ready(self) -> None:
        if self._ik is None:
            raise RuntimeError("call on_reset() after env.reset()")

    def gripper(self, close: bool, obs, step: Step) -> SkillResult:
        """Close or open the fingers where the hand is, and wait until they stop moving.
        Closing on something leaves the fingers apart: ``holding``."""
        command = 0.0 if close else 1.0
        self._closed = close
        hz = self._env.control_hz
        timeout = int(round(self._cfg.phase_timeout_seconds * hz))
        here = self._ik.site_pos
        self._commanded_pos = here
        widths, steps = [], 0
        window, eps = self._cfg.grip_settle_window, self._cfg.grip_settle_epsilon
        while steps < timeout:
            joints = self._ik.solve(here, self._hold_rotation)
            obs, time_up = step(np.concatenate([joints, [command]]).astype(np.float32))
            steps += 1
            widths.append(float(obs.state[-1]))
            if time_up:
                return SkillResult(False, FailureMode.TIMEOUT, "gripper", steps,
                                   self.holding(obs), True)
            if len(widths) > window and np.ptp(widths[-window:]) < eps:
                break
        return SkillResult(True, FailureMode.NONE, "closed" if close else "open", steps,
                           self.holding(obs), False)



def _rot_z(a: float) -> np.ndarray:
    """A world-frame turn about vertical; applied on the left of a hand orientation."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_axis(axis: np.ndarray, a: float) -> np.ndarray:
    """A world-frame rotation by ``a`` about unit ``axis`` (Rodrigues)."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _wrap_full_turn(a: float) -> float:
    """A direction: into [-pi, pi)."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _rotation_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation taking orientation ``b`` to ``a``, radians."""
    c = (np.trace(a @ b.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))
