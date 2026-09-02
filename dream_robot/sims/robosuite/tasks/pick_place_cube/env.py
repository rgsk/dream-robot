"""pick_place_cube on robosuite -- the reference sim adapter.

This is the first task in the repo, so its shape is the shape every later task
copies. Three structural choices are the point of it:

1. **The task wraps robosuite; it does not inherit from it.** ``PickPlaceCube``
   holds a robosuite env in a field and implements ``core.env_api.Env``. The
   only subclass here is ``_LiftWithBin``, which exists solely to inject MJCF
   into the scene -- that is robosuite's supported extension point, and it is
   scene construction, not the task interface. An Isaac version of this task is
   a sibling file, not a special case of a robosuite class.

2. **The adapter is where backend names die.** Camera renaming, the image flip,
   gripper normalisation, and the action convention all happen here. Nothing
   downstream ever sees ``agentview`` or a +1/-1 gripper command.

3. **Every number comes from task.yaml.** None are literals in this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import robosuite as suite
import yaml
from robosuite.environments.manipulation.lift import Lift
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_geom

from dream_robot.core.env_api import FailureMode, StepResult
from dream_robot.core.schema import (
    PANDA,
    Embodiment,
    Observation,
    build_observation,
    validate_action,
)

TASK_YAML = Path(__file__).with_name("task.yaml")


@dataclass(frozen=True)
class BinSpec:
    offset_xy: tuple[float, float]
    inner_half: float
    wall_half_thickness: float
    wall_half_height: float


@dataclass(frozen=True)
class TaskConfig:
    prompt: str
    control_hz: float
    horizon_seconds: float
    camera_hw: tuple[int, int]
    camera_mapping: dict[str, str]
    render_camera: str
    render_hw: tuple[int, int]
    cube_xy_range: tuple[float, float]
    bin: BinSpec
    resting_z_margin: float
    lift_threshold: float
    dropped_radius: float

    @classmethod
    def load(cls, path: Path = TASK_YAML) -> TaskConfig:
        raw = yaml.safe_load(path.read_text())
        b = raw["scene"]["bin"]
        return cls(
            prompt=str(raw["prompt"]),
            control_hz=float(raw["control"]["hz"]),
            horizon_seconds=float(raw["control"]["horizon_seconds"]),
            camera_hw=(int(raw["cameras"]["height"]), int(raw["cameras"]["width"])),
            camera_mapping=dict(raw["cameras"]["mapping"]),
            render_camera=str(raw["render"]["camera"]),
            render_hw=(int(raw["render"]["height"]), int(raw["render"]["width"])),
            cube_xy_range=tuple(raw["scene"]["cube_xy_range"]),
            bin=BinSpec(
                offset_xy=tuple(b["offset_xy"]),
                inner_half=float(b["inner_half"]),
                wall_half_thickness=float(b["wall_half_thickness"]),
                wall_half_height=float(b["wall_half_height"]),
            ),
            resting_z_margin=float(raw["success"]["resting_z_margin"]),
            lift_threshold=float(raw["failure"]["lift_threshold"]),
            dropped_radius=float(raw["failure"]["dropped_radius"]),
        )

    @property
    def horizon_steps(self) -> int:
        return int(round(self.horizon_seconds * self.control_hz))


class _LiftWithBin(Lift):
    """robosuite's Lift scene plus a static open-top bin. MJCF only."""

    def __init__(self, bin_spec: BinSpec, **kwargs):
        self._bin = bin_spec          # must exist before _load_model runs
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        super()._load_model()
        b = self._bin
        cx = self.table_offset[0] + b.offset_xy[0]
        cy = self.table_offset[1] + b.offset_xy[1]
        body = new_body(
            name="goal_bin",
            pos=array_to_string([cx, cy, self.table_offset[2] + b.wall_half_height]),
        )
        span = b.inner_half + b.wall_half_thickness
        t, h = b.wall_half_thickness, b.wall_half_height
        for suffix, size, pos in (
            ("px", [t, span, h], [span, 0, 0]),
            ("nx", [t, span, h], [-span, 0, 0]),
            ("py", [span, t, h], [0, span, 0]),
            ("ny", [span, t, h], [0, -span, 0]),
        ):
            body.append(
                new_geom(
                    name=f"goal_bin_{suffix}", type="box",
                    size=array_to_string(size), pos=array_to_string(pos),
                    rgba=array_to_string([0.15, 0.35, 0.75, 1.0]),
                    # group=1, NOT new_geom's default of 0. robosuite treats
                    # group 0 as collision-only and its cameras do not draw it,
                    # so group 0 yields a bin the cube bounces off that no
                    # policy can see -- physics and pixels disagreeing, with the
                    # success predicate passing anyway.
                    group=1,
                    friction=array_to_string([1.0, 0.005, 0.0001]),
                )
            )
        self.model.worldbody.append(body)

    def bin_center(self) -> np.ndarray:
        return np.array([
            self.table_offset[0] + self._bin.offset_xy[0],
            self.table_offset[1] + self._bin.offset_xy[1],
            self.table_offset[2],
        ])


def _joint_position_controller_config() -> dict:
    """Absolute joint targets -- spec.md's action contract, natively.

    robosuite's JointPositionController with input_type="absolute" assigns
    goal_qpos = action verbatim, and nothing in the step path clips it. The
    default BASIC config is OSC, which is end-effector deltas and banned.
    """
    cfg = suite.load_composite_controller_config(controller="BASIC", robot="Panda")
    # The loader flattens body_parts.arms.{right,left} -> body_parts.{right,left}
    cfg["body_parts"]["right"] = {
        "type": "JOINT_POSITION",
        "input_type": "absolute",
        "input_max": 1, "input_min": -1,
        "output_max": 0.05, "output_min": -0.05,
        "kp": 150, "damping_ratio": 1,
        "impedance_mode": "fixed",           # required for absolute input
        "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
        "qpos_limits": None,
        "interpolation": None, "ramp_ratio": 0.2,
        "gripper": {"type": "GRIP"},
    }
    return cfg


class PickPlaceCube:
    """Implements ``core.env_api.Env``. Structurally, without inheriting it."""

    def __init__(self, config: TaskConfig | None = None):
        self._cfg = config or TaskConfig.load()
        ch, cw = self._cfg.camera_hw
        self._backend_cams = list(self._cfg.camera_mapping)
        self._env = _LiftWithBin(
            self._cfg.bin,
            robots="Panda",
            controller_configs=_joint_position_controller_config(),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=self._backend_cams,
            camera_heights=ch,
            camera_widths=cw,
            control_freq=self._cfg.control_hz,
            horizon=self._cfg.horizon_steps,
        )
        self._raw: dict | None = None
        self._t = 0
        self._max_cube_z = -np.inf
        self._gripper_span = self._measure_gripper_span()

    # --- Env protocol ------------------------------------------------------

    @property
    def embodiment(self) -> Embodiment:
        return PANDA

    @property
    def control_hz(self) -> float:
        return float(self._cfg.control_hz)

    @property
    def prompt(self) -> str:
        """The instruction a language-conditioned policy is given."""
        return self._cfg.prompt

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            # robosuite's placement samplers call np.random.uniform directly, so
            # the global RNG is the only seeding hook there is. Documented in
            # this sim's README as a backend quirk, not a design choice.
            np.random.seed(seed)
        self._raw = self._env.reset()
        self._t = 0
        # Reference height for "was it ever lifted", taken from the cube itself.
        # Measuring against the table instead is wrong: robosuite's placement
        # sampler spawns with a z_offset, so the cube already sits above the
        # table surface and an idle episode reads as a successful grasp.
        self._cube_z0 = float(self._raw["cube_pos"][2])
        self._max_cube_z = self._cube_z0
        return self._observe()

    def step(self, action) -> StepResult:
        act = validate_action(action, PANDA)
        arm = act[PANDA.arm_slice]
        opening = float(act[PANDA.gripper_slice][0])
        # Contract: 0 closed .. 1 open. robosuite GRIP: -1 open .. +1 close.
        self._raw, _, _, _ = self._env.step(np.concatenate([arm, [1.0 - 2.0 * opening]]))
        self._t += 1

        cube = np.asarray(self._raw["cube_pos"], dtype=float)
        self._max_cube_z = max(self._max_cube_z, float(cube[2]))

        success = self._cube_in_bin(cube)
        truncated = (not success) and self._t >= self._cfg.horizon_steps
        return StepResult(
            observation=self._observe(),
            success=success,
            terminated=success,
            truncated=truncated,
            failure_mode=self._diagnose(cube) if truncated else FailureMode.NONE,
        )

    def render(self) -> np.ndarray:
        h, w = self._cfg.render_hw
        frame = self._env.sim.render(
            width=w, height=h, camera_name=self._cfg.render_camera
        )
        return np.ascontiguousarray(frame[::-1])

    def close(self) -> None:
        self._env.close()

    # --- privileged access, for scripted experts only ----------------------
    #
    # spec.md permits the expert to read ground truth -- it is a data generator,
    # and privileged access is what makes it cheap. These are deliberately
    # separate methods rather than extra fields on Observation, so that the path
    # a policy sees and the path an expert sees cannot be confused for each
    # other. _observe() below never touches any of them.

    @property
    def cube_pos(self) -> np.ndarray:
        return np.asarray(self._raw["cube_pos"], dtype=float).copy()

    @property
    def bin_center(self) -> np.ndarray:
        return self._env.bin_center()

    @property
    def mujoco(self):
        """(model, data). Re-read after every reset -- the handles go stale."""
        return self._env.sim.model._model, self._env.sim.data._data

    @property
    def arm_qpos_idx(self) -> np.ndarray:
        return np.asarray(self._env.robots[0]._ref_joint_pos_indexes)

    # --- adapter internals -------------------------------------------------

    def _measure_gripper_span(self) -> float:
        """Max finger separation, read from the model rather than hardcoded."""
        m = self._env.sim.model._model
        names = self._env.robots[0].gripper["right"].joints
        lo_hi = np.array([
            m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names
        ])
        span = float(lo_hi[0][1] - lo_hi[1][0])
        if span <= 0:
            raise ValueError(f"non-positive gripper span {span} from joints {names}")
        return span

    def _observe(self) -> Observation:
        raw = self._raw
        width = float(raw["robot0_gripper_qpos"][0] - raw["robot0_gripper_qpos"][1])
        return build_observation(
            joint_positions=raw["robot0_joint_pos"],
            gripper_openings=[np.clip(width / self._gripper_span, 0.0, 1.0)],
            images={
                canonical: np.ascontiguousarray(raw[f"{backend}_image"][::-1])
                # robosuite renders bottom-up (OpenGL row order). Unflipped, a
                # policy trains happily on upside-down frames and nothing
                # complains until they meet a real camera or another simulator.
                for backend, canonical in self._cfg.camera_mapping.items()
            },
            embodiment=PANDA,
        )

    def _cube_in_bin(self, cube: np.ndarray) -> bool:
        c = self._env.bin_center()
        inside = bool(np.all(np.abs(cube[:2] - c[:2]) < self._cfg.bin.inner_half))
        resting = bool(cube[2] < c[2] + self._cfg.resting_z_margin)
        return inside and resting

    def _diagnose(self, cube: np.ndarray) -> FailureMode:
        """Why this episode failed, in the closed vocabulary of FailureMode."""
        c = self._env.bin_center()
        lifted = self._max_cube_z > self._cube_z0 + self._cfg.lift_threshold
        if not lifted:
            return FailureMode.NO_GRASP
        distance = float(np.linalg.norm(cube[:2] - c[:2]))
        if distance > self._cfg.dropped_radius:
            return FailureMode.DROPPED
        return FailureMode.WRONG_TARGET
