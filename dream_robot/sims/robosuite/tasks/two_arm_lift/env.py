"""two_arm_lift on robosuite -- the first bimanual task (ROADMAP T6).

Same shape as ``pick_place_cube``: wrap robosuite, implement ``core.env_api.Env``,
read every number from ``task.yaml``. Three things are new, and they are the
point of the task:

1. **The embodiment is a value, not a constant.** ``BIMANUAL_PANDA`` is
   ``Embodiment(arm_joints=14, grippers=2)`` -> ``dim == 16``. Nothing else in
   the repo changes shape: BC and ACT read ``state_dim`` / ``action_dim`` off
   dataset metadata, so ROADMAP rule 2 ("bimanual is just a bigger action dim")
   is executed here rather than asserted.

2. **Three cameras.** One static plus one wrist per arm. The policies build one
   encoder per camera name in their config, so camera *count* is data too --
   another claim this task is the first to actually exercise.

3. **Failure has a bimanual bucket.** ``DESYNCHRONISED`` -- one arm holds its
   handle and the other does not. Neither ``NO_GRASP`` nor ``DROPPED`` describes
   it, and it is the exact failure this task exists to measure.

No MJCF authoring: unlike T1's bin, robosuite ships this scene. The adapter is
where the backend's names die, as always.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import robosuite as suite
import yaml
from robosuite.environments.manipulation.two_arm_lift import TwoArmLift
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_geom, new_joint

from dream_robot.sims.robosuite.tasks.two_arm_lift.tray import TrayWithHandlesObject

from dream_robot.core.env_api import FailureMode, StepResult
from dream_robot.core.schema import (
    BIMANUAL_PANDA,
    Embodiment,
    Observation,
    build_observation,
    validate_action,
)

TASK_YAML = Path(__file__).with_name("task.yaml")

#: End-effector sites the expert's IK drives, arm 0 then arm 1. robosuite/Panda
#: names; they stay on the sim side of the seam and no policy ever sees them.
EEF_SITES = ("gripper0_right_grip_site", "gripper1_right_grip_site")

#: Pot handle sites, arm 0's handle first. robosuite pairs gripper0 with
#: handle0 in its own grasp checks, and the `parallel` layout puts each handle
#: on its own arm's side of the table.
HANDLE_SITES = ("pot_handle0", "pot_handle1")

N_ARMS = 2


@dataclass(frozen=True)
class MarbleSpec:
    count: int
    radius: float
    mass: float
    keep_fraction: float

    @property
    def required_inside(self) -> int:
        """How many must still be aboard. Ceil, so 0.75 x 9 is 7, not 6."""
        return int(np.ceil(self.keep_fraction * self.count))

    @property
    def diag_inertia(self) -> float:
        """Solid sphere: I = 2/5 · m · r², the same about all three axes."""
        return 0.4 * self.mass * self.radius ** 2


@dataclass(frozen=True)
class TraySpec:
    spill_angle_deg: float
    lip_over_radius: float
    handle_height: float

    def lip_height(self, marble_radius: float) -> float:
        """Lip height, as a measured multiple of the marble radius.

        Not the static formula. tray.py derives L = r·(1 − cos θ) by balancing a
        ball sitting still against the lip, which gives 0.29 r for 45 deg -- and
        a tray built to it spills at 20 deg, because a ball that has rolled the
        length of a 12 cm tray arrives with momentum and hops a lip that would
        statically hold it. The calibration is in task.yaml with the curve it
        came from.

        What the derivation does give, and what still holds, is L < r: a ball
        cannot escape a wall taller than itself at any angle below 90 deg.
        """
        return float(marble_radius * self.lip_over_radius)


@dataclass(frozen=True)
class TaskConfig:
    prompt: str
    control_hz: float
    horizon_seconds: float
    camera_hw: tuple[int, int]
    camera_mapping: dict[str, str]
    render_camera: str
    render_hw: tuple[int, int]
    arrangement: str
    table_full_size: tuple[float, float, float]
    marbles: MarbleSpec
    tray: TraySpec
    lift_height: float
    tilt_threshold_deg: float
    dropped_lift_fraction: float

    @classmethod
    def load(cls, path: Path = TASK_YAML) -> TaskConfig:
        raw = yaml.safe_load(path.read_text())
        return cls(
            prompt=str(raw["prompt"]),
            control_hz=float(raw["control"]["hz"]),
            horizon_seconds=float(raw["control"]["horizon_seconds"]),
            camera_hw=(int(raw["cameras"]["height"]), int(raw["cameras"]["width"])),
            camera_mapping=dict(raw["cameras"]["mapping"]),
            render_camera=str(raw["render"]["camera"]),
            render_hw=(int(raw["render"]["height"]), int(raw["render"]["width"])),
            arrangement=str(raw["scene"]["arrangement"]),
            table_full_size=tuple(raw["scene"]["table_full_size"]),
            marbles=MarbleSpec(
                count=int(raw["scene"]["marbles"]["count"]),
                radius=float(raw["scene"]["marbles"]["radius"]),
                mass=float(raw["scene"]["marbles"]["mass"]),
                keep_fraction=float(raw["scene"]["marbles"]["keep_fraction"]),
            ),
            tray=TraySpec(
                spill_angle_deg=float(raw["scene"]["tray"]["spill_angle_deg"]),
                lip_over_radius=float(raw["scene"]["tray"]["lip_over_radius"]),
                handle_height=float(raw["scene"]["tray"]["handle_height"]),
            ),
            lift_height=float(raw["success"]["lift_height"]),
            tilt_threshold_deg=float(raw["failure"]["tilt_threshold_deg"]),
            dropped_lift_fraction=float(raw["failure"]["dropped_lift_fraction"]),
        )

    @property
    def horizon_steps(self) -> int:
        return int(round(self.horizon_seconds * self.control_hz))


class _TwoArmLiftWithMarbles(TwoArmLift):
    """robosuite's TwoArmLift plus loose marbles in the pot. MJCF only.

    Same extension point T1 used to add its bin: override ``_load_model`` to
    inject geometry, and leave the task interface alone.

    The marbles are free bodies, not part of the pot, so they have to be placed
    after the pot is placed -- the placement sampler only runs at reset, and the
    pot lands somewhere new each seed. Hence the ``_reset_internal`` override.
    """

    def __init__(self, marbles: MarbleSpec, tray: TraySpec, **kwargs):
        self._marbles = marbles        # must exist before _load_model runs
        self._tray = tray
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        # Swap the graspable object for the tray by rebinding the name
        # robosuite's own _load_model looks up, rather than forking that method.
        # Forking would mean copying robot placement, arena construction, the
        # placement sampler and task assembly into this repo, where they would
        # silently drift from the installed robosuite. The tray subclasses the
        # pot and is still named "pot", so every site, body and joint name
        # downstream is unchanged.
        import robosuite.environments.manipulation.two_arm_lift as _two_arm_lift

        original = _two_arm_lift.PotWithHandlesObject
        _two_arm_lift.PotWithHandlesObject = lambda name: TrayWithHandlesObject(
            name=name,
            lip_height=self._tray.lip_height(self._marbles.radius),
            handle_height=self._tray.handle_height,
        )
        try:
            super()._load_model()
        finally:
            _two_arm_lift.PotWithHandlesObject = original
        for i in range(self._marbles.count):
            body = new_body(name=f"marble_{i}", pos=array_to_string([0, 0, 0]))
            body.append(new_joint(name=f"marble_{i}_free", type="free"))
            # Explicit inertial, not geom mass/density. robosuite's compiled
            # model does not derive inertia from these geoms -- a marble
            # carrying only mass="..." builds a zero-mass body and MuJoCo
            # refuses the model outright ("mass and inertia of moving bodies
            # must be larger than mjMINVAL").
            inertia = array_to_string([self._marbles.diag_inertia] * 3)
            ET.SubElement(
                body, "inertial", pos="0 0 0",
                mass=str(self._marbles.mass), diaginertia=inertia,
            )
            body.append(
                new_geom(
                    name=f"marble_{i}", type="sphere",
                    size=array_to_string([self._marbles.radius]),
                    rgba=array_to_string(_MARBLE_RGBA[i % len(_MARBLE_RGBA)]),
                    # group=1: robosuite's cameras do not draw group 0, and a
                    # marble the policy cannot see is worse than no marble --
                    # the physics and the pixels would disagree (T1's bin bug).
                    group=1,
                    contype="1", conaffinity="1", condim="4",
                    friction=array_to_string([0.4, 0.005, 0.0001]),
                )
            )
            self.model.worldbody.append(body)

    def _setup_references(self) -> None:
        super()._setup_references()
        self.marble_body_ids = [
            self.sim.model.body_name2id(f"marble_{i}")
            for i in range(self._marbles.count)
        ]
        self.marble_qpos_addrs = [
            self.sim.model.get_joint_qpos_addr(f"marble_{i}_free")[0]
            for i in range(self._marbles.count)
        ]
        self.marble_qvel_addrs = [
            self.sim.model.get_joint_qvel_addr(f"marble_{i}_free")[0]
            for i in range(self._marbles.count)
        ]

    def _reset_internal(self) -> None:
        super()._reset_internal()
        # The placement sampler writes the pot's qpos, but site_xpos still holds
        # the PREVIOUS episode's pose until the kinematics are recomputed. Read
        # it before this forward() and every marble is placed where the pot used
        # to be -- which looks exactly like "the marbles fell out at reset".
        self.sim.forward()
        m = self._marbles
        centre = np.array(self.sim.data.site_xpos[self.pot_center_id], dtype=float)
        # Inner floor of the pot: half-height down from the centre, plus the
        # wall thickness, plus the marble's own radius so it rests rather than
        # starts interpenetrating the base.
        floor_z = centre[2] - self.pot.body_half_size[2] + self.pot.thickness + m.radius
        # A LATTICE, not uniform random positions. Marbles sampled independently
        # overlap each other constantly, and MuJoCo resolves an initial
        # interpenetration by firing them apart -- which throws one off an
        # untilted tray and makes every spill measurement read early.
        #
        # The lattice spans the TRAY, not an arbitrary spawn box: sized to a
        # smaller box it runs out of columns and stacks the surplus into layers,
        # and on a tray those tip off immediately.
        pitch = 2.1 * m.radius
        jitter = 0.05 * m.radius
        usable = 2 * (self.pot.inner_half_width - m.radius)
        per_row = max(1, int(usable // pitch) + 1)
        for slot, addr in enumerate(self.marble_qpos_addrs):
            col, row = slot % per_row, (slot // per_row) % per_row
            layer = slot // (per_row * per_row)
            local = (np.array([col, row]) - (per_row - 1) / 2.0) * pitch
            local = local + np.random.uniform(-jitter, jitter, size=2)
            self.sim.data.qpos[addr : addr + 3] = [
                centre[0] + local[0], centre[1] + local[1], floor_z + layer * pitch
            ]
            self.sim.data.qpos[addr + 3 : addr + 7] = [1.0, 0.0, 0.0, 0.0]
        for addr in self.marble_qvel_addrs:
            # Carrying the previous episode's velocities in would launch the
            # marbles out of the pot on the first step of the new one.
            self.sim.data.qvel[addr : addr + 6] = 0.0
        self.sim.forward()


#: Marbles are tinted so a spill reads at a glance in a 128 px frame.
_MARBLE_RGBA = (
    [0.95, 0.85, 0.15, 1.0],
    [0.15, 0.85, 0.85, 1.0],
    [0.95, 0.45, 0.15, 1.0],
    [0.75, 0.25, 0.85, 1.0],
)


def _joint_position_controller_config() -> dict:
    """Absolute joint targets for one Panda -- spec.md's action contract.

    Identical to T1's, and deliberately still a per-robot config: robosuite
    takes a list of one config per robot, and each robot here is a single-arm
    Panda whose body part is named "right". Two of these give a 16-dim action
    (7 joints + 1 gripper, twice) instead of the BASIC default's 14-dim OSC.
    """
    cfg = suite.load_composite_controller_config(controller="BASIC", robot="Panda")
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


class TwoArmLiftTask:
    """Implements ``core.env_api.Env``. Structurally, without inheriting it."""

    def __init__(self, config: TaskConfig | None = None):
        self._cfg = config or TaskConfig.load()
        ch, cw = self._cfg.camera_hw
        self._backend_cams = list(self._cfg.camera_mapping)
        self._env = _TwoArmLiftWithMarbles(
            self._cfg.marbles,
            self._cfg.tray,
            robots=["Panda", "Panda"],
            env_configuration=self._cfg.arrangement,
            controller_configs=[_joint_position_controller_config() for _ in range(N_ARMS)],
            table_full_size=self._cfg.table_full_size,
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
        self._gripper_spans = self._measure_gripper_spans()

    # --- Env protocol ------------------------------------------------------

    @property
    def embodiment(self) -> Embodiment:
        return BIMANUAL_PANDA

    @property
    def control_hz(self) -> float:
        return float(self._cfg.control_hz)

    @property
    def prompt(self) -> str:
        return self._cfg.prompt

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            # robosuite's placement samplers call np.random.uniform directly, so
            # the global RNG is the only seeding hook there is (sim README).
            np.random.seed(seed)
        self._raw = self._env.reset()
        self._t = 0
        self._max_pot_lift = self._pot_lift()
        self._ever_grasped = [False, False]
        return self._observe()

    def step(self, action) -> StepResult:
        act = validate_action(action, BIMANUAL_PANDA)
        arms = act[BIMANUAL_PANDA.arm_slice].reshape(N_ARMS, -1)
        openings = act[BIMANUAL_PANDA.gripper_slice]
        # Contract: 0 closed .. 1 open. robosuite GRIP: -1 open .. +1 close.
        # robosuite concatenates per-robot actions, each [7 joints | 1 gripper].
        command = np.concatenate([
            np.concatenate([arms[i], [1.0 - 2.0 * float(openings[i])]])
            for i in range(N_ARMS)
        ])
        self._raw, _, _, _ = self._env.step(command)
        self._t += 1

        self._max_pot_lift = max(self._max_pot_lift, self._pot_lift())
        for i, held in enumerate(self.handles_held):
            self._ever_grasped[i] = self._ever_grasped[i] or held

        success = (
            self._pot_lift() > self._cfg.lift_height
            and self.marbles_inside >= self._cfg.marbles.required_inside
        )
        truncated = (not success) and self._t >= self._cfg.horizon_steps
        return StepResult(
            observation=self._observe(),
            success=success,
            terminated=success,
            truncated=truncated,
            failure_mode=self._diagnose() if truncated else FailureMode.NONE,
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
    # spec.md permits the expert to read ground truth; _observe() never touches
    # any of this. Everything here is per-arm and ordered arm 0 then arm 1, the
    # same order as the action vector.

    @property
    def handle_pos(self) -> tuple[np.ndarray, np.ndarray]:
        """Each arm's own handle centre, in world coordinates."""
        sim = self._env.sim
        return tuple(
            np.array(sim.data.site_xpos[sim.model.site_name2id(name)], dtype=float)
            for name in HANDLE_SITES
        )

    @property
    def handles_held(self) -> tuple[bool, bool]:
        """Whether each gripper currently has a real grasp on its own handle."""
        pot = self._env.pot
        geoms = (pot.handle0_geoms, pot.handle1_geoms)
        return tuple(
            bool(self._env._check_grasp(
                gripper=self._env.robots[i].gripper["right"], object_geoms=geoms[i]
            ))
            for i in range(N_ARMS)
        )

    @property
    def marbles_inside(self) -> int:
        """How many marbles are still aboard, measured in the TRAY's frame.

        Note the z bound is generous, not the lip. A marble on a tray stands
        proud of its lip by design -- the lip is only 0.29 r tall -- so bounding
        contents below the rim, as a deep pot would let you, counts a full tray
        as empty. What actually distinguishes aboard from spilled is the
        footprint: a marble that leaves goes over the edge and outward.

        World-frame bounds would not do either: they count a marble lying on the
        table beside an upended tray, which is the case this number exists for.
        """
        sim = self._env.sim
        tray_pos = np.array(sim.data.body_xpos[self._env.pot_body_id], dtype=float)
        tray_rot = np.array(sim.data.body_xmat[self._env.pot_body_id]).reshape(3, 3)
        inner = float(self._env.pot.inner_half_width)
        floor = -float(self._env.pot.body_half_size[2]) + float(self._env.pot.thickness)
        radius = self._cfg.marbles.radius
        count = 0
        for body_id in self._env.marble_body_ids:
            world = np.array(sim.data.body_xpos[body_id], dtype=float)
            local = tray_rot.T @ (world - tray_pos)
            on_footprint = np.all(np.abs(local[:2]) < inner)
            resting = floor - radius < local[2] < floor + 6 * radius
            if on_footprint and resting:
                count += 1
        return count

    @property
    def pot_center(self) -> np.ndarray:
        sim = self._env.sim
        return np.array(sim.data.site_xpos[self._env.pot_center_id], dtype=float)

    @property
    def mujoco(self):
        """(model, data). Re-read after every reset -- the handles go stale."""
        return self._env.sim.model._model, self._env.sim.data._data

    @property
    def arm_qpos_idx(self) -> tuple[np.ndarray, np.ndarray]:
        return tuple(
            np.asarray(self._env.robots[i]._ref_joint_pos_indexes) for i in range(N_ARMS)
        )

    # --- adapter internals -------------------------------------------------

    def _measure_gripper_spans(self) -> list[float]:
        """Max finger separation per gripper, read from the model not hardcoded."""
        m = self._env.sim.model._model
        spans = []
        for i in range(N_ARMS):
            names = self._env.robots[i].gripper["right"].joints
            lo_hi = np.array([
                m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                for n in names
            ])
            span = float(lo_hi[0][1] - lo_hi[1][0])
            if span <= 0:
                raise ValueError(f"non-positive gripper span {span} from joints {names}")
            spans.append(span)
        return spans

    def _observe(self) -> Observation:
        raw = self._raw
        joints = np.concatenate([raw[f"robot{i}_joint_pos"] for i in range(N_ARMS)])
        openings = []
        for i in range(N_ARMS):
            qpos = raw[f"robot{i}_gripper_qpos"]
            width = float(qpos[0] - qpos[1])
            openings.append(np.clip(width / self._gripper_spans[i], 0.0, 1.0))
        return build_observation(
            joint_positions=joints,
            gripper_openings=openings,
            images={
                # robosuite renders bottom-up (OpenGL row order).
                canonical: np.ascontiguousarray(raw[f"{backend}_image"][::-1])
                for backend, canonical in self._cfg.camera_mapping.items()
            },
            embodiment=BIMANUAL_PANDA,
        )

    def _pot_lift(self) -> float:
        """Height of the pot's bottom above the table top, in metres.

        The same quantity robosuite's own ``_check_success`` computes, written
        out here so the threshold it is compared against lives in task.yaml.
        """
        sim = self._env.sim
        # bottom_offset, not top_offset: robosuite's own _check_success uses the
        # latter, which is only the same number for an object symmetric about
        # its centre. The tray is not -- its handles stand well above its base.
        pot_bottom = float(
            sim.data.site_xpos[self._env.pot_center_id][2]
            + self._env.pot.bottom_offset[2]
        )
        table_top = float(sim.data.site_xpos[self._env.table_top_id][2])
        return pot_bottom - table_top

    def _pot_tilt_deg(self) -> float:
        """Angle between the pot's own z axis and world z."""
        sim = self._env.sim
        rot = np.array(sim.data.body_xmat[self._env.pot_body_id]).reshape(3, 3)
        cos = float(np.clip(rot[:, 2] @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    def _diagnose(self) -> FailureMode:
        """Why this episode failed, in the closed vocabulary of FailureMode.

        Order matters: the desynchronisation check comes before the tilt check,
        because one arm lifting alone *causes* the tilt, and reporting the
        symptom would hide the failure axis the task exists to measure.
        """
        grasped = self._ever_grasped
        if not any(grasped):
            return FailureMode.NO_GRASP
        if not all(grasped):
            return FailureMode.DESYNCHRONISED
        if self.marbles_inside < self._cfg.marbles.required_inside:
            return FailureMode.KNOCKED_OVER
        if self._pot_tilt_deg() > self._cfg.tilt_threshold_deg:
            return FailureMode.KNOCKED_OVER
        if not all(self.handles_held) and (
            self._max_pot_lift > self._cfg.dropped_lift_fraction * self._cfg.lift_height
        ):
            return FailureMode.DROPPED
        return FailureMode.TIMEOUT
