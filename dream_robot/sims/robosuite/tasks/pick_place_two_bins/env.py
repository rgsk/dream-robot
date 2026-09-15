"""pick_place_two_bins on robosuite -- ROADMAP's T2, the multimodality test.

T1's scene plus a second bin mirrored across the cube. Both bins are valid, so
success is "the cube is resting in either one".

**This task subclasses T1's; the pick_place_cube docstring says later tasks copy
its shape rather than extend it.** T2 is the exception on purpose. It is only a
multimodality test if everything except the second mode is T1: the same
controller, cameras, image flip, gripper normalisation, and observation path.
Inheriting those makes "identical to T1" true by construction instead of by two
copies that have to be kept in step. What T2 overrides is exactly what differs:
the scene (two bins), the success predicate, and the failure diagnosis.

**wrong_target means "ended between the bins".** That is the failure behaviour
cloning is predicted to have: averaging "carry left" and "carry right" gives
"carry nowhere", and the cube comes back down near where it started, on the
strip of table between the two bins. ``between_bins`` is that strip, and it is
public so experiment scripts can use the same definition as the histogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from robosuite.environments.manipulation.lift import Lift
from robosuite.utils.mjcf_utils import array_to_string, new_body, new_geom
from robosuite.utils.placement_samplers import UniformRandomSampler

from dream_robot.core.env_api import FailureMode
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import (
    PickPlaceCube,
    _joint_position_controller_config,
)

TASK_YAML = Path(__file__).with_name("task.yaml")


@dataclass(frozen=True)
class NamedBin:
    name: str
    offset_xy: tuple[float, float]
    rgba: tuple[float, float, float, float]
    inner_half: float
    wall_half_thickness: float
    wall_half_height: float

    @property
    def outer_half(self) -> float:
        return self.inner_half + 2 * self.wall_half_thickness


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
    bins: tuple[NamedBin, ...]
    resting_z_margin: float
    lift_threshold: float
    fixed_cube_pose: bool = False

    @classmethod
    def load(cls, path: Path = TASK_YAML) -> TaskConfig:
        raw = yaml.safe_load(path.read_text())
        shared = raw["scene"]["bin"]
        bins = tuple(
            NamedBin(
                name=str(b["name"]),
                offset_xy=tuple(b["offset_xy"]),
                rgba=tuple(b["rgba"]),
                inner_half=float(shared["inner_half"]),
                wall_half_thickness=float(shared["wall_half_thickness"]),
                wall_half_height=float(shared["wall_half_height"]),
            )
            for b in raw["scene"]["bins"]
        )
        if len(bins) != 2:
            raise ValueError(f"T2 has exactly two bins, task.yaml lists {len(bins)}")
        return cls(
            prompt=str(raw["prompt"]),
            control_hz=float(raw["control"]["hz"]),
            horizon_seconds=float(raw["control"]["horizon_seconds"]),
            camera_hw=(int(raw["cameras"]["height"]), int(raw["cameras"]["width"])),
            camera_mapping=dict(raw["cameras"]["mapping"]),
            render_camera=str(raw["render"]["camera"]),
            render_hw=(int(raw["render"]["height"]), int(raw["render"]["width"])),
            cube_xy_range=tuple(raw["scene"]["cube_xy_range"]),
            bins=bins,
            resting_z_margin=float(raw["success"]["resting_z_margin"]),
            lift_threshold=float(raw["failure"]["lift_threshold"]),
            fixed_cube_pose=bool(raw["scene"].get("fixed_cube_pose", False)),
        )

    @property
    def horizon_steps(self) -> int:
        return int(round(self.horizon_seconds * self.control_hz))


class _LiftWithBins(Lift):
    """robosuite's Lift scene plus N static open-top bins. MJCF only."""

    def __init__(self, bins: tuple[NamedBin, ...], **kwargs):
        self._bins = bins             # must exist before _load_model runs
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        super()._load_model()
        for b in self._bins:
            cx, cy, _ = self._center(b)
            body = new_body(
                name=f"bin_{b.name}",
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
                        name=f"bin_{b.name}_{suffix}", type="box",
                        size=array_to_string(size), pos=array_to_string(pos),
                        rgba=array_to_string(b.rgba),
                        group=1,      # drawn by the cameras; see T1's env.py
                        friction=array_to_string([1.0, 0.005, 0.0001]),
                    )
                )
            self.model.worldbody.append(body)

    def _center(self, b: NamedBin) -> np.ndarray:
        return np.array([
            self.table_offset[0] + b.offset_xy[0],
            self.table_offset[1] + b.offset_xy[1],
            self.table_offset[2],
        ])

    def bin_centers(self) -> tuple[np.ndarray, ...]:
        return tuple(self._center(b) for b in self._bins)


class PickPlaceTwoBins(PickPlaceCube):
    """Implements ``core.env_api.Env``. T1 with two valid goals."""

    def __init__(self, config: TaskConfig | None = None):
        # Not super().__init__: T1's builds a one-bin scene. Everything below
        # except the scene class mirrors it field for field.
        self._cfg = config or TaskConfig.load()
        ch, cw = self._cfg.camera_hw
        self._backend_cams = list(self._cfg.camera_mapping)
        placement = None                          # None: Lift's own random sampler
        if self._cfg.fixed_cube_pose:
            # Lift's default sampler with both ranges and the yaw pinned to zero.
            # reference_pos is Lift's hardcoded table_offset.
            placement = UniformRandomSampler(
                name="ObjectSampler",
                x_range=[0.0, 0.0],
                y_range=[0.0, 0.0],
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=np.array((0.0, 0.0, 0.8)),
                z_offset=0.01,
            )
        self._env = _LiftWithBins(
            self._cfg.bins,
            placement_initializer=placement,
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

    # --- privileged access, for scripted experts and diagnostics only ------

    @property
    def bin_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self._cfg.bins)

    @property
    def bin_centers(self) -> tuple[np.ndarray, ...]:
        return self._env.bin_centers()

    @property
    def bin_center(self) -> np.ndarray:
        # T1's expert reads this. With two bins "the" bin is the expert's coin
        # flip, not a property of the scene -- fail loudly rather than pick one.
        raise AttributeError("two bins: read bin_centers; the target is the expert's choice")

    def which_bin(self, cube: np.ndarray) -> int | None:
        """Index of the bin the cube is resting in, or None."""
        for i, (b, c) in enumerate(zip(self._cfg.bins, self.bin_centers)):
            inside = bool(np.all(np.abs(cube[:2] - c[:2]) < b.inner_half))
            resting = bool(cube[2] < c[2] + self._cfg.resting_z_margin)
            if inside and resting:
                return i
        return None

    def between_bins(self, cube: np.ndarray) -> bool:
        """Is the cube over the strip of table between the two bins?

        XY only, so a cube still held in the air above that strip at timeout
        counts too -- hovering undecided between the bins is the same failure
        as setting the cube down there. Diagnostics separate the two by z.
        """
        (b0, b1), (c0, c1) = self._cfg.bins, self.bin_centers
        axis = c1[:2] - c0[:2]
        length = float(np.linalg.norm(axis))
        u = axis / length
        rel = np.asarray(cube[:2], dtype=float) - c0[:2]
        along = float(rel @ u)
        across = abs(float(rel[0] * u[1] - rel[1] * u[0]))
        return (
            b0.outer_half < along < length - b1.outer_half
            and across < min(b0.outer_half, b1.outer_half)
        )

    # --- overrides of T1's predicates --------------------------------------

    def _cube_in_bin(self, cube: np.ndarray) -> bool:
        return self.which_bin(cube) is not None

    def _diagnose(self, cube: np.ndarray) -> FailureMode:
        lifted = self._max_cube_z > self._cube_z0 + self._cfg.lift_threshold
        if not lifted:
            return FailureMode.NO_GRASP
        if self.between_bins(cube):
            return FailureMode.WRONG_TARGET
        return FailureMode.DROPPED
