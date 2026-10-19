"""assorted_pick_place on robosuite -- ROADMAP's T7, assorted objects on T5's table.

T5 asked "which of two identical cubes?". T7 asks the executor to cope with
*what* it is picking: 2-3 objects per episode, each a random kind (box, bar,
cylinder, ball, or one of robosuite's meshes: can, bottle, milk, cereal,
bread), size, colour and yaw. The instruction names one object by how it
looks -- "put the yellow bar in the left bin" -- and only that object resting
in that bin counts.

**The scene is redrawn on every reset, MJCF and all.** T5 built two cubes once
and repainted them; here the object *set* changes, so the draw happens inside
``_load_model``, which robosuite calls again on every (hard) reset. Kinds,
sizes, colours and the placement are drawn together, from the global RNG,
before anything else -- so an episode is a pure function of its seed, as in T5.

**Sizes are measured, not read from robosuite.** Its meshes' metadata is off:
the listed bottom offsets put a can 2 cm in the air and the bottle's footprint
is 4.0 cm, not the 2.5 cm listed. So each object is compiled alone once, at
draw time, and its footprint and resting height are read from the compiled
geoms (``_measure``).

**Placement is ours, not robosuite's sampler.** Its sampler only refuses
overlapping footprints; a gripper needs clear table *between* objects (see
``min_gap`` in task.yaml). So each object is placed by rejection against the
ones already down, and a scene that cannot be placed is redrawn whole.

**Subclasses T2 for the bins, the controller and the observation path**, as T5
does. What differs from T5 is exactly the scene, the instruction and which
body the success predicate reads.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import robosuite.models.objects as rs_objects
import yaml
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject, MujocoObject
from robosuite.models.tasks import ManipulationTask
from robosuite.models.world import MujocoWorldBase

from dream_robot.core.env_api import FailureMode, StepResult
from dream_robot.core.schema import PANDA, validate_action
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import (
    _joint_position_controller_config,
)
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import (
    NamedBin,
    PickPlaceTwoBins,
    _LiftWithBins,
)

TASK_YAML = Path(__file__).with_name("task.yaml")

Range = tuple[float, float]


@dataclass(frozen=True)
class TaskConfig:
    control_hz: float
    horizon_seconds: float
    camera_hw: tuple[int, int]
    camera_mapping: dict[str, str]
    render_camera: str
    render_hw: tuple[int, int]
    count: tuple[int, int]
    region_x: Range
    region_y: Range
    min_gap: float
    primitives: dict[str, dict[str, Range]]
    meshes: dict[str, dict[str, str]]
    primitive_density: float
    palette: dict[str, tuple[float, float, float, float]]
    bins: tuple[NamedBin, ...]
    template: str
    resting_z_margin: float
    lift_threshold: float

    @classmethod
    def load(cls, path: Path = TASK_YAML) -> TaskConfig:
        raw = yaml.safe_load(path.read_text())
        scene, shared = raw["scene"], raw["scene"]["bin"]
        bins = tuple(
            NamedBin(
                name=str(b["name"]),
                offset_xy=tuple(b["offset_xy"]),
                rgba=tuple(b["rgba"]),
                inner_half=float(shared["inner_half"]),
                wall_half_thickness=float(shared["wall_half_thickness"]),
                wall_half_height=float(shared["wall_half_height"]),
            )
            for b in scene["bins"]
        )
        cfg = cls(
            control_hz=float(raw["control"]["hz"]),
            horizon_seconds=float(raw["control"]["horizon_seconds"]),
            camera_hw=(int(raw["cameras"]["height"]), int(raw["cameras"]["width"])),
            camera_mapping=dict(raw["cameras"]["mapping"]),
            render_camera=str(raw["render"]["camera"]),
            render_hw=(int(raw["render"]["height"]), int(raw["render"]["width"])),
            count=tuple(scene["count"]),
            region_x=tuple(scene["region_x"]),
            region_y=tuple(scene["region_y"]),
            min_gap=float(scene["min_gap"]),
            primitives={
                k: {dim: tuple(r) for dim, r in v.items()}
                for k, v in scene["primitives"].items()
            },
            meshes={k: dict(v) for k, v in scene["meshes"].items()},
            primitive_density=float(scene["primitive_density"]),
            palette={k: tuple(v) for k, v in scene["palette"].items()},
            bins=bins,
            template=str(raw["instruction"]["template"]),
            resting_z_margin=float(raw["success"]["resting_z_margin"]),
            lift_threshold=float(raw["failure"]["lift_threshold"]),
        )
        cfg._check()
        return cfg

    def _check(self) -> None:
        if len(self.bins) != 2:
            raise ValueError("T7 has exactly two bins")
        lo, hi = self.count
        if not 1 <= lo <= hi:
            raise ValueError(f"bad object count range {self.count}")
        if set(self.primitives) & set(self.meshes):
            raise ValueError("a kind is both a primitive and a mesh")
        for kind, spec in self.meshes.items():
            if not hasattr(rs_objects, spec["cls"]):
                raise ValueError(f"mesh {kind}: robosuite has no {spec['cls']}")

    @property
    def horizon_steps(self) -> int:
        return int(round(self.horizon_seconds * self.control_hz))

    @property
    def kinds(self) -> tuple[str, ...]:
        return (*self.primitives, *self.meshes)


@dataclass(frozen=True)
class SceneObject:
    """One drawn object: what it is, how big, where. Ground truth, never observed."""

    kind: str                  # "box", "bar", ..., "can", "milk", ...
    name: str                  # how the instruction says it: "red ball", "milk carton"
    colour: str | None         # palette name; None for a textured mesh
    size: dict[str, float]     # the drawn dimensions (primitives), metres; {} for meshes
    footprint: float           # horizontal radius, metres: the circle it can occupy at any yaw
    rest_height: float         # body origin above the table when upright on it
    xy: tuple[float, float]    # spawn centre, relative to the table centre
    yaw: float                 # spawn yaw, radians

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "colour": self.colour,
            "size": {k: round(v, 4) for k, v in self.size.items()},
            "xy": [round(c, 4) for c in self.xy],
            "yaw_deg": round(float(np.rad2deg(self.yaw)), 1),
        }


@dataclass(frozen=True)
class Instruction:
    target: int                # index into the episode's objects
    side: str
    prompt: str


class _AssortedLift(Lift):
    """Lift's table with this episode's drawn objects instead of its cube. MJCF only.

    ``objects`` and ``models`` are redrawn by every ``_load_model``, i.e. every
    hard reset. Lift's references and observables read ``self.cube``; it is set
    to the first object so they keep working, and nothing of ours reads them.
    """

    MAX_DRAWS = 200

    def __init__(self, cfg: TaskConfig, **kwargs):
        self._t7 = cfg                # must exist before _load_model runs
        self.objects: tuple[SceneObject, ...] = ()
        self.models: tuple[MujocoObject, ...] = ()
        super().__init__(**kwargs)

    def _load_model(self) -> None:
        ManipulationEnv._load_model(self)             # skip Lift's; it builds its cube
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)
        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self.objects, self.models = self._draw_scene()
        self.cube = self.models[0]
        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.models),
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self.object_body_ids = tuple(
            self.sim.model.body_name2id(m.root_body) for m in self.models
        )

    def _reset_internal(self) -> None:
        ManipulationEnv._reset_internal(self)         # skip Lift's sampler
        top = self.table_offset[2]
        for obj, model in zip(self.objects, self.models, strict=True):
            x, y = obj.xy
            pos = [self.table_offset[0] + x, self.table_offset[1] + y, top + obj.rest_height + 0.002]
            quat = [np.cos(obj.yaw / 2), 0.0, 0.0, np.sin(obj.yaw / 2)]   # wxyz, about z
            self.sim.data.set_joint_qpos(model.joints[0], np.concatenate([pos, quat]))

    # --- the draw ----------------------------------------------------------

    def _draw_scene(self) -> tuple[tuple[SceneObject, ...], tuple[MujocoObject, ...]]:
        cfg = self._t7
        for _ in range(self.MAX_DRAWS):
            n = int(np.random.randint(cfg.count[0], cfg.count[1] + 1))
            drawn: list[tuple[SceneObject, MujocoObject]] = []
            for i in range(n):
                spec = self._draw_object(i, taken={o.name for o, _ in drawn})
                if spec is None:
                    break
                drawn.append(spec)
            else:
                return tuple(o for o, _ in drawn), tuple(m for _, m in drawn)
        raise RuntimeError(f"no placeable scene in {self.MAX_DRAWS} draws: loosen region/min_gap")

    def _draw_object(self, i: int, taken: set[str]):
        """Kind, then size and colour, then a spot clear of ``self._placed``.

        Returns (SceneObject, model), or None when this object finds no spot --
        the caller then redraws the whole scene, so big objects are not quietly
        favoured towards being first.
        """
        cfg = self._t7
        if i == 0:
            self._placed: list[tuple[float, float, float]] = []    # (x, y, footprint)
        for _ in range(20):                                         # a name not yet in the scene
            kind = cfg.kinds[np.random.randint(len(cfg.kinds))]
            model, name, colour, size = self._build(kind, f"obj{i}")
            if name not in taken:
                break
        else:
            return None

        r, rest_height = _measure(model)
        (x0, x1), (y0, y1) = cfg.region_x, cfg.region_y
        if x1 - x0 < 2 * r or y1 - y0 < 2 * r:
            return None
        for _ in range(100):
            x = float(np.random.uniform(x0 + r, x1 - r))
            y = float(np.random.uniform(y0 + r, y1 - r))
            if all(np.hypot(x - px, y - py) >= r + pr + cfg.min_gap for px, py, pr in self._placed):
                break
        else:
            return None
        self._placed.append((x, y, r))
        yaw = float(np.random.uniform(-np.pi, np.pi))
        obj = SceneObject(
            kind=kind, name=name, colour=colour, size=size, footprint=r,
            rest_height=rest_height, xy=(x, y), yaw=yaw,
        )
        return obj, model

    def _build(self, kind: str, mj_name: str):
        """(model, instruction name, colour, drawn size) for one kind."""
        cfg = self._t7
        if kind in cfg.meshes:
            spec = cfg.meshes[kind]
            return getattr(rs_objects, spec["cls"])(name=mj_name), spec["name"], None, {}

        size = {dim: float(np.random.uniform(*r)) for dim, r in cfg.primitives[kind].items()}
        colour = list(cfg.palette)[np.random.randint(len(cfg.palette))]
        common = dict(name=mj_name, rgba=list(cfg.palette[colour]), density=cfg.primitive_density)
        if kind in ("box", "bar"):
            shape = [size["half_x"], size["half_y"], size["half_z"]]
            model = BoxObject(size=shape, **common)
        elif kind == "cylinder":
            model = CylinderObject(size=[size["radius"], size["half_height"]], **common)
        elif kind == "ball":
            model = BallObject(size=[size["radius"]], **common)
        else:
            raise ValueError(f"no builder for primitive {kind!r}")
        return model, f"{colour} {kind}", colour, size


def _measure(model: MujocoObject) -> tuple[float, float]:
    """(footprint radius, resting height) of an object, from its compiled geoms.

    Compiles the object alone at the origin, upright, yaw 0. The footprint is the
    farthest xy reach of any geom's bounding box from the body origin -- visual
    geoms included, since a finger must clear those too -- so it holds at any yaw.
    The resting height is how far the lowest *colliding* geom reaches below the
    origin: that is what sits on the table.
    """
    world = MujocoWorldBase()
    world.merge_assets(model)
    world.worldbody.append(model.get_obj())
    m = world.get_model(mode="mujoco")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    origin = d.xpos[m.body(model.root_body).id]
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    reach, lowest = 0.0, np.inf
    for g in range(m.ngeom):
        centre, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        corners = (signs * half + centre) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] - origin
        reach = max(reach, float(np.linalg.norm(corners[:, :2], axis=1).max()))
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            lowest = min(lowest, float(corners[:, 2].min()))
    return reach, -lowest


class _Scene(_LiftWithBins, _AssortedLift):
    """T2's bins on top of the assorted-object Lift (same MRO trick as T5)."""

    def __init__(self, cfg: TaskConfig, **kwargs):
        super().__init__(cfg.bins, cfg=cfg, **kwargs)


class AssortedPickPlace(PickPlaceTwoBins):
    """Implements ``core.env_api.Env``. T5's table with assorted, named objects."""

    def __init__(self, config: TaskConfig | None = None, *, horizon_seconds: float | None = None):
        """``horizon_seconds`` overrides the task's episode clock -- e.g. longer for a
        person or LLM driving by hand, who explores and retries. Evals keep the task's."""
        # Not super().__init__: T2's builds a one-cube scene. Mirrors T5's.
        self._cfg = config or TaskConfig.load()
        if horizon_seconds is not None:
            self._cfg = replace(self._cfg, horizon_seconds=float(horizon_seconds))
        ch, cw = self._cfg.camera_hw
        self._backend_cams = list(self._cfg.camera_mapping)
        self._env = _Scene(
            self._cfg,
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
        self._instruction: Instruction | None = None
        self._z0 = np.zeros(0)
        self._max_z = np.zeros(0)
        self._gripper_span = self._measure_gripper_span()

    # --- Env protocol ------------------------------------------------------

    @property
    def prompt(self) -> str:
        return self.instruction.prompt

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)                      # see T1's env.py
        self._env.reset()                             # scene + placement draw first
        self._instruction = self._draw_instruction()
        self._raw = self._env._get_observations(force_update=True)
        self._t = 0
        self._z0 = self.object_positions[:, 2].copy()
        self._max_z = self._z0.copy()
        return self._observe()

    def step(self, action) -> StepResult:
        act = validate_action(action, PANDA)
        arm = act[PANDA.arm_slice]
        opening = float(act[PANDA.gripper_slice][0])
        self._raw, _, _, _ = self._env.step(np.concatenate([arm, [1.0 - 2.0 * opening]]))
        self._t += 1
        self._max_z = np.maximum(self._max_z, self.object_positions[:, 2])

        success = self.bin_of(self.instruction.target) == self.target_bin
        truncated = (not success) and self._t >= self._cfg.horizon_steps
        return StepResult(
            observation=self._observe(),
            success=success,
            terminated=success,
            truncated=truncated,
            failure_mode=self._diagnose() if truncated else FailureMode.NONE,
        )

    # --- privileged access, for scripted planners and diagnostics only -----

    @property
    def instruction(self) -> Instruction:
        if self._instruction is None:
            raise RuntimeError("no instruction before the first reset()")
        return self._instruction

    @property
    def objects(self) -> tuple[SceneObject, ...]:
        """This episode's objects as drawn. Spawn pose, not the current one."""
        return self._env.objects

    @property
    def object_positions(self) -> np.ndarray:
        """(N, 3) current body origins, world frame."""
        xpos = self._env.sim.data.body_xpos
        return np.array([xpos[i] for i in self._env.object_body_ids], dtype=float)

    @property
    def target(self) -> SceneObject:
        return self.objects[self.instruction.target]

    @property
    def target_pos(self) -> np.ndarray:
        return self.object_positions[self.instruction.target].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        # T1's expert reads this. There is no cube; fail loudly rather than
        # hand it object 0's origin.
        raise AttributeError("T7 has no cube: read target_pos")

    @property
    def target_bin(self) -> int:
        return self.bin_names.index(self.instruction.side)

    def bin_of(self, i: int) -> int | None:
        """Index of the bin object ``i`` rests in, or None."""
        p = self.object_positions[i]
        ceiling = self.objects[i].rest_height + self._cfg.resting_z_margin
        for k, (b, c) in enumerate(zip(self._cfg.bins, self.bin_centers)):
            inside = bool(np.all(np.abs(p[:2] - c[:2]) < b.inner_half))
            if inside and p[2] < c[2] + ceiling:
                return k
        return None

    def lifted(self, i: int) -> bool:
        return bool(self._max_z[i] > self._z0[i] + self._cfg.lift_threshold)

    # --- internals ---------------------------------------------------------

    def _draw_instruction(self) -> Instruction:
        """Target, then side. Global RNG, after the scene, as T5."""
        target = int(np.random.randint(len(self.objects)))
        side = self.bin_names[np.random.randint(len(self.bin_names))]
        name = self.objects[target].name
        return Instruction(target, side, self._cfg.template.format(object=name, side=side))

    def _diagnose(self) -> FailureMode:
        t = self.instruction.target
        others_lifted = any(self.lifted(i) for i in range(len(self.objects)) if i != t)
        if others_lifted and not self.lifted(t):
            return FailureMode.WRONG_TARGET
        landed = self.bin_of(t)
        if landed is not None and landed != self.target_bin:
            return FailureMode.WRONG_TARGET
        if not self.lifted(t):
            return FailureMode.NO_GRASP
        return FailureMode.DROPPED
