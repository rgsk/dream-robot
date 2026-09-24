"""instructed_pick_place on robosuite -- ROADMAP's T5, the instruction test.

T2's two bins plus a second cube. Each episode draws an instruction, "put the
{color} cube in the {side} box", and only the named cube resting in the named
bin counts. Nothing in the scene says which of the four (cube, bin) outcomes
is wanted; only the words do.

**Subclasses T2, as T2 subclasses T1.** The controller, cameras, image flip,
gripper normalisation and observation path are T1's by construction, and the
bins are T2's. What T5 overrides is exactly what differs: the scene (a second
cube, recoloured per episode), what reset draws (an instruction), and the
success and failure predicates (the *named* cube, the *named* bin).

**The instruction is drawn by the env, not the expert.** In T2 the expert's
coin was the only source of the target, and nothing downstream needed it. Here
a policy must be told the target, so it has to exist before the first action,
and the env is the one thing both the expert and the policy talk to. It is
read through ``prompt``, the same property every task already has; T5 is only
the first task whose prompt changes between episodes.

**Colours are painted after reset, not built into the MJCF.** robosuite
rebuilds the scene on every reset, but setting ``geom_rgba`` on the built model
does not depend on that, and it keeps the draw order simple: placement first
(robosuite's sampler consumes the global RNG), then the instruction, then paint,
then re-render so the first observation already shows the new colours.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from robosuite.environments.manipulation.lift import Lift
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler

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

Pairs = Literal["seen", "held_out"]
Wording = Literal["template", "paraphrase"]


@dataclass(frozen=True)
class TaskConfig:
    control_hz: float
    horizon_seconds: float
    camera_hw: tuple[int, int]
    camera_mapping: dict[str, str]
    render_camera: str
    render_hw: tuple[int, int]
    cube_slots_xy: tuple[tuple[float, float], ...]
    cube_jitter_xy: tuple[float, float]
    cube_half_size: tuple[float, float]
    palette: dict[str, tuple[float, float, float, float]]
    bins: tuple[NamedBin, ...]
    template: str
    paraphrases: tuple[str, ...]
    held_out: tuple[tuple[str, str], ...]
    resting_z_margin: float
    lift_threshold: float

    @classmethod
    def load(cls, path: Path = TASK_YAML) -> TaskConfig:
        raw = yaml.safe_load(path.read_text())
        scene, shared, ins = raw["scene"], raw["scene"]["bin"], raw["instruction"]
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
            cube_slots_xy=tuple(tuple(s) for s in scene["cube_slots_xy"]),
            cube_jitter_xy=tuple(scene["cube_jitter_xy"]),
            cube_half_size=tuple(scene["cube_half_size"]),
            palette={k: tuple(v) for k, v in scene["palette"].items()},
            bins=bins,
            template=str(ins["template"]),
            paraphrases=tuple(ins["paraphrases"]),
            held_out=tuple(tuple(p) for p in ins["held_out"]),
            resting_z_margin=float(raw["success"]["resting_z_margin"]),
            lift_threshold=float(raw["failure"]["lift_threshold"]),
        )
        cfg._check()
        return cfg

    def _check(self) -> None:
        if len(self.bins) != 2 or len(self.cube_slots_xy) != 2:
            raise ValueError("T5 has exactly two bins and two cube slots")
        sides = {b.name for b in self.bins}
        for color, side in self.held_out:
            if color not in self.palette or side not in sides:
                raise ValueError(f"held_out pair {(color, side)} names no colour/bin")
        if len(self.palette) < 2:
            raise ValueError("the palette needs two colours: a target and a distractor")

    @property
    def horizon_steps(self) -> int:
        return int(round(self.horizon_seconds * self.control_hz))

    def pairs(self, which: Pairs) -> tuple[tuple[str, str], ...]:
        """(colour, side) targets drawn under ``which``, in a fixed order."""
        every = [(c, b.name) for c in self.palette for b in self.bins]
        held = set(self.held_out)
        return tuple(p for p in every if (p in held) == (which == "held_out"))


@dataclass(frozen=True)
class Instruction:
    """What one episode asks for, and the scene facts that make it true."""

    color: str
    side: str
    distractor_color: str
    target_slot: int          # which of the two cube bodies is the named one
    prompt: str


class _TwoCubeLift(Lift):
    """Lift's scene with two plain cubes in fixed slots instead of one. MJCF only.

    Replaces Lift's ``_load_model`` rather than extending it: Lift's builds a
    red-wood cube with its own sampler, and neither survives. The rest of Lift
    (references, observables, reward) only reads ``self.cube``, which is still
    here as slot 0.
    """

    def __init__(self, cfg: TaskConfig, **kwargs):
        self._t5 = cfg                # must exist before _load_model runs
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

        lo, hi = self._t5.cube_half_size
        # White and untextured: reset paints each episode's colours over it.
        self.cubes = tuple(
            BoxObject(name=name, size_min=[lo] * 3, size_max=[hi] * 3, rgba=[1, 1, 1, 1])
            for name in ("cube", "cube_b")
        )
        self.cube = self.cubes[0]

        jx, jy = self._t5.cube_jitter_xy
        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        for cube, (sx, sy) in zip(self.cubes, self._t5.cube_slots_xy, strict=True):
            self.placement_initializer.append_sampler(UniformRandomSampler(
                name=f"{cube.name}_sampler",
                mujoco_objects=cube,
                x_range=[sx - jx, sx + jx],
                y_range=[sy - jy, sy + jy],
                rotation=None,                        # random yaw, as T1
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset,
                z_offset=0.01,
            ))

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=list(self.cubes),
        )

    def _setup_references(self) -> None:
        super()._setup_references()
        self.cube_body_ids = tuple(
            self.sim.model.body_name2id(c.root_body) for c in self.cubes
        )


class _Scene(_LiftWithBins, _TwoCubeLift):
    """T2's bins on top of the two-cube Lift.

    MRO: _LiftWithBins._load_model calls super()._load_model(), which lands in
    _TwoCubeLift's instead of Lift's -- so the bin code is T2's, not a copy.
    """

    def __init__(self, cfg: TaskConfig, **kwargs):
        super().__init__(cfg.bins, cfg=cfg, **kwargs)


class InstructedPickPlace(PickPlaceTwoBins):
    """Implements ``core.env_api.Env``. T2 with a second cube and a spoken target.

    ``pairs`` picks which (colour, side) targets reset may draw: the recorded
    ones, or the held-out ones. ``wording`` picks the recorded template or an
    eval-only paraphrase.
    """

    def __init__(
        self,
        config: TaskConfig | None = None,
        *,
        pairs: Pairs = "seen",
        wording: Wording = "template",
    ):
        # Not super().__init__: T2's builds a one-cube scene. Everything below
        # except the scene class mirrors it field for field.
        self._cfg = config or TaskConfig.load()
        self._pairs = self._cfg.pairs(pairs)
        self._wordings = (
            (self._cfg.template,) if wording == "template" else self._cfg.paraphrases
        )
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
        self._z0 = np.zeros(2)
        self._max_z = np.zeros(2)
        self._gripper_span = self._measure_gripper_span()

    # --- Env protocol ------------------------------------------------------

    @property
    def prompt(self) -> str:
        """This episode's instruction. Changes on every reset."""
        if self._instruction is None:
            raise RuntimeError("no instruction before the first reset()")
        return self._instruction.prompt

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)                      # see T1's env.py
        self._env.reset()                             # placement draws first
        self._instruction = self._draw_instruction()
        self._paint()
        self._raw = self._env._get_observations(force_update=True)
        self._t = 0
        # Per-cube lift references, as T1's single one: from the spawn, not
        # the table, since the sampler drops cubes from a z_offset.
        self._z0 = self._cube_positions()[:, 2].copy()
        self._max_z = self._z0.copy()
        return self._observe()

    def step(self, action) -> StepResult:
        act = validate_action(action, PANDA)
        arm = act[PANDA.arm_slice]
        opening = float(act[PANDA.gripper_slice][0])
        self._raw, _, _, _ = self._env.step(np.concatenate([arm, [1.0 - 2.0 * opening]]))
        self._t += 1

        cubes = self._cube_positions()
        self._max_z = np.maximum(self._max_z, cubes[:, 2])

        success = self.which_bin(self.cube_pos) == self.target_bin
        truncated = (not success) and self._t >= self._cfg.horizon_steps
        return StepResult(
            observation=self._observe(),
            success=success,
            terminated=success,
            truncated=truncated,
            failure_mode=self._diagnose(self.cube_pos) if truncated else FailureMode.NONE,
        )

    # --- privileged access, for scripted experts and diagnostics only ------

    @property
    def instruction(self) -> Instruction:
        if self._instruction is None:
            raise RuntimeError("no instruction before the first reset()")
        return self._instruction

    @property
    def cube_pos(self) -> np.ndarray:
        """The *named* cube. T1's expert reads this name, so it aims correctly."""
        return self._cube_positions()[self.instruction.target_slot].copy()

    @property
    def distractor_pos(self) -> np.ndarray:
        return self._cube_positions()[1 - self.instruction.target_slot].copy()

    @property
    def target_bin(self) -> int:
        return self.bin_names.index(self.instruction.side)

    def instruction_failure(self) -> Literal["wrong_cube", "wrong_bin"] | None:
        """Which half of the instruction the current state gets wrong, if either.

        Split out of WRONG_TARGET for diagnostics. It is a method rather than a
        StepResult field because StepResult has no info dict, on purpose (see
        env_api.py), and T2's ``between_bins`` set the precedent.
        """
        if self._lifted(1 - self.instruction.target_slot) and not self._lifted(
            self.instruction.target_slot
        ):
            return "wrong_cube"
        landed = self.which_bin(self.cube_pos)
        if landed is not None and landed != self.target_bin:
            return "wrong_bin"
        return None

    # --- internals ---------------------------------------------------------

    def _draw_instruction(self) -> Instruction:
        """Target pair, then distractor colour, then slot, then wording.

        Global RNG throughout, after placement, so the whole episode is a pure
        function of the seed -- as T2's coin was.
        """
        color, side = self._pairs[np.random.randint(len(self._pairs))]
        others = [c for c in self._cfg.palette if c != color]
        distractor = others[np.random.randint(len(others))]
        slot = int(np.random.randint(2))
        wording = self._wordings[np.random.randint(len(self._wordings))]
        return Instruction(
            color=color,
            side=side,
            distractor_color=distractor,
            target_slot=slot,
            prompt=wording.format(color=color, side=side),
        )

    def _paint(self) -> None:
        model = self._env.sim.model._model
        ins = self._instruction
        colors = [ins.distractor_color, ins.distractor_color]
        colors[ins.target_slot] = ins.color
        for body_id, color in zip(self._env.cube_body_ids, colors, strict=True):
            model.geom_rgba[model.geom_bodyid == body_id] = self._cfg.palette[color]

    def _cube_positions(self) -> np.ndarray:
        xpos = self._env.sim.data.body_xpos
        return np.array([xpos[i] for i in self._env.cube_body_ids], dtype=float)

    def _lifted(self, slot: int) -> bool:
        return bool(self._max_z[slot] > self._z0[slot] + self._cfg.lift_threshold)

    def _diagnose(self, cube: np.ndarray) -> FailureMode:
        if self.instruction_failure() is not None:
            return FailureMode.WRONG_TARGET
        if not self._lifted(self.instruction.target_slot):
            return FailureMode.NO_GRASP
        return FailureMode.DROPPED
