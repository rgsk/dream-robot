"""A RoboCasa kitchen task behind our Env shape, for the agent server.

Runs in ``envs/robocasa/.venv`` only: RoboCasa pins its own Python, robosuite
and MuJoCo (envs/README.md). This module imports nothing from the main venv's
robosuite tasks.

**What maps to what.** The robot is RoboCasa's PandaOmron: a Panda on a wheeled
base. ``observation.state`` is the arm's 7 joints + gripper, as for every Panda
task here; the base and torso are action-only for now. Cameras are renamed to
canonical names (schema rule 3); ``render()`` is the wide front view.

**Seeds: a seeded reset builds the env anew.** Replacing the env's ``rng`` and resetting
was not enough: RoboCasa's reset carries state over from the episode before -- seed 1's
potato alternated between y -3.62 and -4.75 on successive resets of one env, the same
kitchen each time. So ``reset(seed=...)`` builds a fresh env with that seed (RoboCasa's
own seeding, via ``create_env``), and a seed names one scene. An unseeded reset keeps
the env.

**The arm takes absolute joint targets, as every Panda task here** (spec.md's action
contract): RoboCasa's default arm controller is OSC end-effector deltas, so ours is
swapped in through its config loader. ``step`` takes our 8 numbers (7 joints, gripper
opening 0-1) and holds the base and torso where reset put them -- the robot stands
still; driving the base is a later step. RoboCasa's own success check is passed on as
``success``, which only a referee reads (agent/server.py never exposes it).

**Stiffer arm than T1's (kp ARM_KP, not 150).** At 150 this arm sags 0.2-0.35 rad off its
joint targets. The primitives step the IK from where the arm *is*, at most 0.06 rad a step,
so a sag larger than a step is never caught up: a 38 cm move covered 8 cm in its 5 s. At
500 the worst joint is 0.09 rad off (a 12 Nm wrist motor), the rest < 0.02.

**A ready pose: the hand READY_DROP straight down from where reset left it.** RoboCasa
starts this arm almost straight up (hand z 1.55), elbow (joint 4) at -0.15 rad against a
-0.07 limit. Reaching down and forward needs the elbow to straighten past that limit; the
IK does not know limits, so it kept asking and the arm stayed put (a 38 cm move stalled
29 cm short). Lowering the hand straight down bends the elbow, and stays over the robot's
own base -- free space. A fixed "45 cm ahead at 1.20 m" instead landed inside this
kitchen's coffee machine (joint 2 at its 80 Nm limit): a pose chosen without looking is
not safe here. Solved offline (scratch state, positions only, joints clamped to their
limits, seeded bent), then driven to; gripper down as reset left it.

**Depth for aiming.** ``rgbd`` and ``robot_mask`` are T7's, on this env's cameras, so
``point_at`` / ``object_at`` work here unchanged.
"""

from __future__ import annotations

import contextlib
from unittest import mock

import mujoco
import numpy as np

from dream_robot.core.camera import RGBD
from dream_robot.core.env_api import FailureMode, StepResult
from dream_robot.core.schema import PANDA, Embodiment, Observation, build_observation, validate_action

# backend camera -> canonical name
# backend camera -> canonical name: the observation's cameras
CAMERAS = {"robot0_agentview_center": "front", "robot0_eye_in_hand": "wrist"}
# Calibrated depth for aiming (point_at / object_at): left / right see the counter past
# the arm, where front (centre) is mostly arm. Rendered on demand, never observed, so
# their names need not be canonical.
DEPTH_CAMERAS = {"robot0_agentview_left": "left", "robot0_agentview_right": "right", **CAMERAS}
RENDER_CAMERA = "robot0_frontview"
GRIPPER_SPAN = 0.08          # metres, Panda fingers fully open
# The driver prompt's "## The cell" section for a kitchen (agent/server.py swaps it in).
CELL_PROMPT = """## The cell
- A kitchen. The robot stands on a wheeled base that stays still: only the arm moves. It
  reaches about 0.85 m from its shoulder.
- Cameras: `scene` (behind the robot, wide), `front` (from the robot's head, centre -- the
  arm hides much of the counter), `wrist` (on the hand, looking down between the fingers);
  for `point_at` / `object_at` also `left` and `right` (from the robot's head, either
  side): they see the counter past the arm -- aim with these.
- Points are metres, world frame, z up from the floor. There is no one table height:
  counters, appliances and shelves differ. `object_at` gives `support_z`, the height of
  what a thing stands on; use it where this prompt says "the table top" (z = 0.80 there is
  another cell's table).
- No `find_objects` here (it needs one known table): aim with `object_at` / `point_at`.
- A cup or mug comes back as a container, with a `rim_grasp`: it closes on the wall, one
  finger inside, one outside. That is how to pick one up; it held a 3.5 mm mug wall.
"""
HORIZON_SECONDS = 300.0      # driving by hand: as T7's server runs
EEF_SITE = "gripper0_right_grip_site"
READY_DROP = 0.25            # metres the hand is lowered at reset
READY_STEPS = 60             # control steps to drive there (3 s at 20 Hz)...
READY_SETTLE = 40            # ...then held, so the latched "down" is down
READY_SEED = np.array([0.0, np.pi / 16, 0.0, -np.pi / 2 - np.pi / 3, 0.0, np.pi - 0.2, np.pi / 4])
ARM_KP = 500                 # see the module docstring: 150 sags past the IK's step


class Kitchen:
    def __init__(self, task: str = "CoffeeSetupMug", *, camera_hw=(256, 256),
                 render_hw=(512, 512), horizon_seconds: float = HORIZON_SECONDS):
        self.task = task
        self._render_hw = render_hw
        self._horizon_seconds = float(horizon_seconds)
        self._camera_hw = camera_hw
        self._env = self._build(0)
        self._raw: dict | None = None
        self._hold: np.ndarray | None = None     # the 13-dim action, base + torso held

    # --- what the arm primitives (sims/robosuite/skills.py) read ------------

    @property
    def arm_qpos_idx(self) -> np.ndarray:
        return np.asarray(self._env.robots[0]._ref_joint_pos_indexes)

    @property
    def _gripper_span(self) -> float:
        return GRIPPER_SPAN

    @property
    def horizon_steps(self) -> int:
        return int(round(self._horizon_seconds * self.control_hz))

    @property
    def cell_prompt(self) -> str:
        return CELL_PROMPT

    @property
    def depth_cameras(self) -> tuple[str, ...]:
        return tuple(DEPTH_CAMERAS.values())

    @property
    def embodiment(self) -> Embodiment:
        return PANDA

    @property
    def control_hz(self) -> float:
        return float(self._env.control_freq)

    @property
    def prompt(self) -> str:
        return self._env.get_ep_meta().get("lang", "")

    def _build(self, seed: int):
        import robocasa.utils.env_utils as env_utils

        with _joint_position_arm(env_utils):
            return env_utils.create_env(
                self.task, seed=seed, camera_names=list(CAMERAS),
                camera_heights=self._camera_hw[0], camera_widths=self._camera_hw[1],
            )

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self._env.close()
            self._env = self._build(seed)
        self._raw = self._env.reset()
        # Base velocity 0, torso at its reset height, the trailing mode entry 0.
        cc = self._env.robots[0].composite_controller
        self._hold = np.zeros(self._env.action_spec[0].shape)
        s, e = cc._action_split_indexes["torso"]
        torso = self._env.robots[0]._ref_joints_indexes_dict["torso"]
        self._hold[s:e] = self._env.sim.data.qpos[torso]
        self._go_ready()
        return self._observe()

    def _go_ready(self) -> None:
        from dream_robot.sims.robosuite.ik import DifferentialIK

        model, data = self.mujoco
        scratch = mujoco.MjData(model)
        scratch.qpos[:] = data.qpos
        mujoco.mj_fwdPosition(model, scratch)       # mj_jacSite needs the com terms too
        ik = DifferentialIK(model, scratch, site_name=EEF_SITE, qpos_idx=self.arm_qpos_idx,
                            damping=0.05, max_joint_step=0.2)
        goal = ik.site_pos - [0.0, 0.0, READY_DROP]
        down = ik.site_mat                           # reset's gripper-down, latched first
        joints = [model.jnt_qposadr.tolist().index(i) for i in self.arm_qpos_idx]
        lo, hi = model.jnt_range[joints, 0] + 0.05, model.jnt_range[joints, 1] - 0.05
        scratch.qpos[self.arm_qpos_idx] = READY_SEED
        mujoco.mj_fwdPosition(model, scratch)
        for _ in range(300):
            q = np.clip(ik.solve(goal, down), lo, hi)
            scratch.qpos[self.arm_qpos_idx] = q
            mujoco.mj_fwdPosition(model, scratch)
            if np.linalg.norm(ik.site_pos - goal) < 1e-3:
                break
        q0 = np.asarray(data.qpos[self.arm_qpos_idx], dtype=float)
        for i in range(1, READY_STEPS + READY_SETTLE):
            target = q0 + (q - q0) * min(i / READY_STEPS, 1.0)
            self._raw, _, _, _ = self._env.step(self._full(target, opening=1.0))

    def _full(self, joints: np.ndarray, opening: float) -> np.ndarray:
        """Our 7 joints + opening as RoboCasa's 13-dim action, base and torso held."""
        cc = self._env.robots[0].composite_controller
        full = self._hold.copy()
        s, e = cc._action_split_indexes["right"]
        full[s:e] = joints
        s, e = cc._action_split_indexes["right_gripper"]
        full[s:e] = 1.0 - 2.0 * opening                       # 1 open -> -1
        return full

    def step(self, action) -> StepResult:
        act = validate_action(action, PANDA)
        full = self._full(act[PANDA.arm_slice], float(act[PANDA.gripper_slice][0]))
        self._raw, _, _, _ = self._env.step(full)
        return StepResult(
            observation=self._observe(), success=bool(self._env._check_success()),
            terminated=False, truncated=False, failure_mode=FailureMode.NONE,
        )

    def rgbd(self, camera: str = "front", hw: tuple[int, int] = (256, 256)) -> RGBD:
        """A calibrated depth frame, as T7's ``rgbd`` (same conventions, see there)."""
        backend = {v: k for k, v in DEPTH_CAMERAS.items()}[camera]
        sim = self._env.sim
        h, w = hw
        rgb, depth = sim.render(width=w, height=h, camera_name=backend, depth=True)
        rgb, depth = rgb[::-1], depth[::-1]
        m = sim.model._model
        near, far = m.vis.map.znear * m.stat.extent, m.vis.map.zfar * m.stat.extent
        metres = near / (1.0 - depth * (1.0 - near / far))
        cam_id = sim.model.camera_name2id(backend)
        f = 0.5 * h / np.tan(np.deg2rad(m.cam_fovy[cam_id]) / 2)
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = sim.data.cam_xmat[cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = sim.data.cam_xpos[cam_id]
        return RGBD(rgb=np.ascontiguousarray(rgb), depth=metres, K=K, cam_to_world=T)

    def robot_mask(self, camera: str = "front", hw: tuple[int, int] = (256, 256)) -> np.ndarray:
        """(H, W) bool: the robot's own pixels (self-filtering), as T7's."""
        backend = {v: k for k, v in DEPTH_CAMERAS.items()}[camera]
        sim = self._env.sim
        h, w = hw
        seg = sim.render(width=w, height=h, camera_name=backend, segmentation=True)[::-1]
        m = sim.model._model
        roots = {int(m.body_rootid[sim.model.body_name2id(r.robot_model.root_body)])
                 for r in self._env.robots}
        is_robot_geom = np.isin(m.body_rootid[m.geom_bodyid], list(roots))
        is_geom = seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
        return is_geom & is_robot_geom[np.clip(seg[..., 1], 0, m.ngeom - 1)]

    def render(self) -> np.ndarray:
        h, w = self._render_hw
        frame = self._env.sim.render(width=w, height=h, camera_name=RENDER_CAMERA)
        return np.ascontiguousarray(frame[::-1])

    def frame(self, camera: str, hw: tuple[int, int] = (512, 512)) -> np.ndarray:
        """One RGB frame from "scene" (the render camera) or a canonical camera, any size.
        For recording; the observation's own images stay at their size."""
        backend = RENDER_CAMERA if camera == "scene" else {v: k for k, v in CAMERAS.items()}[camera]
        h, w = hw
        return np.ascontiguousarray(self._env.sim.render(width=w, height=h, camera_name=backend)[::-1])

    def view(self) -> tuple[tuple[float, float, float], float, float, float]:
        """Where a viewer should look: (lookat, distance, azimuth, elevation).

        A kitchen's origin can be inside a wall; aim at the robot instead -- a point
        0.6 m ahead of its base at counter height, seen from behind it and a little to
        the side. MuJoCo's azimuth is the camera's heading, so the robot's yaw + 30.
        """
        x, y, _ = self._raw["robot0_base_pos"]
        qx, qy, qz, qw = self._raw["robot0_base_quat"]           # robosuite: xyzw
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        ahead = np.array([x, y]) + 0.6 * np.array([np.cos(yaw), np.sin(yaw)])
        return (float(ahead[0]), float(ahead[1]), 0.9), 2.2, float(np.degrees(yaw)) + 30.0, -30.0

    @property
    def mujoco(self):
        """(model, data). Rebuilt on every reset -- re-read after one."""
        return self._env.sim.model._model, self._env.sim.data._data

    def close(self) -> None:
        self._env.close()

    def _observe(self) -> Observation:
        raw = self._raw
        q = raw["robot0_gripper_qpos"]
        opening = float(np.clip(abs(q[0] - q[1]) / GRIPPER_SPAN, 0.0, 1.0))
        return build_observation(
            joint_positions=raw["robot0_joint_pos"][:7],
            gripper_openings=[opening],
            images={c: np.ascontiguousarray(raw[f"{b}_image"][::-1]) for b, c in CAMERAS.items()},
            embodiment=PANDA,
        )


@contextlib.contextmanager
def _joint_position_arm(env_utils):
    """create_env builds its controller config itself; swap its loader for one whose arm
    takes absolute joint targets (T1's JOINT_POSITION settings but ARM_KP), the rest
    RoboCasa's."""
    load = env_utils.load_composite_controller_config

    def ours(controller=None, robot=None):
        cfg = load(controller=controller, robot=robot)
        cfg["body_parts"]["right"] = {
            "type": "JOINT_POSITION", "input_type": "absolute",
            "input_max": 1, "input_min": -1, "output_max": 0.05, "output_min": -0.05,
            "kp": ARM_KP, "damping_ratio": 1, "impedance_mode": "fixed",
            "kp_limits": [0, 1000], "damping_ratio_limits": [0, 10], "qpos_limits": None,
            "interpolation": None, "ramp_ratio": 0.2,
            "gripper": cfg["body_parts"]["right"]["gripper"],
        }
        return cfg

    with mock.patch.object(env_utils, "load_composite_controller_config", ours):
        yield
