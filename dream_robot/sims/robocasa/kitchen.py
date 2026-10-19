"""A RoboCasa kitchen task behind our Env shape, for the agent server.

Runs in ``envs/robocasa/.venv`` only: RoboCasa pins its own Python, robosuite
and MuJoCo (envs/README.md). This module imports nothing from the main venv's
robosuite tasks.

**What maps to what.** The robot is RoboCasa's PandaOmron: a Panda on a wheeled
base. ``observation.state`` is the arm's 7 joints + gripper, as for every Panda
task here; the base and torso are action-only for now. Cameras are renamed to
canonical names (schema rule 3); ``render()`` is the wide front view.

**Seeds.** RoboCasa draws layout, style and objects from the env's ``rng``;
``reset(seed=...)`` replaces it, so a seed names one kitchen.
"""

from __future__ import annotations

import numpy as np

from dream_robot.core.schema import PANDA, Embodiment, Observation, build_observation

# backend camera -> canonical name
CAMERAS = {"robot0_agentview_center": "front", "robot0_eye_in_hand": "wrist"}
RENDER_CAMERA = "robot0_frontview"
GRIPPER_SPAN = 0.08          # metres, Panda fingers fully open


class Kitchen:
    def __init__(self, task: str = "CoffeeSetupMug", *, camera_hw=(256, 256),
                 render_hw=(512, 512)):
        from robocasa.utils.env_utils import create_env

        self.task = task
        self._render_hw = render_hw
        self._env = create_env(
            task, seed=0, camera_names=list(CAMERAS),
            camera_heights=camera_hw[0], camera_widths=camera_hw[1],
        )
        self._raw: dict | None = None

    @property
    def embodiment(self) -> Embodiment:
        return PANDA

    @property
    def control_hz(self) -> float:
        return float(self._env.control_freq)

    @property
    def prompt(self) -> str:
        return self._env.get_ep_meta().get("lang", "")

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self._env.rng = np.random.default_rng(seed)
        self._raw = self._env.reset()
        return self._observe()

    def render(self) -> np.ndarray:
        h, w = self._render_hw
        frame = self._env.sim.render(width=w, height=h, camera_name=RENDER_CAMERA)
        return np.ascontiguousarray(frame[::-1])

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
