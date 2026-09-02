"""A simulator-free ``Env`` and ``Policy``, for testing everything above the seam.

``core`` is supposed to work without a simulator -- that is the whole argument
for the dataset-on-disk seam -- so the recorder's tests run against a fake whose
dynamics are three lines of arithmetic. If these tests ever need robosuite,
something has leaked downward.

**Why this is not in conftest.py.** ``draccus``, which lerobot depends on,
installs its own top-level ``tests`` package into site-packages, so
``from tests.conftest import ...`` imports draccus's test suite instead of ours.
Shared fixtures still live in conftest.py; anything a test needs to *import* --
a class to subclass, a constant to assert against -- lives here and is imported
as a plain sibling module, which pytest puts on sys.path for us.

Deliberately **not** a Panda. ``Embodiment(arm_joints=3)`` and 32x32 cameras
would break instantly on anything that hardcoded spec.md's 8-dim state or the
128x128 image size, which is exactly the assumption a bimanual task or a second
backend will violate first (ROADMAP rule 2).
"""

from __future__ import annotations

import numpy as np

from dream_robot.core.env_api import FailureMode, StepResult
from dream_robot.core.schema import Embodiment, build_observation

FAKE_EMBODIMENT = Embodiment(arm_joints=3, grippers=1)
# Above core.dataset.MIN_IMAGE_SIDE: below it the AV1 encoder takes the whole
# process down with SIGFPE rather than raising. Still tiny enough that a
# round-trip test encodes and decodes in well under a second.
FAKE_HW = (32, 32)

# Render bigger than the cameras, as every real task does: the video wants a
# legible wide view while the observation stays pinned to the dataset's
# resolution. observation_panel scales the camera column to the wide view's
# height by integer repetition, so the two must divide.
FAKE_RENDER_HW = (64, 64)


class FakeEnv:
    """Implements ``core.env_api.Env`` with a first-order lag and a step counter.

    ``lag`` is the fraction of the commanded move left undone after one step, so
    ``|action[t] - state[t+1]| == lag * |action[t] - state[t]|``. That makes the
    recorder's tracking check something a test can aim at: ``lag=0.1`` is a
    well-behaved absolute-position controller, ``lag=1.0`` is a controller that
    ignores its target, which is what delta mode looks like from outside.

    Every camera frame is filled with the step index, so a test can assert which
    observation ended up next to which action instead of trusting the loop.
    """

    def __init__(
        self,
        *,
        embodiment: Embodiment = FAKE_EMBODIMENT,
        succeed_after: int | None = 5,
        horizon: int = 8,
        lag: float = 0.1,
        hw: tuple[int, int] = FAKE_HW,
        render_hw: tuple[int, int] = FAKE_RENDER_HW,
        failure_mode: FailureMode = FailureMode.NO_GRASP,
        cameras: tuple[str, ...] = ("top", "wrist"),
    ):
        self._embodiment = embodiment
        self._succeed_after = succeed_after
        self._horizon = horizon
        self._lag = lag
        self._hw = hw
        self._render_hw = render_hw
        self._failure_mode = failure_mode
        self._cameras = cameras
        self._state = np.zeros(embodiment.dim, dtype=np.float32)
        self._t = 0
        self.closed = False
        self.seeds: list[int | None] = []

    @property
    def embodiment(self) -> Embodiment:
        return self._embodiment

    @property
    def control_hz(self) -> float:
        return 30.0

    def reset(self, *, seed: int | None = None):
        self.seeds.append(seed)
        rng = np.random.default_rng(seed)
        self._state = np.zeros(self._embodiment.dim, dtype=np.float32)
        self._state[self._embodiment.arm_slice] = rng.normal(0, 0.01, self._embodiment.arm_joints)
        self._state[self._embodiment.gripper_slice] = 0.5
        self._t = 0
        return self._observe()

    def step(self, action) -> StepResult:
        act = np.asarray(action, dtype=np.float32)
        arm = self._embodiment.arm_slice
        self._state[arm] = act[arm] - self._lag * (act[arm] - self._state[arm])
        self._state[self._embodiment.gripper_slice] = act[self._embodiment.gripper_slice]
        self._t += 1

        success = self._succeed_after is not None and self._t >= self._succeed_after
        truncated = (not success) and self._t >= self._horizon
        return StepResult(
            observation=self._observe(),
            success=success,
            terminated=success,
            truncated=truncated,
            failure_mode=self._failure_mode if truncated else FailureMode.NONE,
        )

    def render(self) -> np.ndarray:
        return np.full((*self._render_hw, 3), 7, dtype=np.uint8)

    def close(self) -> None:
        self.closed = True

    def _observe(self):
        return build_observation(
            joint_positions=self._state[self._embodiment.arm_slice],
            gripper_openings=self._state[self._embodiment.gripper_slice],
            images={
                name: np.full((*self._hw, 3), self._t % 256, dtype=np.uint8)
                for name in self._cameras
            },
            embodiment=self._embodiment,
        )


class FakePolicy:
    """Commands ``state + delta`` on the arm and holds the gripper half open.

    Because the action is a fixed offset from the observation it was given,
    ``action[t] - state[t] == delta`` exactly -- so a test can prove the
    recorder paired each observation with the action taken in response to it,
    rather than with its neighbour.
    """

    def __init__(self, *, delta: float = 0.05, embodiment: Embodiment = FAKE_EMBODIMENT):
        self.delta = delta
        self._embodiment = embodiment
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def __call__(self, observation) -> np.ndarray:
        action = np.array(observation.state, dtype=np.float32, copy=True)
        action[self._embodiment.arm_slice] += self.delta
        action[self._embodiment.gripper_slice] = 0.5
        return action
