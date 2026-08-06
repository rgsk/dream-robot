"""The protocol is provable before any simulator exists -- that is the point.

FakeEnv below is a complete, contract-honouring task in ~40 lines. Step 6's
robosuite adapter runs the same check_env_conformance() against real physics.
"""

import ast
import inspect

import numpy as np
import pytest

from dream_robot.core import env_api
from dream_robot.core.env_api import (
    Env,
    FailureMode,
    StepResult,
    check_env_conformance,
)
from dream_robot.core.schema import PANDA, Observation, build_observation


class FakeEnv:
    """A task with no physics: the arm 'moves' by adopting the commanded target.

    Satisfies Env structurally, without inheriting from it.
    """

    def __init__(self, *, horizon: int = 5, res: int = 4, cameras=("top", "wrist")):
        self.embodiment = PANDA
        self.control_hz = 30.0
        self._horizon = horizon
        self._res = res
        self._cameras = cameras
        self._t = 0
        self._state = np.zeros(PANDA.dim, dtype=np.float32)

    def _observe(self) -> Observation:
        return build_observation(
            joint_positions=self._state[PANDA.arm_slice],
            gripper_openings=self._state[PANDA.gripper_slice],
            images={c: np.zeros((self._res, self._res, 3), np.uint8) for c in self._cameras},
            embodiment=PANDA,
        )

    def reset(self, *, seed: int | None = None) -> Observation:
        self._t = 0
        self._state = np.zeros(PANDA.dim, dtype=np.float32)
        return self._observe()

    def step(self, action) -> StepResult:
        self._state = np.asarray(action, dtype=np.float32)   # perfect tracking
        self._t += 1
        success = bool(self._state[0] > 0.5)
        timed_out = self._t >= self._horizon and not success
        return StepResult(
            observation=self._observe(),
            success=success,
            terminated=success,
            truncated=timed_out,
            failure_mode=FailureMode.TIMEOUT if timed_out else FailureMode.NONE,
        )

    def render(self) -> np.ndarray:
        return np.zeros((16, 24, 3), np.uint8)   # wider than the obs cameras

    def close(self) -> None:
        pass


# --- the protocol ----------------------------------------------------------

def test_fake_env_satisfies_protocol_without_inheriting():
    env = FakeEnv()
    assert isinstance(env, Env)
    assert Env not in type(env).__mro__


def test_conformance_passes_on_a_correct_task():
    check_env_conformance(FakeEnv())


# --- what conformance actually catches -------------------------------------

def test_wrong_control_rate_rejected():
    env = FakeEnv()
    env.control_hz = 20.0        # ../rl-prac ran at 20; spec.md freezes 30
    with pytest.raises(ValueError, match="freezes every task at 30 Hz"):
        check_env_conformance(env)


def test_backend_camera_name_rejected_at_the_env_boundary():
    with pytest.raises(ValueError, match="non-canonical camera name"):
        check_env_conformance(FakeEnv(cameras=("agentview",)))


def test_render_must_be_uint8_hwc():
    env = FakeEnv()
    env.render = lambda: np.zeros((3, 16, 16), np.uint8)   # torch layout
    with pytest.raises(ValueError, match=r"render\(\) must return uint8"):
        check_env_conformance(env)


def test_state_dim_mismatch_rejected():
    env = FakeEnv()
    env._observe = lambda: Observation(
        state=np.zeros(7, np.float32),
        images={"top": np.zeros((4, 4, 3), np.uint8)},
    )
    with pytest.raises(ValueError, match=r"shape \(8,\)"):
        check_env_conformance(env)


def test_observation_built_directly_still_checked():
    """An adapter can bypass build_observation. Conformance re-checks anyway."""
    env = FakeEnv()
    env._observe = lambda: Observation(
        state=np.zeros(8, np.float32),
        images={"agentview": np.zeros((4, 4, 3), np.uint8)},
    )
    with pytest.raises(ValueError, match="non-canonical camera name"):
        check_env_conformance(env)


# --- StepResult invariants -------------------------------------------------

def test_success_with_failure_mode_is_incoherent():
    with pytest.raises(ValueError, match="successful step has no failure mode"):
        StepResult(
            observation=FakeEnv().reset(), success=True, terminated=True,
            truncated=False, failure_mode=FailureMode.DROPPED,
        )


def test_terminated_and_truncated_are_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        StepResult(
            observation=FakeEnv().reset(), success=False,
            terminated=True, truncated=True,
        )


def test_timeout_reports_a_failure_mode():
    env = FakeEnv(horizon=2)
    env.reset()
    zero = np.zeros(PANDA.dim, dtype=np.float32)
    env.step(zero)
    last = env.step(zero)
    assert last.truncated and last.failure_mode is FailureMode.TIMEOUT


def test_failure_modes_are_a_closed_set():
    """A free-text reason string would make histograms non-summable."""
    assert {m.value for m in FailureMode} == {
        "none", "no_grasp", "dropped", "wrong_target", "knocked_over", "timeout",
    }


# --- structural guarantees -------------------------------------------------

def test_step_result_has_no_passthrough_channel():
    """No info dict, no reward -- see StepResult's docstring for why."""
    fields = set(StepResult.__dataclass_fields__)
    assert fields == {
        "observation", "success", "terminated", "truncated", "failure_mode"
    }


def test_env_api_imports_no_simulator():
    tree = ast.parse(inspect.getsource(env_api))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"numpy", "dataclasses", "enum", "typing", "__future__", "dream_robot"}
