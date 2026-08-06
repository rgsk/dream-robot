"""The robosuite adapter, checked against the same contract as FakeEnv.

Slow: each test builds a MuJoCo scene and renders offscreen. Marked so the fast
core suite stays instant.
"""

import numpy as np
import pytest

from dream_robot.core.env_api import Env, FailureMode, check_env_conformance
from dream_robot.core.schema import PANDA

pytestmark = pytest.mark.slow

robosuite = pytest.importorskip("robosuite")

from dream_robot.sims.robosuite.tasks.pick_place_cube.env import (  # noqa: E402
    PickPlaceCube,
    TaskConfig,
)


@pytest.fixture(scope="module")
def env():
    e = PickPlaceCube()
    yield e
    e.close()


def test_config_loads_from_yaml():
    cfg = TaskConfig.load()
    assert cfg.control_hz == 30.0                      # spec.md, frozen
    assert cfg.horizon_steps == 600                    # 20 s at 30 Hz
    assert cfg.camera_mapping == {
        "agentview": "top", "robot0_eye_in_hand": "wrist",
    }


def test_satisfies_env_protocol(env):
    assert isinstance(env, Env)
    assert Env not in type(env).__mro__            # wraps, does not inherit


def test_conformance(env):
    """The same function FakeEnv passes, now against real physics."""
    check_env_conformance(env)


def test_action_space_is_absolute_joint_targets(env):
    """spec.md bans EE-delta. The underlying controller must be 8-dim joint."""
    assert env._env.action_dim == PANDA.dim == 8


def test_observation_carries_no_ground_truth(env):
    obs = env.reset(seed=0)
    assert obs.state.shape == (8,) and obs.state.dtype == np.float32
    assert set(obs.images) == {"top", "wrist"}
    # cube_pos exists in the backend observation and must not reach here.
    assert "cube_pos" in env._raw
    assert not hasattr(obs, "cube_pos")


def test_gripper_is_normalised_not_raw_finger_metres(env):
    obs = env.reset(seed=0)
    opening = float(obs.state[PANDA.gripper_slice][0])
    assert 0.0 <= opening <= 1.0
    # Panda rests open, so a raw finger width (~0.08 m) would fail this.
    assert opening > 0.5


def test_seeding_is_reproducible(env):
    a = env.reset(seed=7)
    cube_a = env._raw["cube_pos"].copy()
    env.reset(seed=8)
    b = env.reset(seed=7)
    assert np.allclose(cube_a, env._raw["cube_pos"])
    assert np.allclose(a.state, b.state)


def test_different_seeds_move_the_cube(env):
    env.reset(seed=1)
    one = env._raw["cube_pos"].copy()
    env.reset(seed=2)
    assert not np.allclose(one, env._raw["cube_pos"])


def test_images_are_not_upside_down(env):
    """robosuite renders bottom-up; the adapter must flip.

    The table is bright and the backdrop above it is dark, so the lower half of
    a correctly-oriented `top` frame is brighter than the upper half.
    """
    obs = env.reset(seed=0)
    top = obs.images["top"].astype(float)
    h = top.shape[0]
    assert top[h // 2:].mean() > top[: h // 2].mean()


def test_bin_is_visible_to_the_cameras(env):
    """group=0 geoms collide but are never drawn. The bin is blue; find it."""
    obs = env.reset(seed=0)
    top = obs.images["top"].astype(int)
    blue = (top[..., 2] - top[..., 0] > 40) & (top[..., 2] > 60)
    assert blue.sum() > 20, f"only {blue.sum()} blue pixels; is the bin group=0?"


def test_idle_episode_times_out_as_no_grasp(env):
    """Holding still never grasps, so the failure bucket must say so."""
    obs = env.reset(seed=0)
    hold = np.concatenate([obs.state[PANDA.arm_slice], [1.0]]).astype(np.float32)
    result = None
    for _ in range(env._cfg.horizon_steps):
        result = env.step(hold)
        if result.truncated or result.terminated:
            break
    assert result.truncated and not result.success
    assert result.failure_mode is FailureMode.NO_GRASP
