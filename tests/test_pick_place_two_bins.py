"""T2 adapter and coin-flip expert. Slow: real physics and offscreen rendering."""

import numpy as np
import pytest

from dream_robot.core.env_api import Env, FailureMode, check_env_conformance
from dream_robot.core.schema import PANDA

pytestmark = pytest.mark.slow

pytest.importorskip("robosuite")

from dream_robot.core.registry import make_env, make_policy  # noqa: E402
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import (  # noqa: E402
    PickPlaceTwoBins,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.expert import (  # noqa: E402
    ExpertPolicy,
)


@pytest.fixture(scope="module")
def env():
    e = PickPlaceTwoBins()
    yield e
    e.close()


def rollout(env, expert, seed, target):
    obs = env.reset(seed=seed)
    expert.reset()
    expert.target_bin = target
    result = None
    for _ in range(env._cfg.horizon_steps):
        action, _ = expert(obs)
        result = env.step(action)
        obs = result.observation
        if result.terminated or result.truncated:
            break
    return result


def test_config_matches_t1_except_the_bins():
    from dream_robot.sims.robosuite.tasks.pick_place_cube.env import TaskConfig as T1Config

    t1, t2 = T1Config.load(), TaskConfig.load()
    for field in ("control_hz", "horizon_seconds", "camera_hw", "camera_mapping",
                  "cube_xy_range", "resting_z_margin", "lift_threshold"):
        assert getattr(t1, field) == getattr(t2, field), field
    assert t2.bins[0].offset_xy == t1.bin.offset_xy
    assert t2.bins[0].inner_half == t1.bin.inner_half


def test_conformance(env):
    assert isinstance(env, Env)
    check_env_conformance(env)


def test_bins_are_mirrored_across_the_table_centre(env):
    a, b = env.bin_centers
    assert np.allclose(a[:2] * [1, -1], b[:2])


def test_both_bins_are_visible_in_the_top_camera(env):
    """One blue blob on each side of the image."""
    obs = env.reset(seed=0)
    top = obs.images["top"].astype(int)
    blue = (top[..., 2] - top[..., 0] > 40) & (top[..., 2] > 60)
    half = top.shape[1] // 2
    assert blue[:, :half].sum() > 20 and blue[:, half:].sum() > 20


def test_between_bins_is_the_strip_between_them(env):
    (a, b) = env.bin_centers
    mid = (a + b) / 2
    assert env.between_bins(mid)
    assert not env.between_bins(a) and not env.between_bins(b)
    assert not env.between_bins(mid + [0.15, 0.0, 0.0])      # beyond the bins' width


def test_expert_coin_is_a_function_of_the_seed(env):
    expert = ExpertPolicy(env)
    flips = []
    for seed in (3, 4, 3):
        env.reset(seed=seed)
        expert.reset()
        flips.append(expert.target_bin)
    assert flips[0] == flips[2]


def test_expert_coin_lands_both_ways(env):
    expert = ExpertPolicy(env)
    seen = set()
    for seed in range(20):
        env.reset(seed=seed)
        expert.reset()
        seen.add(expert.target_bin)
    assert seen == {0, 1}


@pytest.mark.parametrize("target", [0, 1])
def test_expert_places_in_the_chosen_bin(env, target):
    expert = ExpertPolicy(env)
    result = rollout(env, expert, seed=0, target=target)
    assert result.success, result.failure_mode
    assert env.which_bin(env.cube_pos) == target


def test_idle_episode_times_out_as_no_grasp(env):
    obs = env.reset(seed=0)
    hold = np.concatenate([obs.state[PANDA.arm_slice], [1.0]]).astype(np.float32)
    for _ in range(env._cfg.horizon_steps):
        result = env.step(hold)
        if result.truncated or result.terminated:
            break
    assert result.truncated and result.failure_mode is FailureMode.NO_GRASP


def test_registry_builds_the_t2_env_and_its_expert():
    e = make_env("robosuite/pick_place_two_bins")
    try:
        actor = make_policy("expert", e)
        assert isinstance(actor._policy, ExpertPolicy)
    finally:
        e.close()
