"""Expert + IK + adapter, end to end. The first cell's data generator, working.

Slow: real physics and offscreen rendering.
"""

import numpy as np
import pytest

from dream_robot.core.env_api import FailureMode
from dream_robot.core.schema import PANDA, joint_tracking_error, validate_action

pytestmark = pytest.mark.slow

pytest.importorskip("robosuite")

from dream_robot.sims.robosuite.tasks.pick_place_cube.env import (  # noqa: E402
    PickPlaceCube,
)
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import (  # noqa: E402
    ExpertPolicy,
)


def rollout(env, policy, seed):
    """One scripted episode. Returns (result, actions, states, phases)."""
    obs = env.reset(seed=seed)
    policy.reset()
    actions, states, phases = [], [], []
    result = None
    for _ in range(env._cfg.horizon_steps):
        action, command = policy(obs)
        validate_action(action, PANDA)          # the contract, every single step
        actions.append(action)
        states.append(obs.state)
        phases.append(command.phase)
        result = env.step(action)
        obs = result.observation
        if result.success or result.truncated or command.timed_out:
            break
    return result, np.array(actions), np.array(states), phases


@pytest.fixture(scope="module")
def rig():
    env = PickPlaceCube()
    yield env, ExpertPolicy(env)
    env.close()


def test_expert_places_the_cube(rig):
    env, policy = rig
    result, _, _, phases = rollout(env, policy, seed=0)
    assert result.success, f"failed in phase {phases[-1]!r} as {result.failure_mode}"
    assert result.terminated and not result.truncated
    assert result.failure_mode is FailureMode.NONE


def test_expert_visits_the_phases_in_order(rig):
    """A prefix, not the whole list: the episode terminates the moment the cube
    is in the bin, which happens during `release`. `retreat` only runs when
    success is slower to register.
    """
    env, policy = rig
    _, _, _, phases = rollout(env, policy, seed=0)
    ordered = list(dict.fromkeys(phases))
    full = ["approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat"]
    assert ordered == full[: len(ordered)]
    assert "release" in ordered


def test_actions_are_absolute_joint_targets_that_track(rig):
    """spec.md's integrity invariant, measured on a real episode.

    action[t] and state[t+1] are the same eight numbers in the same units, so a
    recorder writing the wrong column shows up here rather than as a mysterious
    success rate later. Non-zero because the PD loop lags its commanded target.
    """
    env, policy = rig
    _, actions, states, _ = rollout(env, policy, seed=0)
    mean, mx = joint_tracking_error(actions, states, PANDA)
    assert mean < 0.05, f"joint tracking mean {mean:.4f} rad -- controller not following"
    assert mx <= policy._cfg.ik.max_joint_step + 1e-6


def test_episode_length_varies_with_the_scene(rig):
    """The empirical payoff of predicate transitions.

    Different cube placements take different numbers of steps, so 'how far
    through the episode am I' carries no information about what to do -- which
    is what makes spec.md's ban on time features free to honour.
    """
    env, policy = rig
    lengths = []
    for seed in (0, 1, 2, 3):
        result, actions, _, _ = rollout(env, policy, seed=seed)
        assert result.success, f"seed {seed} failed: {result.failure_mode}"
        lengths.append(len(actions))
    assert len(set(lengths)) > 1, f"all episodes were {lengths[0]} steps -- fixed timing?"


def test_gripper_command_stays_in_contract_units(rig):
    env, policy = rig
    _, actions, _, _ = rollout(env, policy, seed=0)
    grip = actions[:, PANDA.gripper_slice]
    assert grip.min() >= 0.0 and grip.max() <= 1.0
    assert set(np.unique(grip)) == {0.0, 1.0}   # the expert is bang-bang
