"""Two arms, one tray, real physics: the bimanual data generator working.

The unit tests next door prove the barrier holds in arithmetic. This proves it
holds against a simulator, and that the 16-dim action it produces is still the
contract spec.md froze.

Slow: real physics and offscreen rendering.
"""

from dataclasses import replace

import numpy as np
import pytest

from dream_robot.core.env_api import FailureMode
from dream_robot.core.schema import BIMANUAL_PANDA, joint_tracking_error, validate_action

pytestmark = pytest.mark.slow

pytest.importorskip("robosuite")

from dream_robot.sims.robosuite.tasks.two_arm_lift.env import TwoArmLiftTask  # noqa: E402
from dream_robot.sims.robosuite.tasks.two_arm_lift.expert import (  # noqa: E402
    ExpertConfig,
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
        validate_action(action, BIMANUAL_PANDA)     # the contract, every step
        actions.append(action)
        states.append(obs.state)
        phases.append(tuple(command.arms[i].phase for i in range(2)))
        result = env.step(action)
        obs = result.observation
        if result.success or result.truncated or command.timed_out:
            break
    return result, np.array(actions), np.array(states), phases


@pytest.fixture(scope="module")
def rig():
    env = TwoArmLiftTask()
    yield env, ExpertPolicy(env)
    env.close()


def test_expert_lifts_the_tray_without_spilling_it(rig):
    env, policy = rig
    result, _, _, phases = rollout(env, policy, seed=1000)
    assert result.success, f"failed in {phases[-1]} as {result.failure_mode}"
    assert result.failure_mode is FailureMode.NONE
    # Every ball, not "most". A clean lift losing the odd ball looked like the
    # price of a packed tray and was really a placement bug: the spawn lattice
    # was laid out in world axes while the tray is randomly yawed, so the
    # corners of the grid hung over the rim. Now that it rotates with the tray,
    # all 20 eval seeds finish 25 of 25, and anything less is a regression.
    assert env.marbles_inside == env._cfg.marbles.count
    assert env._pot_tilt_deg() < env._cfg.tilt_threshold_deg


def test_the_arms_never_leave_each_other_behind(rig):
    """Lockstep, measured against physics rather than asserted in the config."""
    env, policy = rig
    _, _, _, phases = rollout(env, policy, seed=1000)
    assert all(left == right for left, right in phases)


def test_neither_arm_lifts_before_both_grippers_have_closed(rig):
    """The barrier's whole purpose, read off the recorded episode.

    At the first frame of `lift`, both grippers must already be shut. If one is
    still open, that arm never grasped and the other is about to lever the tray
    up by one handle -- a demonstration of the failure, recorded as if it were
    the skill.
    """
    env, policy = rig
    _, actions, states, phases = rollout(env, policy, seed=1000)
    first_lift = next(i for i, p in enumerate(phases) if p[0] == "lift")
    openings = states[first_lift][BIMANUAL_PANDA.gripper_slice]
    assert np.all(openings < 0.5), f"gripper openings at the lift: {openings}"
    assert np.all(actions[first_lift][BIMANUAL_PANDA.gripper_slice] == 0.0)


def test_actions_are_absolute_joint_targets_that_track(rig):
    """spec.md's integrity invariant at 16 dimensions.

    The bimanual version of T1's test, and the one that catches a swapped pair
    of seven-joint halves: a transposed action still has the right shape, but
    arm 0's targets would then be compared against arm 1's realised angles and
    the tracking error blows up.
    """
    env, policy = rig
    _, actions, states, _ = rollout(env, policy, seed=1000)
    mean, mx = joint_tracking_error(actions, states, BIMANUAL_PANDA)
    assert mean < 0.05, f"joint tracking mean {mean:.4f} rad -- controller not following"
    assert mx <= policy._cfg.ik.max_joint_step + 1e-6


def test_episode_length_varies_with_the_scene(rig):
    env, policy = rig
    lengths = []
    for seed in (1001, 1004, 1005):
        result, actions, _, _ = rollout(env, policy, seed=seed)
        assert result.success, f"seed {seed} failed: {result.failure_mode}"
        lengths.append(len(actions))
    assert len(set(lengths)) > 1, f"all episodes were {lengths[0]} steps -- fixed timing?"


def test_without_the_barrier_the_same_expert_desynchronises(rig):
    """The ablation, in physics: this is what the barrier is buying.

    Seed 1001 with the barrier lifts the tray; without it, arm 0 grasps, lifts
    alone and levers the tray away from arm 1, which never gets its handle. The
    episode ends as DESYNCHRONISED -- the bucket T6 added to the histogram.
    """
    env, _ = rig
    loose = ExpertPolicy(env, replace(ExpertConfig.load(), sync_barrier=False))
    result, _, _, phases = rollout(env, loose, seed=1001)
    assert not result.success
    assert result.failure_mode is FailureMode.DESYNCHRONISED
    assert any(left != right for left, right in phases), "arms never diverged at all"


def test_success_needs_the_tray_held_up_not_just_lifted(rig):
    """The lift has to survive two seconds, not just happen.

    Without the hold, success fires on the frame the tray crosses the height --
    which is how a tray hoisted at 48 degrees scored a clean success, having had
    no time yet to fall or to spill. Assert the gap directly: the episode cannot
    end on the frame it first goes above the line.
    """
    env, policy = rig
    obs = env.reset(seed=1000)
    policy.reset()
    first_above = None
    for step in range(env._cfg.horizon_steps):
        action, _ = policy(obs)
        result = env.step(action)
        obs = result.observation
        if first_above is None and env._pot_lift() > env._cfg.lift_height:
            first_above = step
        if result.success:
            assert first_above is not None
            assert step - first_above >= env._cfg.hold_steps - 1, (
                f"succeeded {step - first_above} steps after clearing the height, "
                f"hold is {env._cfg.hold_steps}"
            )
            return
    raise AssertionError("expert never succeeded")
