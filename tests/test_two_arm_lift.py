"""T6 bimanual adapter. Slow: real physics and offscreen rendering."""

import numpy as np
import pytest

from dream_robot.core.env_api import Env, FailureMode, check_env_conformance
from dream_robot.core.schema import CANONICAL_CAMERAS, BIMANUAL_PANDA

pytestmark = pytest.mark.slow

pytest.importorskip("robosuite")

from dream_robot.core.registry import make_env  # noqa: E402
from dream_robot.sims.robosuite.tasks.two_arm_lift.env import (  # noqa: E402
    N_ARMS,
    TwoArmLiftTask,
)


@pytest.fixture(scope="module")
def env():
    e = TwoArmLiftTask()
    yield e
    e.close()


def test_registered():
    e = make_env("robosuite/two_arm_lift")
    assert isinstance(e, Env)
    e.close()


def test_conforms_to_the_contract(env):
    check_env_conformance(env, steps=3, seed=0)


def test_embodiment_is_sixteen_dim(env):
    """ROADMAP rule 2, executed: bimanual is a bigger action dim, not a rewrite."""
    assert env.embodiment == BIMANUAL_PANDA
    assert env.embodiment.dim == 16
    obs = env.reset(seed=0)
    assert obs.state.shape == (16,)
    assert obs.state[env.embodiment.arm_slice].shape == (14,)
    assert obs.state[env.embodiment.gripper_slice].shape == (2,)


def test_four_canonical_cameras_and_no_backend_names(env):
    """Overhead, room-level, one per wrist -- ALOHA's set, which ACT was built on.

    The room-level view is not decoration: straight down cannot show how high a
    gripper is above the table, and failing to close on a handle is BC's
    dominant failure here.
    """
    obs = env.reset(seed=0)
    assert set(obs.images) == {"top", "front", "wrist_left", "wrist_right"}
    assert set(obs.images) <= set(CANONICAL_CAMERAS)
    # The single-arm name must not leak into a two-arm dataset, and a backend
    # name must never survive the adapter.
    assert "wrist" not in obs.images
    assert not any("robot" in name or "agentview" in name for name in obs.images)


def test_action_halves_drive_their_own_arm(env):
    """A transposed or swapped action vector is THE bimanual wiring bug.

    It cannot be caught by shapes -- both halves are 7 long -- and a policy
    trained through a swap still converges, to a robot that moves the wrong arm.
    So: hold arm 0 at its current pose, move only arm 1, and check that exactly
    one of them went anywhere.
    """
    obs = env.reset(seed=0)
    q = obs.state[env.embodiment.arm_slice].reshape(N_ARMS, 7).copy()
    target = q.copy()
    target[1, 0] += 0.20                      # arm 1, joint 0, only

    for _ in range(15):
        action = np.concatenate([target.reshape(-1), [1.0, 1.0]]).astype(np.float32)
        obs = env.step(action).observation

    moved = np.abs(
        obs.state[env.embodiment.arm_slice].reshape(N_ARMS, 7) - q
    ).max(axis=1)
    assert moved[0] < 0.02, f"arm 0 was told to hold and moved {moved[0]:.3f} rad"
    assert moved[1] > 0.10, f"arm 1 was told to move and moved {moved[1]:.3f} rad"


def test_gripper_convention_is_zero_closed_one_open(env):
    """spec.md's 0..1, not robosuite's +1/-1 -- and applied per gripper."""
    obs = env.reset(seed=0)
    arms = obs.state[env.embodiment.arm_slice]
    start = obs.state[env.embodiment.gripper_slice].copy()
    for _ in range(20):
        obs = env.step(
            np.concatenate([arms, [0.0, 0.0]]).astype(np.float32)
        ).observation
    closed = obs.state[env.embodiment.gripper_slice]
    assert np.all(closed < start - 0.1), f"commanding 0.0 did not close: {start} -> {closed}"
    assert np.all(closed >= 0.0) and np.all(closed <= 1.0)


def test_idle_episode_fails_as_no_grasp(env):
    """Both arms still: nothing was ever held, so it is not DESYNCHRONISED."""
    obs = env.reset(seed=0)
    hold = np.concatenate([obs.state[env.embodiment.arm_slice], [1.0, 1.0]]).astype(np.float32)
    result = None
    for _ in range(env._cfg.horizon_steps):
        result = env.step(hold)
        if result.terminated or result.truncated:
            break
    assert result.truncated and not result.success
    assert result.failure_mode is FailureMode.NO_GRASP


def test_success_threshold_comes_from_task_yaml(env):
    """The number lives in the task file, not inside robosuite's _check_success."""
    assert env._cfg.lift_height == pytest.approx(0.10)
    env.reset(seed=0)
    assert env._pot_lift() == pytest.approx(0.0, abs=0.01)
    assert env._pot_tilt_deg() == pytest.approx(0.0, abs=1.0)


# --- marbles ---------------------------------------------------------------

def test_marbles_start_in_the_pot_and_stay_there(env):
    """Placed relative to the pot, which the sampler moves every seed."""
    n = env._cfg.marbles.count
    for seed in (0, 1, 2):
        obs = env.reset(seed=seed)
        assert env.marbles_inside == n, f"seed {seed}: {env.marbles_inside}/{n} at reset"
        hold = np.concatenate(
            [obs.state[env.embodiment.arm_slice], [1.0, 1.0]]
        ).astype(np.float32)
        for _ in range(45):
            env.step(hold)
        assert env.marbles_inside == n, f"seed {seed}: marbles left an untouched pot"


def _roll_pot(env, degrees: float, steps: int = 60):
    """Roll the pot in place by ``degrees`` and let the marbles respond."""
    obs = env.reset(seed=0)
    sim = env._env.sim
    addr = sim.model.get_joint_qpos_addr("pot_joint0")[0]
    half = np.radians(degrees) / 2.0
    sim.data.qpos[addr + 3 : addr + 7] = [np.cos(half), np.sin(half), 0.0, 0.0]
    sim.forward()
    hold = np.concatenate(
        [obs.state[env.embodiment.arm_slice], [1.0, 1.0]]
    ).astype(np.float32)
    for _ in range(steps):
        env.step(hold)


def test_tilt_past_the_threshold_is_knocked_over(env):
    """The numeric failure line. Tilt alone, marbles not consulted.

    ``steps=0``: a free pot resting on the table rights itself from 60 deg in a
    few frames, so letting physics run would measure the table's stability
    rather than the diagnosis. The state under test is the tilted one.
    """
    _roll_pot(env, 60.0, steps=0)
    assert env._pot_tilt_deg() > env._cfg.tilt_threshold_deg
    # Nothing was actually grasped, so the honest diagnosis is NO_GRASP -- the
    # grasp checks come first precisely so a tip never masks the reason.
    assert env._diagnose() is FailureMode.NO_GRASP
    env._ever_grasped = [True, True]
    assert env._diagnose() is FailureMode.KNOCKED_OVER


def test_inverting_the_pot_spills_the_marbles(env):
    """The visible failure. Note the angle: 160 deg, not 45.

    An open box does not spill below ~90 deg -- gravity in the pot's frame keeps
    a component pointing INTO it, so the contents pile against a wall rather
    than reaching the rim. Measured across fills and depths in
    scripts/t6_spill_curve.py. A test written at 45 deg would fail, and it would
    be the test that was wrong.
    """
    _roll_pot(env, 160.0)
    assert env.marbles_inside < env._cfg.marbles.required_inside
    env._ever_grasped = [True, True]
    assert env._diagnose() is FailureMode.KNOCKED_OVER


def test_required_inside_rounds_up():
    """0.75 of 9 is 7 marbles, not 6."""
    from dream_robot.sims.robosuite.tasks.two_arm_lift.env import MarbleSpec

    spec = MarbleSpec(count=9, radius=0.018, mass=0.02, keep_fraction=0.75)
    assert spec.required_inside == 7


def test_lip_is_shorter_than_the_marble(env):
    """The hard constraint from the derivation: L < r, or nothing ever spills.

    robosuite's pot fails it by 11x, which is why swapping the object was the
    only way to make a tilt cost anything.
    """
    lip = env._cfg.tray.lip_height(env._cfg.marbles.radius)
    assert 0.0 < lip < env._cfg.marbles.radius
