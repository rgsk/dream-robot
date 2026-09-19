"""The two-arm phase machine and its barrier, tested with no simulator running.

Same property T1's expert test exists for -- the decision structure is a pure
function of numbers -- and one addition that is specific to this task: the
barrier is the thing that decides whether the recorded demonstrations teach
coordination or teach the failure the task exists to measure, so it is tested
directly rather than inferred from a success rate.
"""

from dataclasses import replace

import numpy as np
import pytest

from dream_robot.sims.robosuite.tasks.two_arm_lift.expert import (
    CLOSED,
    OPEN,
    PHASES,
    ArmContext,
    ExpertConfig,
    TwoArmExpert,
    grasp_frame,
)

CENTRE = np.array([0.0, 0.0, 0.81])
HANDLES = (np.array([0.0, -0.15, 0.87]), np.array([0.0, 0.15, 0.87]))
FAR = np.array([0.5, 0.5, 1.5])


@pytest.fixture
def cfg():
    return ExpertConfig.load()


def ctxs(machine, cfg, *, on_target=(True, True), settled=(True, True)):
    """A context per arm, each either exactly on its phase's waypoint or far off."""
    out = []
    for arm in range(2):
        i = machine._i[arm]
        phase = PHASES[min(i, len(PHASES) - 1)]
        probe = ArmContext(FAR, HANDLES[arm], OPEN, settled[arm])
        target = np.asarray(phase.waypoint(probe, cfg), dtype=float)
        out.append(
            ArmContext(
                eef_pos=target if on_target[arm] else FAR,
                handle_pos=HANDLES[arm],
                gripper_opening=OPEN,
                grip_settled=settled[arm],
            )
        )
    return tuple(out)


# --- the barrier ------------------------------------------------------------

def test_neither_arm_lifts_until_both_grippers_have_settled(cfg):
    """The one sentence this whole task rests on, as a test.

    Drive both arms to `grasp`, then settle only arm 0's gripper. Arm 0 must
    stay in `grasp` holding its waypoint: an arm that leaves early lifts the
    tray by one handle and tips it, which is exactly the demonstration we must
    never record.
    """
    machine = TwoArmExpert(cfg)
    while machine.phases != ("grasp", "grasp"):
        machine(ctxs(machine, cfg))

    for _ in range(20):
        command = machine(ctxs(machine, cfg, settled=(True, False)))
        assert machine.phases == ("grasp", "grasp")
        assert [a.gripper for a in command.arms] == [CLOSED, CLOSED]
        assert command.arms[0].waiting and not command.arms[1].waiting

    machine(ctxs(machine, cfg, settled=(True, True)))
    assert machine.phases == ("lift", "lift")


def test_the_arms_stay_in_lockstep_whatever_order_they_arrive_in(cfg):
    """Lockstep is derived from the barrier, not assumed: assert it every step."""
    rng = np.random.default_rng(0)
    machine = TwoArmExpert(cfg)
    for _ in range(400):
        arrived = tuple(bool(x) for x in rng.integers(0, 2, size=2))
        machine(ctxs(machine, cfg, on_target=arrived, settled=arrived))
        assert machine._i[0] == machine._i[1], machine.phases


def test_an_arm_held_at_the_barrier_is_not_an_arm_that_is_stuck(cfg):
    """Waiting must not read as failing, or a slow partner blames the fast one."""
    machine = TwoArmExpert(cfg)
    steps = int(cfg.phase_timeout_seconds * 30) + 10
    for _ in range(steps):
        command = machine(ctxs(machine, cfg, on_target=(True, False)))
    assert command.arms[0].waiting and not command.arms[0].timed_out
    assert command.arms[1].timed_out          # this one really cannot get there
    assert command.timed_out                  # and the episode gives up


def test_without_the_barrier_one_arm_lifts_while_the_other_descends(cfg):
    """The ablation has to be reachable, or `sync_barrier` is decoration.

    This is the failure mode the task measures (`DESYNCHRONISED`), reproduced in
    the expert on purpose so a second dataset can be recorded from it.
    """
    machine = TwoArmExpert(replace(cfg, sync_barrier=False))
    for _ in range(3):
        machine(ctxs(machine, cfg, on_target=(True, False), settled=(True, False)))
    # Three steps, three phases: arm 0 has grasped and is lifting while arm 1
    # has not yet finished its approach.
    assert machine.phases == ("lift", "approach")


def test_settle_windows_are_per_arm(cfg):
    """A shared window would let one arm's fingers vouch for the other's."""
    machine = TwoArmExpert(cfg)
    for _ in range(cfg.grip_settle_window + 2):
        machine.grip_settled(0, 0.2)
    assert machine.grip_settled(0, 0.2)
    assert not machine.grip_settled(1, 0.2)


# --- the property inherited from T1 -----------------------------------------

def test_no_phase_has_a_step_budget():
    """../rl-prac's phases carried `steps`; ours must not, at any cost."""
    for phase in PHASES:
        assert not hasattr(phase, "steps") and not hasattr(phase, "duration")
    assert set(type(PHASES[0]).__dataclass_fields__) == {
        "name", "waypoint", "gripper", "reached"
    }


# --- the grasp frame --------------------------------------------------------

def test_grasp_frame_closes_the_fingers_across_the_bar():
    """x̂ is the handle's outward direction (⊥ the bar), ẑ points down, and the
    frame is a right-handed rotation."""
    for handle in HANDLES:
        R = grasp_frame(handle, CENTRE)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.allclose(R[:, 2], [0.0, 0.0, -1.0])
        outward = handle - CENTRE
        outward = outward[:2] / np.linalg.norm(outward[:2])
        assert abs(float(R[:2, 0] @ outward)) == pytest.approx(1.0, abs=1e-9)


def test_grasp_frame_follows_the_trays_yaw():
    """A yawed tray must yaw the wrist: the two frames differ by that angle."""
    for yaw in (0.0, np.pi / 6, -np.pi / 3):
        rot = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                        [np.sin(yaw), np.cos(yaw), 0.0],
                        [0.0, 0.0, 1.0]])
        handle = CENTRE + rot @ (HANDLES[0] - CENTRE)
        R = grasp_frame(handle, CENTRE)
        base = grasp_frame(HANDLES[0], CENTRE)
        measured = np.arctan2(R[1, 0], R[0, 0]) - np.arctan2(base[1, 0], base[0, 0])
        assert np.isclose(np.cos(measured), np.cos(yaw), atol=1e-9)


def test_grasp_frame_takes_the_shorter_way_round():
    """Both ±d̂ grasp the same bar; the wrist should not travel 180 deg to pick."""
    handle = HANDLES[0]
    toward = grasp_frame(handle, CENTRE, current_x=np.array([0.0, -1.0, 0.0]))
    away = grasp_frame(handle, CENTRE, current_x=np.array([0.0, 1.0, 0.0]))
    assert np.allclose(toward[:, 0], [0.0, -1.0, 0.0])
    assert np.allclose(away[:, 0], [0.0, 1.0, 0.0])
    assert np.linalg.det(away) == pytest.approx(1.0)
