"""The phase machine, tested with no simulator running.

That this file needs no physics is the design property, not a convenience. In
../rl-prac the expert was inseparable from the env, so its logic was only ever
exercised by running the whole simulation -- which meant it was never tested at
all. Here the decision structure is a pure function of numbers.
"""

import numpy as np
import pytest

from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import (
    CLOSED,
    OPEN,
    PHASES,
    ExpertConfig,
    ExpertContext,
    ScriptedExpert,
)

CUBE = np.array([0.0, 0.0, 0.83])
BIN = np.array([0.0, 0.20, 0.80])


@pytest.fixture
def cfg():
    return ExpertConfig.load()


def ctx(cfg, eef, *, opening=OPEN, settled=False):
    return ExpertContext(
        eef_pos=np.asarray(eef, dtype=float),
        cube_pos=CUBE,
        bin_center=BIN,
        gripper_opening=opening,
        grip_settled=settled,
    )


# --- the property that matters ---------------------------------------------

def test_no_phase_has_a_step_budget():
    """../rl-prac's Phase carried `steps`. Ours must not, at any cost.

    A step budget makes episode length constant, which makes the clock
    informative, which is how time features end up in the observation.
    """
    for phase in PHASES:
        assert not hasattr(phase, "steps")
        assert not hasattr(phase, "duration")
    assert set(type(PHASES[0]).__dataclass_fields__) == {
        "name", "waypoint", "gripper", "reached"
    }


def test_phase_length_depends_on_the_world_not_the_clock(cfg):
    """Same phase, two starting distances, two different durations."""
    def steps_to_advance(start):
        expert = ScriptedExpert(cfg)
        eef = np.asarray(start, dtype=float)
        target = CUBE + [0, 0, cfg.approach_height]
        for i in range(1, 500):
            cmd = expert(ctx(cfg, eef))
            if cmd.phase != "approach":
                return i
            eef += 0.01 * (target - eef) / max(np.linalg.norm(target - eef), 1e-9)
        raise AssertionError("never advanced")

    near = steps_to_advance(CUBE + [0, 0, cfg.approach_height + 0.05])
    far = steps_to_advance(CUBE + [0, 0, cfg.approach_height + 0.30])
    assert far > near * 2


# --- phase ordering and gripper commands ------------------------------------

def test_phases_run_in_order_when_every_predicate_is_met(cfg):
    expert = ScriptedExpert(cfg)
    seen = []
    for _ in range(len(PHASES)):
        # An eef that is always exactly on target, and a settled, open gripper:
        # every predicate fires immediately.
        c = ExpertContext(
            eef_pos=np.array([0.0, 0.0, 1e6]),   # satisfies the z-height ones
            cube_pos=CUBE, bin_center=BIN,
            gripper_opening=1.0, grip_settled=True,
        )
        cmd = expert(ExpertContext(**{**c.__dict__, "eef_pos": expert_target(expert, c, cfg)}))
        seen.append(cmd.phase)
    assert seen == [p.name for p in PHASES]


def expert_target(expert, c, cfg):
    """The waypoint the expert is about to ask for, so we can sit exactly on it."""
    idx = min(expert._i, len(PHASES) - 1)
    return np.asarray(PHASES[idx].waypoint(c, cfg), dtype=float)


def test_gripper_commands_are_contract_units():
    """0 closed .. 1 open, never robosuite's -1/+1."""
    assert (OPEN, CLOSED) == (1.0, 0.0)
    for phase in PHASES:
        assert phase.gripper in (OPEN, CLOSED)


def test_gripper_closes_only_from_grasp_until_release():
    closed = [p.name for p in PHASES if p.gripper == CLOSED]
    assert closed == ["grasp", "lift", "carry", "lower"]


# --- the grasp predicate ----------------------------------------------------

def test_grip_settled_needs_a_full_window(cfg):
    expert = ScriptedExpert(cfg)
    for _ in range(cfg.grip_settle_window):
        assert expert.grip_settled(0.5) is False
    assert expert.grip_settled(0.5) is True


def test_grip_still_moving_is_not_settled(cfg):
    expert = ScriptedExpert(cfg)
    opening = 1.0
    for _ in range(cfg.grip_settle_window * 3):
        opening -= 0.01                       # fingers still closing
        assert expert.grip_settled(opening) is False


# --- the failure path -------------------------------------------------------

def test_phase_times_out_instead_of_hanging(cfg):
    expert = ScriptedExpert(cfg, control_hz=30.0)
    stuck = ctx(cfg, CUBE + [0, 0, 10.0])     # never reaches anything
    timeout_steps = int(round(cfg.phase_timeout_seconds * 30.0))
    for _ in range(timeout_steps - 1):
        assert expert(stuck).timed_out is False
    assert expert(stuck).timed_out is True


def test_timeout_does_not_advance_the_phase(cfg):
    expert = ScriptedExpert(cfg, control_hz=30.0)
    stuck = ctx(cfg, CUBE + [0, 0, 10.0])
    for _ in range(int(round(cfg.phase_timeout_seconds * 30.0)) + 5):
        expert(stuck)
    assert expert.phase == "approach"


def test_reset_returns_to_the_first_phase(cfg):
    expert = ScriptedExpert(cfg)
    on_target = ctx(cfg, CUBE + [0, 0, cfg.approach_height])
    expert(on_target)
    assert expert.phase == "descend"
    expert.reset()
    assert expert.phase == "approach"


# --- config -----------------------------------------------------------------

def test_geometry_comes_from_task_yaml(cfg):
    """No waypoint literals in expert.py."""
    assert cfg.approach_height == 0.10
    assert cfg.drop_height == 0.085
    assert cfg.ik.max_joint_step == 0.06
