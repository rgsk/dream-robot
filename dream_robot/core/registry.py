"""Name -> constructor. ``"robosuite/pick_place_cube"`` and ``"bc"`` become objects.

**Every import in this file is lazy, inside a factory.** That is not a style
preference, it is the thing that makes ROADMAP's venv split work. Isaac pins its
own Python runtime and will not share an environment with LeRobot and MuJoCo, so
in Isaac's venv the robosuite factory must never be imported, and in a training
venv with no simulator the task factories must never be imported either.
Top-level imports here would couple every backend and every policy into one
dependency set, which is exactly what the dataset-on-disk seam exists to avoid.

Registering a task or a policy is one entry. It is the only place a name in the
results matrix is tied to code.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

# --- tasks ------------------------------------------------------------------


def _robosuite_pick_place_cube(**kwargs):
    from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube

    return PickPlaceCube(**kwargs)


def _robosuite_pick_place_two_bins(**kwargs):
    from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import PickPlaceTwoBins

    return PickPlaceTwoBins(**kwargs)


def _robosuite_pick_place_two_bins_fixed(**kwargs):
    from dataclasses import replace

    from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import (
        PickPlaceTwoBins,
        TaskConfig,
    )

    return PickPlaceTwoBins(replace(TaskConfig.load(), fixed_cube_pose=True), **kwargs)


def _robosuite_two_arm_lift(**kwargs):
    from dream_robot.sims.robosuite.tasks.two_arm_lift.env import TwoArmLiftTask

    return TwoArmLiftTask(**kwargs)


TASKS: dict[str, Callable] = {
    "robosuite/pick_place_cube": _robosuite_pick_place_cube,
    "robosuite/pick_place_two_bins": _robosuite_pick_place_two_bins,
    "robosuite/pick_place_two_bins_fixed": _robosuite_pick_place_two_bins_fixed,
    "robosuite/two_arm_lift": _robosuite_two_arm_lift,
}


def make_env(name: str, **kwargs):
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(TASKS)}")
    return TASKS[name](**kwargs)


# --- policies ---------------------------------------------------------------
#
# Every factory returns something satisfying core.record.Policy -- reset() and
# __call__(Observation) -> action. The scripted expert is registered alongside
# the learned ones on purpose: it is the ceiling row of the matrix, and it is
# only a comparable number if it is measured through the same harness.


def _expert(env, *, checkpoint: Path | None = None, device: str | None = None):
    # Each task ships its own expert, in an ``expert`` module beside its env.
    import importlib

    task_package = type(env).__module__.rsplit(".", 1)[0]
    expert = importlib.import_module(f"{task_package}.expert")
    return expert.ExpertActor(expert.ExpertPolicy(env))


def _bc(env, *, checkpoint: Path | None = None, device: str | None = None):
    from dream_robot.policies.bc.policy import BCPolicy

    if checkpoint is None:
        raise ValueError("policy 'bc' needs --checkpoint")
    return BCPolicy.load(Path(checkpoint), device=device)


def _act(env, *, checkpoint: Path | None = None, device: str | None = None):
    from dream_robot.policies.act.policy import ACTPolicy

    if checkpoint is None:
        raise ValueError("policy 'act' needs --checkpoint")
    return ACTPolicy.load(Path(checkpoint), device=device)


POLICIES: dict[str, Callable] = {
    "expert": _expert,
    "bc": _bc,
    "act": _act,
}


def make_policy(name: str, env, *, checkpoint: Path | None = None, device: str | None = None):
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; registered: {sorted(POLICIES)}")
    return POLICIES[name](env, checkpoint=checkpoint, device=device)
