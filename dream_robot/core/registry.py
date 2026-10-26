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


def _robosuite_instructed_pick_place(**kwargs):
    from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import InstructedPickPlace

    return InstructedPickPlace(**kwargs)


def _robosuite_assorted_pick_place(**kwargs):
    from dream_robot.sims.robosuite.tasks.assorted_pick_place.env import AssortedPickPlace

    return AssortedPickPlace(**kwargs)


def _robosuite_two_arm_lift(**kwargs):
    from dream_robot.sims.robosuite.tasks.two_arm_lift.env import TwoArmLiftTask

    return TwoArmLiftTask(**kwargs)


TASKS: dict[str, Callable] = {
    "robosuite/pick_place_cube": _robosuite_pick_place_cube,
    "robosuite/pick_place_two_bins": _robosuite_pick_place_two_bins,
    "robosuite/pick_place_two_bins_fixed": _robosuite_pick_place_two_bins_fixed,
    "robosuite/two_arm_lift": _robosuite_two_arm_lift,
    # T5. One entry; the eval splits are constructor arguments, not tasks:
    # pairs="held_out" and/or wording="paraphrase".
    "robosuite/instructed_pick_place": _robosuite_instructed_pick_place,
    # T7. Assorted objects, one named; the agent stack's generalization test.
    "robosuite/assorted_pick_place": _robosuite_assorted_pick_place,
}


def task_factory(name: str) -> Callable:
    """The constructor for ``name``. ``robocasa/<Task>`` names any RoboCasa task, from
    its own venv (envs/robocasa) -- there are hundreds, so they are not listed."""
    if name in TASKS:
        return TASKS[name]
    if name.startswith("robocasa/"):
        task = name.split("/", 1)[1]

        def _robocasa(**kwargs):
            from dream_robot.sims.robocasa.kitchen import Kitchen

            return Kitchen(task, **kwargs)

        return _robocasa
    raise KeyError(f"unknown task {name!r}; registered: {sorted(TASKS)} or robocasa/<Task>")


def make_env(name: str, **kwargs):
    return task_factory(name)(**kwargs)


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


def _flow_matching(env, *, checkpoint: Path | None = None, device: str | None = None):
    from dream_robot.policies.flow_matching.policy import FlowPolicy

    if checkpoint is None:
        raise ValueError("policy 'flow_matching' needs --checkpoint")
    # env.prompt is read only by a conditioned (T5) checkpoint, after each reset.
    return FlowPolicy.load(Path(checkpoint), device=device, prompt=lambda: env.prompt)


def _smolvla(env, *, checkpoint: Path | None = None, device: str | None = None):
    # A fine-tuned pretrained_model/ dir. Zero-shot on the base checkpoint needs
    # a dataset for shapes and stats, so it goes through SmolVLAPolicy.load directly.
    from dream_robot.policies.smolvla.policy import SmolVLAPolicy

    if checkpoint is None:
        raise ValueError("policy 'smolvla' needs --checkpoint (a pretrained_model/ dir)")
    return SmolVLAPolicy.load(env, checkpoint=Path(checkpoint), device=device)


def _oracle_idm(env, *, checkpoint: Path | None = None, device: str | None = None,
                execute: int | None = None, horizon: int | None | str = "k"):
    # The IDM fed the expert's future, forked from the live sim (ROADMAP "Next:
    # our own executor", step 1). Privileged: robosuite tasks with an expert only.
    import importlib

    import torch

    from dream_robot.policies.idm.policy import IDMPolicy
    from dream_robot.sims.robosuite.oracle import ExpertFuture

    if checkpoint is None:
        raise ValueError("policy 'oracle_idm' needs --checkpoint")
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net, meta = IDMPolicy.load_net(Path(checkpoint), resolved)
    task_package = type(env).__module__.rsplit(".", 1)[0]
    expert = importlib.import_module(f"{task_package}.expert")
    # horizon "k": one frame per imagination (the first runs); None: the whole
    # remaining episode as one clip.
    k = net.config.k
    future = ExpertFuture(env, expert.ExpertPolicy(env), spacing=k,
                          horizon=k if horizon == "k" else horizon)
    return IDMPolicy(net, future, device=resolved, execute=execute, metadata=meta)


def _cosmos_policy(env, *, checkpoint: Path | str | None = None, device: str | None = None,
                   execute: int | None = None):
    # The model runs in its own env as an action server (experiments/cosmos_policy/
    # scripts/serve.py), which loaded the checkpoint; ``checkpoint`` here is its address.
    from dream_robot.policies.cosmos_policy.policy import CosmosPolicy

    return CosmosPolicy(env, address=str(checkpoint or "localhost:6000"), execute=execute)


def _uwm(env, *, checkpoint: Path | str | None = None, device: str | None = None,
         execute: int | None = None):
    # Like cosmos_policy: the model runs in its own env as an action server
    # (experiments/uwm/scripts/serve.py); ``checkpoint`` here is its address.
    from dream_robot.policies.uwm.policy import UWMPolicy

    return UWMPolicy(env, address=str(checkpoint or "localhost:6001"), execute=execute)


def _uva(env, *, checkpoint: Path | str | None = None, device: str | None = None,
         execute: int | None = None):
    # Like uwm: the model runs in its own env as an action server
    # (experiments/uva/scripts/serve.py); ``checkpoint`` here is its address.
    from dream_robot.policies.uva.policy import UVAPolicy

    return UVAPolicy(env, address=str(checkpoint or "localhost:6003"), execute=execute)


def _vpp2(env, *, checkpoint: Path | str | None = None, device: str | None = None,
          execute: int | None = None):
    # VPP2 speaks the Cosmos server protocol (experiments/vpp/scripts/serve.py), so the
    # Cosmos client plays its 32-action chunks; ``checkpoint`` here is its address.
    from dream_robot.policies.cosmos_policy.policy import CosmosPolicy

    return CosmosPolicy(env, address=str(checkpoint or "localhost:6002"), execute=execute)


POLICIES: dict[str, Callable] = {
    "expert": _expert,
    "bc": _bc,
    "act": _act,
    "flow_matching": _flow_matching,
    "smolvla": _smolvla,
    "oracle_idm": _oracle_idm,
    "cosmos_policy": _cosmos_policy,
    "uwm": _uwm,
    "uva": _uva,
    "vpp2": _vpp2,
}


def make_policy(name: str, env, *, checkpoint: Path | None = None, device: str | None = None):
    if name not in POLICIES:
        raise KeyError(f"unknown policy {name!r}; registered: {sorted(POLICIES)}")
    return POLICIES[name](env, checkpoint=checkpoint, device=device)
