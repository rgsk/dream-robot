"""The permutation entrypoint: any policy x any task, one command.

    MUJOCO_GL=glfw uv run python -m dream_robot.core.run \
        --env robosuite/pick_place_cube --policy bc \
        --checkpoint experiments/bc_pick_place_cube/checkpoint.pt

    MUJOCO_GL=glfw uv run python -m dream_robot.core.run \
        --env robosuite/pick_place_cube --policy expert

Writes ``experiments/<exp_id>/results.json`` and a video, every time (ROADMAP
rule 5). The point of routing both the learned policy and the scripted expert
through this one script is that the expert's row of the matrix is produced by
the same harness as every other row -- otherwise the ceiling is measured
differently from the things being compared against it, and the comparison is
decorative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dream_robot.core.eval import EVAL_SEED_START, evaluate, write_results
from dream_robot.core.registry import make_env, make_policy

DEFAULT_EXPERIMENTS = Path("experiments")


def experiment_id(env_name: str, policy_name: str) -> str:
    """``robosuite/pick_place_cube`` + ``bc`` -> ``bc_on_robosuite_pick_place_cube``.

    Deterministic, so re-running a cell overwrites its own results rather than
    accumulating timestamped directories that have to be reconciled by hand.
    """
    return f"{policy_name}_on_{env_name.replace('/', '_')}"


def checkpoint_provenance(path: Path | None) -> dict:
    """What produced the weights, lifted out of the checkpoint itself."""
    if path is None:
        return {}
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    dataset = blob.get("dataset", {})
    return {
        "checkpoint": str(path),
        "epoch": blob.get("epoch"),
        "val_l1_rad": blob.get("val_l1_rad"),
        "dataset_repo_id": dataset.get("repo_id"),
        "demonstrations": len(dataset.get("train_episodes", [])),
        "held_out_episodes": dataset.get("val_episodes"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", required=True, help="e.g. robosuite/pick_place_cube")
    p.add_argument("--policy", required=True, help="e.g. bc, expert")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    p.add_argument("--video-episodes", type=int, default=3)
    p.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    out = args.experiments / experiment_id(args.env, args.policy)
    env = make_env(args.env)
    try:
        policy = make_policy(
            args.policy, env, checkpoint=args.checkpoint, device=args.device
        )
        print(f"{args.policy} on {args.env}, {args.episodes} episodes "
              f"from seed {args.seed_start}\n")
        result = evaluate(
            env,
            policy,
            task=args.env,
            policy_name=args.policy,
            episodes=args.episodes,
            seed_start=args.seed_start,
            video_episodes=args.video_episodes,
            video_path=out / "videos" / "rollouts.mp4",
        )
    finally:
        env.close()

    path = write_results(
        result, out / "results.json", extra=checkpoint_provenance(args.checkpoint)
    )
    print(f"\n{result.summary()}")
    print(f"results -> {path.resolve()}")
    if result.video:
        print(f"video   -> {result.video.resolve()}")
    return 0


if __name__ == "__main__":
    # LeRobot's video encoder can spawn a process pool whose forkserver
    # re-imports __main__; without this guard an evaluation would restart itself.
    raise SystemExit(main())
