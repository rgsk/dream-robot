"""Record scripted-expert demonstrations of pick_place_cube into a dataset.

    MUJOCO_GL=glfw uv run python -m dream_robot.sims.robosuite.tasks.pick_place_cube.record \
        --episodes 20 --verify

Thin on purpose, like ``demo.py``: it builds this task's env and expert, hands
them to ``core.record``, and prints what came back. Everything that is not
specific to robosuite lives in core, so the Isaac version of this file will be
the same twenty lines with two different imports.

``--verify`` is the part worth running. It reopens the dataset that was just
written, decodes episode 0 out of it, and renders a video **from the bytes on
disk** rather than from anything still in memory. A recorder that wrote the
wrong column, dropped the flip, or paired observations with the next step's
action produces a file that is silently wrong and a video that is obviously
wrong. The seam is only real if you have looked through it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dream_robot.core.dataset import episode_frames, open_dataset
from dream_robot.core.record import JointNoise, record_episodes
from dream_robot.core.video import camera_strip, write_video
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube, TaskConfig
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy

REPO_ID = "dream_robot/pick_place_cube"
DEFAULT_ROOT = Path("data/robosuite/pick_place_cube")
DEFAULT_VIDEO = Path("experiments/expert_demo/videos/dataset_roundtrip.mp4")


def verify(root: Path, out: Path, *, episode: int, scale: int) -> Path:
    """Decode one episode back out of the dataset and write it as a video."""
    dataset = open_dataset(REPO_ID, root, episode=episode)
    cameras = [key.rsplit(".", 1)[-1] for key in dataset.meta.camera_keys]
    per_camera = {name: episode_frames(dataset, name) for name in cameras}
    frames = [
        camera_strip({name: per_camera[name][t] for name in cameras}, scale=scale)
        for t in range(len(dataset))
    ]
    path = write_video(frames, out, fps=dataset.meta.fps)
    print(
        f"verified: episode {episode}, {len(frames)} frames, cameras {cameras} "
        f"-> {path.resolve()}"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=20, help="successful episodes to keep")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument(
        "--keep-failures",
        action="store_true",
        help="record failed episodes too; poisons behaviour cloning, see core/record.py",
    )
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=0.0,
        help="Gaussian noise (rad) on executed arm joints; the dataset keeps the clean action",
    )
    p.add_argument("--verify", action="store_true", help="write a video read back from disk")
    p.add_argument("--verify-episode", type=int, default=0)
    p.add_argument("--verify-out", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--verify-scale", type=int, default=3)
    args = p.parse_args(argv)

    cfg = TaskConfig.load()
    env = PickPlaceCube(cfg)
    try:
        summary = record_episodes(
            env,
            ExpertActor(ExpertPolicy(env)),
            repo_id=REPO_ID,
            root=args.root,
            task_prompt=cfg.prompt,
            episodes=args.episodes,
            robot_type="panda",
            seed_start=args.seed_start,
            max_attempts=args.max_attempts,
            keep_failures=args.keep_failures,
            perturb=JointNoise(args.noise_sigma, env.embodiment) if args.noise_sigma > 0 else None,
        )
    finally:
        env.close()

    print(
        f"\n{summary.kept} episodes / {summary.total_frames} frames -> "
        f"{summary.root.resolve()}\n"
        f"expert success rate {summary.success_rate:.0%} over {summary.attempted} attempts"
    )
    if summary.failure_histogram:
        print(f"failures: {summary.failure_histogram}")

    if args.verify:
        verify(
            args.root,
            args.verify_out,
            episode=args.verify_episode,
            scale=args.verify_scale,
        )
    return 0


if __name__ == "__main__":
    # Not decoration. LeRobot's video encoder can spawn a process pool, and the
    # forkserver re-imports this module as __main__ in each child -- without the
    # guard the recording restarts itself recursively. core/record.py keeps
    # encoding serial so this cannot bite, and the guard stays as the second lock.
    raise SystemExit(main())
