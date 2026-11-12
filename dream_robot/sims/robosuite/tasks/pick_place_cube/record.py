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
from dataclasses import replace
from pathlib import Path

from dream_robot.core.dataset import episode_frames, open_dataset
from dream_robot.core.record import JointNoise, record_episodes
from dream_robot.core.video import camera_strip, write_video
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube, TaskConfig
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy


class AimBias:
    """Executes the expert's biased action (ExpertPolicy.executed) instead of the recorded one, plus
    optional joint noise on top. The dataset keeps the clean action: see ExpertPolicy's aim_bias_deg."""

    def __init__(self, policy: ExpertPolicy, degrees: float, noise: JointNoise | None):
        self.policy, self.degrees, self.noise = policy, degrees, noise

    def __call__(self, action, rng):
        executed = self.policy.executed if self.policy.executed is not None else action
        return self.noise(executed, rng) if self.noise else executed

    def to_dict(self) -> dict:
        return {'type': 'aim_bias', 'degrees': self.degrees} | (
            {'noise': self.noise.to_dict()} if self.noise else {}
        )


REPO_ID = 'dream_robot/pick_place_cube'
DEFAULT_ROOT = Path('data/robosuite/pick_place_cube')
DEFAULT_VIDEO = Path('experiments/expert_demo/videos/dataset_roundtrip.mp4')


def verify(root: Path, out: Path, *, episode: int, scale: int) -> Path:
    """Decode one episode back out of the dataset and write it as a video."""
    dataset = open_dataset(REPO_ID, root, episode=episode)
    cameras = [key.rsplit('.', 1)[-1] for key in dataset.meta.camera_keys]
    per_camera = {name: episode_frames(dataset, name) for name in cameras}
    frames = [
        camera_strip({name: per_camera[name][t] for name in cameras}, scale=scale)
        for t in range(len(dataset))
    ]
    path = write_video(frames, out, fps=dataset.meta.fps)
    print(
        f'verified: episode {episode}, {len(frames)} frames, cameras {cameras} -> {path.resolve()}'
    )
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--episodes', type=int, default=20, help='successful episodes to keep')
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--max-attempts', type=int, default=None)
    p.add_argument(
        '--keep-failures',
        action='store_true',
        help='record failed episodes too; poisons behaviour cloning, see core/record.py',
    )
    p.add_argument(
        '--noise-sigma',
        type=float,
        default=0.0,
        help='Gaussian noise (rad) on executed arm joints; the dataset keeps the clean action',
    )
    p.add_argument(
        '--cube-x',
        type=float,
        nargs=2,
        default=None,
        metavar=('MIN', 'MAX'),
        help='cube spawn x range in metres about the table centre (default: task.yaml)',
    )
    p.add_argument(
        '--cube-y',
        type=float,
        nargs=2,
        default=None,
        metavar=('MIN', 'MAX'),
        help='cube spawn y range in metres about the table centre (default: task.yaml)',
    )
    p.add_argument(
        '--square-up',
        action='store_true',
        help='the expert turns the gripper to close square to the cube (default: fixed orientation)',
    )
    p.add_argument(
        '--aim-bias-deg',
        type=float,
        default=0.0,
        help='with --square-up: execute a per-episode wrist bias in [-deg, deg] until the lift, '
        'record the clean action (an imperfect grasp that still carries on)',
    )
    p.add_argument('--verify', action='store_true', help='write a video read back from disk')
    p.add_argument('--verify-episode', type=int, default=0)
    p.add_argument('--verify-out', type=Path, default=DEFAULT_VIDEO)
    p.add_argument('--verify-scale', type=int, default=3)
    args = p.parse_args(argv)

    cfg = TaskConfig.load()
    if args.cube_x or args.cube_y:
        x, y = cfg.cube_xy_range
        cfg = replace(
            cfg,
            cube_xy_range=(
                tuple(args.cube_x) if args.cube_x else x,
                tuple(args.cube_y) if args.cube_y else y,
            ),
        )
    print(f'cube spawn range (x, y): {cfg.cube_xy_range}')
    env = PickPlaceCube(cfg)
    expert = ExpertPolicy(env, square_up=args.square_up, aim_bias_deg=args.aim_bias_deg)
    noise = JointNoise(args.noise_sigma, env.embodiment) if args.noise_sigma > 0 else None
    perturb = AimBias(expert, args.aim_bias_deg, noise) if args.aim_bias_deg else noise
    try:
        summary = record_episodes(
            env,
            ExpertActor(expert),
            repo_id=REPO_ID,
            root=args.root,
            task_prompt=cfg.prompt,
            episodes=args.episodes,
            robot_type='panda',
            seed_start=args.seed_start,
            max_attempts=args.max_attempts,
            keep_failures=args.keep_failures,
            perturb=perturb,
        )
    finally:
        env.close()

    print(
        f'\n{summary.kept} episodes / {summary.total_frames} frames -> '
        f'{summary.root.resolve()}\n'
        f'expert success rate {summary.success_rate:.0%} over {summary.attempted} attempts'
    )
    if summary.failure_histogram:
        print(f'failures: {summary.failure_histogram}')

    if args.verify:
        verify(
            args.root,
            args.verify_out,
            episode=args.verify_episode,
            scale=args.verify_scale,
        )
    return 0


if __name__ == '__main__':
    # Not decoration. LeRobot's video encoder can spawn a process pool, and the
    # forkserver re-imports this module as __main__ in each child -- without the
    # guard the recording restarts itself recursively. core/record.py keeps
    # encoding serial so this cannot bite, and the guard stays as the second lock.
    raise SystemExit(main())
