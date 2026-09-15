"""Record coin-flip expert demonstrations of pick_place_two_bins into a dataset.

    MUJOCO_GL=glfw uv run python -m dream_robot.sims.robosuite.tasks.pick_place_two_bins.record \
        --episodes 100 --noise-sigma 0.025 --verify

    # fixed cube pose, exactly half the kept demos into each bin
    ... record --episodes 100 --noise-sigma 0.025 --fixed-cube --coin balanced

T1's record.py with additions. It writes ``bin_choices.json`` beside the
dataset -- which bin the expert picked on every attempt -- because the dataset
itself must not contain the choice, and the mode balance of the demonstrations
is part of any T2 result.

**``--coin balanced``** makes the kept demos split exactly evenly. A fair coin
cannot promise that, and failed attempts are discarded, so balancing attempts
would not either. Instead each episode goes to whichever bin has fewer *kept*
demos so far (a fair coin on ties), which needs to know whether the previous
attempt was kept: ``_SuccessTap`` watches the env for that. With an even
episode count the kept split is then exactly half and half.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from dream_robot.core.dataset import episode_frames, open_dataset
from dream_robot.core.record import JointNoise, record_episodes
from dream_robot.core.video import camera_strip, write_video
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import (
    PickPlaceTwoBins,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.expert import (
    ExpertActor,
    ExpertPolicy,
)

REPO_ID = "dream_robot/pick_place_two_bins"
DEFAULT_ROOT = Path("data/robosuite/pick_place_two_bins")
DEFAULT_VIDEO = Path("experiments/t2/videos/dataset_roundtrip.mp4")


class _SuccessTap:
    """The env, passed through, remembering whether the previous episode succeeded.

    The recorder keeps an episode exactly when it succeeds (keep_failures is
    off), so "the previous episode succeeded" is "the previous episode was kept".
    """

    def __init__(self, env):
        self._env = env
        self._current = False
        self.previous_succeeded = False

    def reset(self, *, seed=None):
        self.previous_succeeded, self._current = self._current, False
        return self._env.reset(seed=seed)

    def step(self, action):
        result = self._env.step(action)
        self._current = self._current or result.success
        return result

    def __getattr__(self, name):
        return getattr(self._env, name)


class _ChoiceLog(ExpertActor):
    """Remembers the coin on every reset; optionally overrides it to balance kept demos."""

    def __init__(self, policy: ExpertPolicy, *, tap: _SuccessTap | None = None):
        super().__init__(policy)
        self._tap = tap
        self.kept = [0] * len(policy._env.bin_centers)
        self.choices: list[int] = []

    def reset(self) -> None:
        super().reset()
        if self._tap is not None:
            if self.choices and self._tap.previous_succeeded:
                self.kept[self.choices[-1]] += 1
            fewest = min(self.kept)
            options = [i for i, k in enumerate(self.kept) if k == fewest]
            self._policy.target_bin = options[int(np.random.randint(len(options)))]
        self.choices.append(self._policy.target_bin)


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
    print(f"verified: episode {episode}, {len(frames)} frames -> {path.resolve()}")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=20, help="successful episodes to keep")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--noise-sigma", type=float, default=0.0)
    p.add_argument("--fixed-cube", action="store_true", help="cube at table centre, zero yaw")
    p.add_argument("--coin", choices=("fair", "balanced"), default="fair")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify-episode", type=int, default=0)
    p.add_argument("--verify-out", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--verify-scale", type=int, default=3)
    args = p.parse_args(argv)

    cfg = TaskConfig.load()
    if args.fixed_cube:
        cfg = replace(cfg, fixed_cube_pose=True)
    env = PickPlaceTwoBins(cfg)
    tap = _SuccessTap(env) if args.coin == "balanced" else None
    actor = _ChoiceLog(ExpertPolicy(env), tap=tap)
    try:
        summary = record_episodes(
            tap if tap is not None else env,
            actor,
            repo_id=REPO_ID,
            root=args.root,
            task_prompt=cfg.prompt,
            episodes=args.episodes,
            robot_type="panda",
            seed_start=args.seed_start,
            max_attempts=args.max_attempts,
            perturb=JointNoise(args.noise_sigma, env.embodiment) if args.noise_sigma > 0 else None,
        )
        names = env.bin_names
    finally:
        env.close()

    if len(actor.choices) != len(summary.episodes):
        raise RuntimeError(
            f"{len(actor.choices)} coin flips for {len(summary.episodes)} attempts"
        )
    rows = [
        {"seed": e.seed, "target_bin": names[c], "success": e.success, "kept": e.kept}
        for e, c in zip(summary.episodes, actor.choices)
    ]
    (summary.root / "bin_choices.json").write_text(
        json.dumps(
            {"coin": args.coin, "fixed_cube_pose": cfg.fixed_cube_pose, "attempts": rows},
            indent=2,
        )
        + "\n"
    )
    balance = Counter(r["target_bin"] for r in rows if r["kept"])

    print(
        f"\n{summary.kept} episodes / {summary.total_frames} frames -> "
        f"{summary.root.resolve()}\n"
        f"expert success rate {summary.success_rate:.0%} over {summary.attempted} attempts\n"
        f"coin {args.coin}, fixed cube pose {cfg.fixed_cube_pose}; "
        f"kept episodes by bin: {dict(balance)}"
    )
    if summary.failure_histogram:
        print(f"failures: {summary.failure_histogram}")

    if args.verify:
        verify(args.root, args.verify_out, episode=args.verify_episode, scale=args.verify_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
