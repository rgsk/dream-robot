"""Record instructed expert demonstrations of instructed_pick_place into a dataset.

    MUJOCO_GL=egl uv run python -m dream_robot.sims.robosuite.tasks.instructed_pick_place.record \
        --episodes 100 --noise-sigma 0.025 --verify

T2's record.py, minus the coin. Every frame carries its own episode's
instruction (``task_prompt=lambda: env.prompt``), and only the seen
(colour, side) pairs with the recorded template are ever drawn: the held-out
pairs and the paraphrases exist for eval alone.

It writes ``instructions.json`` beside the dataset -- the full draw on every
attempt, distractor colour and slot included -- because the dataset stores
only the words, and "how many kept demos per pair" is part of any T5 result.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from dream_robot.core.dataset import episode_frames, open_dataset
from dream_robot.core.record import JointNoise, record_episodes
from dream_robot.core.video import camera_strip, write_video
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import (
    InstructedPickPlace,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.instructed_pick_place.expert import (
    ExpertActor,
    ExpertPolicy,
)

REPO_ID = "dream_robot/instructed_pick_place"
DEFAULT_ROOT = Path("data/robosuite/instructed_pick_place")
DEFAULT_VIDEO = Path("experiments/t5/videos/dataset_roundtrip.mp4")


class _InstructionLog(ExpertActor):
    """Remembers the env's full draw on every reset."""

    def __init__(self, policy: ExpertPolicy):
        super().__init__(policy)
        self.draws: list[dict] = []

    def reset(self) -> None:
        super().reset()
        self.draws.append(asdict(self._policy._env.instruction))


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
    print(f"verified: episode {episode}, {len(frames)} frames, "
          f"prompt {dataset[0]['task']!r} -> {path.resolve()}")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=20, help="successful episodes to keep")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--noise-sigma", type=float, default=0.0)
    p.add_argument("--camera-size", type=int, default=None,
                   help="square camera frames of this size instead of task.yaml's (128); "
                        "same seeds give the same episodes, only the pictures differ")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--verify-episode", type=int, default=0)
    p.add_argument("--verify-out", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--verify-scale", type=int, default=3)
    args = p.parse_args(argv)

    config = TaskConfig.load()
    if args.camera_size:
        config = replace(config, camera_hw=(args.camera_size, args.camera_size))
    env = InstructedPickPlace(config, pairs="seen", wording="template")
    actor = _InstructionLog(ExpertPolicy(env))
    try:
        summary = record_episodes(
            env,
            actor,
            repo_id=REPO_ID,
            root=args.root,
            task_prompt=lambda: env.prompt,
            episodes=args.episodes,
            robot_type="panda",
            seed_start=args.seed_start,
            max_attempts=args.max_attempts,
            perturb=JointNoise(args.noise_sigma, env.embodiment) if args.noise_sigma > 0 else None,
        )
    finally:
        env.close()

    if len(actor.draws) != len(summary.episodes):
        raise RuntimeError(f"{len(actor.draws)} draws for {len(summary.episodes)} attempts")
    rows = [
        {"seed": e.seed, **d, "success": e.success, "kept": e.kept}
        for e, d in zip(summary.episodes, actor.draws, strict=True)
    ]
    (summary.root / "instructions.json").write_text(
        json.dumps({"attempts": rows}, indent=2) + "\n"
    )
    by_pair = Counter(f"{r['color']}->{r['side']}" for r in rows if r["kept"])

    print(
        f"\n{summary.kept} episodes / {summary.total_frames} frames -> "
        f"{summary.root.resolve()}\n"
        f"expert success rate {summary.success_rate:.0%} over {summary.attempted} attempts\n"
        f"kept episodes by pair: {dict(sorted(by_pair.items()))}"
    )
    if summary.failure_histogram:
        print(f"failures: {summary.failure_histogram}")

    if args.verify:
        verify(args.root, args.verify_out, episode=args.verify_episode, scale=args.verify_scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
