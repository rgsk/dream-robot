"""SmolVLA base checkpoint on T5, no fine-tuning. A plumbing test, not a result.

    MUJOCO_GL=egl uv run python experiments/t5/scripts/zero_shot.py --episodes 5

Expected success ~0%: smolvla_base was pretrained on SO-100 arms, and nothing in
it maps to a Panda's joint space. What this checks is that the whole path runs:
our Observation -> LeRobot batch (state, top, wrist, the episode's instruction)
-> VLM + action expert -> 8 Panda joints the env accepts, through the same eval
harness every other policy uses. And, from the video, what an untuned VLA's
motion looks like on our arm.

Shapes and normalisation stats come from the 6-episode smoke dataset: enough to
build the model, and irrelevant to a 0% number.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dream_robot.core.eval import EVAL_SEED_START, evaluate, write_results
from dream_robot.policies.smolvla.policy import BASE_CHECKPOINT, SmolVLAPolicy
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import InstructedPickPlace
from dream_robot.sims.robosuite.tasks.instructed_pick_place.record import REPO_ID

OUT = Path("experiments/t5/zero_shot")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--dataset-root", type=Path, default=Path("scratch/t5/smoke"))
    p.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    args = p.parse_args()

    env = InstructedPickPlace(pairs="seen", wording="template")
    try:
        t0 = time.time()
        policy = SmolVLAPolicy.load(
            env, checkpoint=BASE_CHECKPOINT, dataset_root=args.dataset_root, repo_id=REPO_ID
        )
        params = sum(p.numel() for p in policy._policy.parameters())
        print(f"loaded {BASE_CHECKPOINT} in {time.time() - t0:.0f}s: {params / 1e6:.0f}M params, "
              f"cameras {policy.cameras}, device {policy._device}")
        if torch.cuda.is_available():
            print(f"GPU memory after load: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

        t0 = time.time()
        result = evaluate(
            env, policy,
            task="robosuite/instructed_pick_place", policy_name="smolvla_base_zero_shot",
            episodes=args.episodes, seed_start=args.seed_start,
            video_dir=OUT / "videos", video_per_outcome=2,
        )
        wall = time.time() - t0
    finally:
        env.close()

    steps = sum(e.steps for e in result.episodes)
    path = write_results(
        result, OUT / "results.json",
        extra={"checkpoint": BASE_CHECKPOINT, "stats_from": str(args.dataset_root),
               "wall_seconds": round(wall), "steps_per_second": round(steps / wall, 1)},
    )
    print(f"\n{result.summary()}\n{steps} steps in {wall:.0f}s ({steps / wall:.1f} steps/s)")
    print(f"results -> {path}")
    for name, video in result.videos.items():
        print(f"{name:9s} -> {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
