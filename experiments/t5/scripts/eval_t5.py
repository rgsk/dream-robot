"""Evaluate a policy on T5 and say *which half* of the instruction each failure got wrong.

    MUJOCO_GL=egl uv run python experiments/t5/scripts/eval_t5.py \
        --checkpoint experiments/t5/smolvla_clean_n150/checkpoints/020000/pretrained_model \
        --pairs seen --wording template --episodes 60 --out experiments/t5/eval_20k/seen

``core.run`` reports the closed FailureMode histogram, where both halves of T5's
failure are one bucket, WRONG_TARGET. This script drives the same
``core.eval.evaluate`` (same seeds, same videos) through a thin tap on the env
that, at every episode end, records the instruction and
``env.instruction_failure()``:

- wrong_cube -- the distractor was lifted and the named cube never was: the
  colour word was not grounded;
- wrong_bin  -- the named cube is resting in the other bin: the side word was not.

It also reports success by target slot, because the eval seeds need not match
the demos' slot balance (seeds 1000-1019 put the target in slot 0 17 times; the
150 demos had it there 65 times), and a policy that learnt "grab slot 1" would
otherwise look like a colour-reading one, or vice versa.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from dream_robot.core.eval import EVAL_SEED_START, evaluate, write_results
from dream_robot.policies.smolvla.policy import SmolVLAPolicy
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import InstructedPickPlace


class _InstructionTap:
    """The env, passed through, writing one row per finished episode."""

    def __init__(self, env: InstructedPickPlace):
        self._env = env
        self.rows: list[dict] = []
        self._seed: int | None = None

    def reset(self, *, seed=None):
        self._seed = seed
        return self._env.reset(seed=seed)

    def step(self, action):
        result = self._env.step(action)
        if result.terminated or result.truncated:
            self.rows.append({
                "seed": self._seed,
                "prompt": self._env.prompt,
                **{k: v for k, v in asdict(self._env.instruction).items() if k != "prompt"},
                "success": result.success,
                "failure_mode": str(result.failure_mode),
                "instruction_failure": None if result.success else self._env.instruction_failure(),
            })
        return result

    def __getattr__(self, name):
        return getattr(self._env, name)


def breakdown(rows: list[dict]) -> dict:
    def rate(group):
        return {"success": sum(r["success"] for r in group), "episodes": len(group)}

    by = defaultdict(list)
    for r in rows:
        by[f"slot {r['target_slot']}"].append(r)
        by[f"{r['color']}->{r['side']}"].append(r)
    return {
        "success": rate(rows),
        "instruction_failures": dict(Counter(
            r["instruction_failure"] or "other" for r in rows if not r["success"]
        )),
        "by": {k: rate(v) for k, v in sorted(by.items())},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pairs", choices=("seen", "held_out"), default="seen")
    p.add_argument("--wording", choices=("template", "paraphrase"), default="template")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video-per-outcome", type=int, default=2)
    args = p.parse_args()

    env = InstructedPickPlace(pairs=args.pairs, wording=args.wording)
    tap = _InstructionTap(env)
    try:
        policy = SmolVLAPolicy.load(env, checkpoint=args.checkpoint)
        split = f"{args.pairs}/{args.wording}"
        print(f"smolvla on T5 [{split}], {args.episodes} episodes from seed {args.seed_start}\n")
        result = evaluate(
            tap, policy,
            task=f"robosuite/instructed_pick_place[{split}]", policy_name="smolvla",
            episodes=args.episodes, seed_start=args.seed_start,
            video_dir=args.out / "videos", video_per_outcome=args.video_per_outcome,
        )
    finally:
        env.close()

    summary = breakdown(tap.rows)
    write_results(result, args.out / "results.json",
                  extra={"checkpoint": str(args.checkpoint), "split": split, "t5": summary})
    (args.out / "episodes.json").write_text(json.dumps(tap.rows, indent=1) + "\n")

    s = summary["success"]
    print(f"\n[{split}] success {s['success']}/{s['episodes']}  "
          f"failures {summary['instruction_failures']}")
    for key, v in summary["by"].items():
        print(f"  {key:14s} {v['success']:3d}/{v['episodes']}")
    for name, video in result.videos.items():
        print(f"{name:9s} -> {video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
