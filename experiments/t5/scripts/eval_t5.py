"""Evaluate a policy on T5 and say *which half* of the instruction each failure got wrong.

    MUJOCO_GL=egl uv run python experiments/t5/scripts/eval_t5.py \
        --checkpoint experiments/t5/smolvla_clean_n150/checkpoints/020000/pretrained_model \
        --pairs seen --wording template --episodes 60 --out experiments/t5/eval_20k/seen

    # the one-hot baseline: --policy flow_matching --checkpoint <dir>/checkpoint.pt
    # the oracle IDM: --policy oracle_idm --checkpoint <dir>/checkpoint.pt [--execute N]
    # Cosmos Policy: start serve.py (experiments/cosmos_policy/scripts), then
    #   --policy cosmos_policy --checkpoint localhost:6000 --camera-size 224 [--execute 16]

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
from dataclasses import asdict, replace
from pathlib import Path

from dream_robot.core.eval import EVAL_SEED_START, evaluate, write_results
from dream_robot.core.registry import POLICIES
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import (
    InstructedPickPlace,
    TaskConfig,
)


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
    p.add_argument("--policy", choices=("smolvla", "flow_matching", "oracle_idm", "cosmos_policy"), default="smolvla")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="cosmos_policy: the action server's host:port")
    p.add_argument("--pairs", choices=("seen", "held_out"), default="seen")
    p.add_argument("--wording", choices=("template", "paraphrase"), default="template")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--video-per-outcome", type=int, default=2)
    p.add_argument("--execute", type=int, default=None,
                   help="oracle_idm / cosmos_policy: actions played per plan "
                        "(default k / half the chunk)")
    p.add_argument("--camera-size", type=int, default=None,
                   help="square camera size; must match the training frames (cosmos_policy: 224)")
    p.add_argument("--horizon", default="k",
                   help="oracle_idm: steps imagined per clip: k (one frame), a number, or full")
    args = p.parse_args()

    config = TaskConfig.load()
    if args.camera_size:
        # the video panel stacks top over wrist beside the wide render: render 2x camera height
        (rh, rw), scale = config.render_hw, 2 * args.camera_size / config.render_hw[0]
        config = replace(config, camera_hw=(args.camera_size, args.camera_size),
                         render_hw=(round(rh * scale), round(rw * scale)))
    env = InstructedPickPlace(config, pairs=args.pairs, wording=args.wording)
    tap = _InstructionTap(env)
    try:
        extra = {}
        if args.policy == "oracle_idm":
            horizon = {"k": "k", "full": None}.get(args.horizon)
            extra = {"execute": args.execute,
                     "horizon": int(args.horizon) if args.horizon not in ("k", "full") else horizon}
        elif args.policy == "cosmos_policy":
            extra = {"execute": args.execute}
        policy = POLICIES[args.policy](env, checkpoint=args.checkpoint, **extra)
        split = f"{args.pairs}/{args.wording}"
        print(f"{args.policy} on T5 [{split}], "
              f"{args.episodes} episodes from seed {args.seed_start}\n")
        result = evaluate(
            tap, policy,
            task=f"robosuite/instructed_pick_place[{split}]",
            policy_name=args.policy,
            episodes=args.episodes, seed_start=args.seed_start,
            video_dir=args.out / "videos", video_per_outcome=args.video_per_outcome,
        )
    finally:
        env.close()

    summary = breakdown(tap.rows)
    clips = getattr(policy, "clips_per_episode", None)
    write_results(result, args.out / "results.json",
                  extra={"checkpoint": str(args.checkpoint), "split": split, "t5": summary,
                         **({"horizon": args.horizon, "clips_per_episode": clips,
                             "replans_per_episode": policy.replans_per_episode} if clips else {}),
                         **({"server": policy.server, "execute": policy.execute,
                             "mean_call_seconds": sum(policy.call_seconds) / len(policy.call_seconds)}
                            if args.policy == "cosmos_policy" and policy.call_seconds else {})})
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
