"""Is the oracle's rewind exact? Two checks on T5, expert-driven, no learning.

1. Imagining must not change the real episode: the expert's run with an
   imagination every ``k`` steps ends in the same joint positions, step for step,
   as the same seed without any.
2. The imagined future must be the real future: the expert is deterministic, so
   the frame imagined at step t is the frame the plain run shows at t + k.

    MUJOCO_GL=egl uv run python experiments/idm/scripts/check_rewind.py --seeds 3
"""

from __future__ import annotations

import argparse

import numpy as np

from dream_robot.sims.robosuite.oracle import ExpertFuture
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import InstructedPickPlace
from dream_robot.sims.robosuite.tasks.instructed_pick_place.expert import ExpertPolicy


def plain(env, seed):
    obs = env.reset(seed=seed)
    expert = ExpertPolicy(env)
    expert.reset()
    states, frames = [obs.state.copy()], [obs.images["top"].copy()]
    while True:
        action, _ = expert(obs)
        r = env.step(action)
        obs = r.observation
        states.append(obs.state.copy())
        frames.append(obs.images["top"].copy())
        if r.terminated or r.truncated:
            return np.array(states), frames, r.success


def with_imagination(env, seed, k):
    obs = env.reset(seed=seed)
    expert = ExpertPolicy(env)
    expert.reset()
    oracle = ExpertFuture(env, ExpertPolicy(env), k=k)
    oracle.reset()
    states, futures = [obs.state.copy()], {}
    t = 0
    while True:
        future = oracle.observe(obs, imagine=t % k == 0)
        if future is not None:
            futures[t] = (future.images["top"].copy(), oracle.imagined_steps[-1])
        action, _ = expert(obs)
        r = env.step(action)
        obs, t = r.observation, t + 1
        states.append(obs.state.copy())
        if r.terminated or r.truncated:
            return np.array(states), futures, r.success


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    env = InstructedPickPlace()
    try:
        for seed in range(1000, 1000 + args.seeds):
            s0, frames, ok0 = plain(env, seed)
            s1, futures, ok1 = with_imagination(env, seed, args.k)
            same_len = len(s0) == len(s1)
            drift = np.abs(s0 - s1).max() if same_len else float("nan")
            px = [
                np.abs(img.astype(int) - frames[t + n].astype(int)).max()
                for t, (img, n) in futures.items() if t + n < len(frames)
            ]
            print(f"seed {seed}: plain {len(s0)} steps success={ok0} | imagined {len(s1)} "
                  f"steps success={ok1} | max state diff {drift:.2e} | "
                  f"{len(px)} futures, max pixel diff vs real t+k {max(px)}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
