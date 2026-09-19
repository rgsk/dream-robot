"""Run the scripted two-arm expert over seeds and print what happened.

    MUJOCO_GL=egl uv run python experiments/t6/scripts/expert_rollout.py [n_seeds] [--no-barrier]

The tuning loop for step 2, and later the ablation rig: with --no-barrier each
arm advances on its own predicates, which is the DESYNCHRONISED failure the
task exists to measure.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import replace

import numpy as np

from dream_robot.sims.robosuite.tasks.two_arm_lift.env import TwoArmLiftTask
from dream_robot.sims.robosuite.tasks.two_arm_lift.expert import ExpertConfig, ExpertPolicy


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 5
    barrier = "--no-barrier" not in sys.argv
    cfg = ExpertConfig.load()
    if not barrier:
        cfg = replace(cfg, sync_barrier=False)
    env = TwoArmLiftTask()
    policy = ExpertPolicy(env, cfg)
    print(f"barrier={barrier}  seeds 1000..{1000 + n - 1}")
    print(f"{'seed':>5} {'result':>8} {'steps':>6} {'phase':>16} {'lift':>6} "
          f"{'tilt':>6} {'marbles':>7} {'held':>8}")
    wins = 0
    for seed in range(1000, 1000 + n):
        obs = env.reset(seed=seed)
        policy.reset()
        command = None
        result = None
        for t in range(env._cfg.horizon_steps):
            action, command = policy(obs)
            result = env.step(action)
            obs = result.observation
            if result.success or result.truncated or command.timed_out:
                break
        outcome = ("SUCCESS" if result.success
                   else f"stuck" if command.timed_out else str(result.failure_mode))
        wins += bool(result.success)
        print(f"{seed:>5} {outcome:>8} {t + 1:>6} {command.phase:>16} "
              f"{env._pot_lift():>6.3f} {env._pot_tilt_deg():>6.1f} "
              f"{env.marbles_inside:>7} {str(env.handles_held):>8}")
    print(f"\n{wins}/{n}")
    env.close()


if __name__ == "__main__":
    main()
