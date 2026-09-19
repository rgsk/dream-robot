"""Does a tilted lift spill if you make it hold the tray up?

    MUJOCO_GL=egl uv run python experiments/t6/scripts/hold_after_lift.py [--no-barrier]

The success test today fires the instant the tray clears 0.10 m, which is how a
48 degree hoist scored SUCCESS with 8 of 9 marbles still aboard: at that angle
the marbles need seconds to roll the length of the tray and hop the lip, and the
episode ends in milliseconds.

So instead of gating success on the tilt ANGLE, gate it on holding the height.
This rig asks the question the change rests on: held up for 1, 2, 3 seconds,
does a tilted tray actually shed its marbles -- and does a clean one keep them?
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import replace

import numpy as np

from dream_robot.sims.robosuite.tasks.two_arm_lift.env import TwoArmLiftTask
from dream_robot.sims.robosuite.tasks.two_arm_lift.expert import ExpertConfig, ExpertPolicy

HOLD_SECONDS = (0.0, 1.0, 2.0, 3.0)
GIVE_UP_STEPS = 260          # an episode that has not lifted by here never will


def main():
    barrier = "--no-barrier" not in sys.argv
    cfg = ExpertConfig.load()
    if not barrier:
        cfg = replace(cfg, sync_barrier=False)
    env = TwoArmLiftTask()
    policy = ExpertPolicy(env, cfg)
    hz = env.control_hz
    marks = [int(round(s * hz)) for s in HOLD_SECONDS]

    print(f"barrier={barrier}   required aboard: {env._cfg.marbles.required_inside}"
          f" of {env._cfg.marbles.count}")
    head = "  ".join(f"{s:>10.0f}s" for s in HOLD_SECONDS)
    print(f"{'seed':>5} {'lifted':>7} | tilt / marbles aboard / tray height, held for {head}")
    rows = []
    for seed in range(1000, 1020):
        obs = env.reset(seed=seed)
        policy.reset()
        lifted_at = None
        samples = {}
        for t in range(env._cfg.horizon_steps):
            action, command = policy(obs)
            result = env.step(action)
            obs = result.observation
            if lifted_at is None and result.success:
                lifted_at = t
            if lifted_at is not None:
                held = t - lifted_at
                if held in marks:
                    samples[held] = (env._pot_tilt_deg(), env.marbles_inside,
                                     env._pot_lift())
                if held >= marks[-1]:
                    break
            elif t >= GIVE_UP_STEPS:
                break
        if lifted_at is None:
            print(f"{seed:>5} {'no':>7} | never reached the height")
            continue
        cells = []
        for m in marks:
            if m in samples:
                tilt, aboard, lift = samples[m]
                cells.append(f"{tilt:3.0f}d/{aboard}/{lift:.2f}")
            else:
                cells.append("-")
        rows.append((seed, samples))
        print(f"{seed:>5} {'yes':>7} | {'   '.join(f'{c:>6}' for c in cells)}")

    need = env._cfg.marbles.required_inside
    for m, s in zip(marks, HOLD_SECONDS):
        kept = [r for _, samp in rows if m in samp for r in [samp[m][1]]]
        if kept:
            passing = sum(k >= need for k in kept)
            up = [r for _, samp in rows if m in samp for r in [samp[m][2]]]
            still_up = sum(h > 0.10 for h in up)
            print(f"hold {s:.0f}s: {passing}/{len(kept)} keep {need}+ marbles, "
                  f"{still_up}/{len(up)} still hold the tray above 0.10 m")
    env.close()


if __name__ == "__main__":
    main()
