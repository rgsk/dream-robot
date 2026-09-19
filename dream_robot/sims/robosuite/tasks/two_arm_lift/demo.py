"""Watch the scripted two-arm expert, with the barrier on or off.

    MUJOCO_GL=egl uv run python -m dream_robot.sims.robosuite.tasks.two_arm_lift.demo
    MUJOCO_GL=egl uv run python -m dream_robot.sims.robosuite.tasks.two_arm_lift.demo --no-barrier

Thin, like T1's: a rollout loop, a panel per step, one file out. Not the eval
harness -- it reports whether each episode lifted the tray and nothing else.

``--no-barrier`` is the reason this file takes a flag at all. The same expert
with its synchronisation removed is the clearest statement of what the barrier
buys, and it is a video rather than a column: one arm grasps, lifts alone, and
the tray pivots on the handle the other arm has not reached yet.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from dream_robot.core.video import observation_panel, write_video
from dream_robot.sims.robosuite.tasks.two_arm_lift.env import TaskConfig, TwoArmLiftTask
from dream_robot.sims.robosuite.tasks.two_arm_lift.expert import ExpertConfig, ExpertPolicy

DEFAULT_OUT = Path("experiments/t6/videos/expert_two_arm_lift.mp4")
PANEL_ORDER = ("top", "wrist_left", "wrist_right")


def rollout_frames(env: TwoArmLiftTask, policy: ExpertPolicy, seed: int, hold: int):
    """One scripted episode. Returns (frames, lifted, reason)."""
    obs = env.reset(seed=seed)
    policy.reset()
    frames: list[np.ndarray] = []
    result = None
    command = None

    for _ in range(env._cfg.horizon_steps):
        action, command = policy(obs)
        frames.append(observation_panel(env.render(), obs.images, order=PANEL_ORDER))
        result = env.step(action)
        obs = result.observation
        if result.success or result.truncated or command.timed_out:
            break

    frames.extend([frames[-1]] * hold)
    if result.success:
        return frames, True, f"lifted, tilt {env._pot_tilt_deg():.0f} deg"
    if command.timed_out:
        return frames, False, f"stuck in {command.phase}"
    return frames, False, str(result.failure_mode)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--no-barrier",
        action="store_true",
        help="each arm advances on its own predicates: the DESYNCHRONISED ablation",
    )
    p.add_argument("--render-camera", default=None)
    p.add_argument("--hold-frames", type=int, default=15)
    args = p.parse_args(argv)

    cfg = TaskConfig.load()
    if args.render_camera:
        cfg = replace(cfg, render_camera=args.render_camera)
    expert_cfg = ExpertConfig.load()
    if args.no_barrier:
        expert_cfg = replace(expert_cfg, sync_barrier=False)

    env = TwoArmLiftTask(cfg)
    policy = ExpertPolicy(env, expert_cfg)
    try:
        frames: list[np.ndarray] = []
        for seed in args.seeds:
            episode, lifted, reason = rollout_frames(env, policy, seed, args.hold_frames)
            frames.extend(episode)
            print(f"seed {seed}: {'LIFTED' if lifted else 'FAILED'} ({reason}), "
                  f"{len(episode) - args.hold_frames} steps")
        path = write_video(frames, args.out, fps=env.control_hz)
        print(f"\n{len(frames)} frames @ {env.control_hz:g} fps -> {path.resolve()}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
