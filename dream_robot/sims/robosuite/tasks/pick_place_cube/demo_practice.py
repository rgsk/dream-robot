"""Watch the scripted expert do the task. Regenerates the demo video.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from dream_robot.core.video_practice import observation_panel, write_video
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import (
    PickPlaceCube,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertPolicy

DEFAULT_OUT = Path("experiments/expert_demo/videos/expert_pick_place_cube_practice.mp4")

def rollout_frames(
    cfg: TaskConfig, env: PickPlaceCube, policy: ExpertPolicy, seed: int, hold: int
):
    """One scripted episode. Returns (frames, placed, reason, steps)."""
    obs = env.reset(seed=seed)
    policy.reset()
    frames: list[np.ndarray] = []
    result = None
    command = None
    frames.append(observation_panel(env.render(), obs.images))
    steps = 0
    for _ in range(cfg.horizon_steps):
        action, command = policy(obs)
        result = env.step(action)
        obs = result.observation
        frames.append(observation_panel(env.render(), obs.images))
        steps += 1
        if result.success or result.truncated or command.timed_out:
            break

    # Hold the final frame so the placement is readable at 30 fps.
    frames.extend([frames[-1]] * hold)

    if result.success:
        placed, reason = True, "placed"
    # Two different failures: the expert gave up on a phase, or the task did.
    elif command.timed_out:
        placed, reason = False, f"stuck in {command.phase}"
    else:
        placed, reason = False, str(result.failure_mode)

    return frames, placed, reason, steps


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--render-camera",
        default=None,
        help="override task.yaml's render camera (video only; never the dataset)",
    )
    p.add_argument("--hold-frames", type=int, default=15)
    args = p.parse_args(argv)
    cfg = TaskConfig.load()
    if args.render_camera:
        cfg = replace(cfg, render_camera=args.render_camera)    
    env = PickPlaceCube(cfg)
    policy = ExpertPolicy(env)
    all_placed = True
    try:
        frames: list[np.ndarray] = []
        for seed in args.seeds:
            episode, placed, reason, steps = rollout_frames(
                cfg, env, policy, seed, args.hold_frames
            )
            if not placed:
                all_placed = False
            frames.extend(episode)
            print(
                f"seed {seed}: {'PLACED' if placed else 'FAILED'} ({reason}), "
                f"{steps} steps"
            )

        path = write_video(frames, args.out, fps=env.control_hz)
        print(f"\n{len(frames)} frames @ {env.control_hz:g} fps -> {path.resolve()}")
    finally:
        env.close()
    return 0 if all_placed else 1

if __name__ == "__main__":
    raise SystemExit(main())
