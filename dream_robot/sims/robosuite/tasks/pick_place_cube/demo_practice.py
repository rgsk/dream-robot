"""Watch the scripted expert do the task. Regenerates the demo video.

    python -m dream_robot.sims.robosuite.tasks.pick_place_cube.demo

This is the task's own entrypoint, and it is deliberately thin: a rollout loop,
a panel per step, one file out. When core/run.py and the registry land, the
rollout loop moves there and this shrinks to a registration.

It is not the eval harness. It reports whether each episode placed the cube and
nothing else -- no success rate, no cycle time, no failure histogram, no
results.json. That is core/eval.py's job, and conflating the two is how a demo
script quietly becomes the thing that reports your numbers.
"""
from pathlib import Path
from dataclasses import replace
import argparse
import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_cube.env_practice import PickPlaceCube, TaskConfig
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert_practice import ExpertPolicy

DEFAULT_OUT = Path("experiments/expert_demo/videos/expert_pick_place_cube_practice.mp4")

def rollout_frames(env: PickPlaceCube, policy: ExpertPolicy, seed: int, hold: int):
    """One scripted episode. Returns (frames, placed, reason)."""
    obs = env.reset(seed=seed)
    policy.reset()
    frames: list[np.ndarray] = []
    result = None
    command = None

    for _ in range(env._cfg.horizon_steps):
        action, command = policy(obs)
        frames.append(observation_panel(env.render(), obs.images))
        result = env.step(action)
        obs = result.observation
        if result.success or result.truncated or command.timed_out:
            break

    # Hold the final frame so the placement is readable at 30 fps.
    frames.extend([frames[-1]] * hold)
    if result.success:
        return frames, True, "placed"
    # Two different failures: the expert gave up on a phase, or the task did.
    if command.timed_out:
        return frames, False, f"stuck in {command.phase}"
    return frames, False, str(result.failure_mode)


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
    try:
        frames: list[np.ndarray] = []
        for seed in args.seeds:
            episode, placed, reason = rollout_frames(env, policy, seed, args.hold_frames)
    finally:
            env.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
