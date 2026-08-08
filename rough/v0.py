# rough/v0.py
import argparse
from collections.abc import Mapping, Sequence

import imageio.v3 as iio
import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_cube.env import (
    PickPlaceCube,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertPolicy


def upscale_nearest(frame: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return frame
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def observation_panel(
    wide: np.ndarray,
    images: Mapping[str, np.ndarray],
    *,
    order: Sequence[str] = ("top", "wrist"),
) -> np.ndarray:
    present = [name for name in order if name in images]
    if not present:
        raise ValueError(f"none of {list(order)} in images {sorted(images)}")
    if wide.shape[0] % len(present):
        raise ValueError(
            f"render height {wide.shape[0]}px is not divisible by {len(present)} cameras "
            f"{present}; each panel would be {wide.shape[0] / len(present)}px. "
            f"Set render.height in task.yaml to a multiple of {len(present)}."
        )

    target_h = wide.shape[0] // len(present)
    column = []
    for image_name in present:
        image = images[image_name]
        if target_h % image.shape[0]:
            raise ValueError(
                f"camera {image_name} is {image.shape[0]}px tall, which does not divide the "
                f"{target_h}px panel height. Integer scaling only -- see upscale_nearest."
            )
        column.append(upscale_nearest(image, target_h // image.shape[0]))

    stacked = np.concatenate(column, axis=0)
    panel = np.concatenate([wide, stacked], axis=1)
    return panel


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
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--hold-frames", type=int, default=15)
    args = p.parse_args(argv)
    cfg = TaskConfig.load()
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

        iio.imwrite(
            "rough/v0.mp4", np.stack(frames), fps=env.control_hz, codec="libx264"
        )
        print(len(frames), "frames")
    finally:
        env.close()
    return 0 if all_placed else 1

if __name__ == "__main__":
    raise SystemExit(main())
