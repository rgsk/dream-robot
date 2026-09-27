"""Film exactly what a T5 policy sees: its top + wrist frames, enlarged, with the prompt.

    MUJOCO_GL=egl uv run python experiments/t5/scripts/film_policy_view.py \
        --checkpoint experiments/t5/fm_onehot_n150/checkpoint.pt \
        --seeds 1000-1007 --out experiments/t5/videos/fm_onehot_policy_view.mp4

Seeds run in order from a freshly loaded policy, as ``core.eval`` runs them, so
seeds starting at 1000 replay the eval's episodes (the policy's noise generator
is seeded once at load, not per episode). Frames are upscaled nearest-neighbour:
no detail is added. The last frame of each episode holds for a second with the outcome.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dream_robot.core.registry import POLICIES
from dream_robot.core.video import write_video
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import InstructedPickPlace

SCALE = 4
GAP = 8
BAR = 56


def frame(images: dict, caption: str, status: str, color: str) -> np.ndarray:
    panels = [
        Image.fromarray(images[name]).resize(
            (images[name].shape[1] * SCALE, images[name].shape[0] * SCALE), Image.NEAREST
        )
        for name in ("top", "wrist")
    ]
    width = sum(p.width for p in panels) + GAP
    canvas = Image.new("RGB", (width, panels[0].height + BAR), "white")
    canvas.paste(panels[0], (0, BAR))
    canvas.paste(panels[1], (panels[0].width + GAP, BAR))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), caption, fill="black", font=ImageFont.load_default(size=20))
    draw.text((8, 30), status, fill=color, font=ImageFont.load_default(size=18))
    return np.asarray(canvas)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", choices=("smolvla", "flow_matching"), default="flow_matching")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--pairs", choices=("seen", "held_out"), default="seen")
    p.add_argument("--seeds", default="1000-1007", help="first-last, inclusive")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    first, last = (int(x) for x in args.seeds.split("-"))

    env = InstructedPickPlace(pairs=args.pairs, wording="template")
    frames: list[np.ndarray] = []
    try:
        policy = POLICIES[args.policy](env, checkpoint=args.checkpoint)
        for seed in range(first, last + 1):
            observation = env.reset(seed=seed)
            policy.reset()
            caption = f"seed {seed}: '{env.prompt}'"
            steps = 0
            while True:
                frames.append(frame(observation.images, caption,
                                    f"step {steps}   left: top 128x128   right: wrist 128x128",
                                    "gray"))
                result = env.step(policy(observation))
                observation = result.observation
                steps += 1
                if result.terminated or result.truncated:
                    break
            failure = None if result.success else env.instruction_failure()
            status = ("SUCCESS" if result.success
                      else f"FAIL: {failure or result.failure_mode}")
            print(f"seed {seed}  {env.prompt!r:42s} {status}")
            frames += [frame(observation.images, caption, f"step {steps}   {status}",
                             "green" if result.success else "red")] * int(env.control_hz)
    finally:
        env.close()

    write_video(frames, args.out, fps=env.control_hz)
    print(f"video -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
