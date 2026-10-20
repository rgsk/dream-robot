"""The oracle IDM on T5 with thinking time: drawing a clip and reading it cost sim time.

Two costs, both paid in ``hold`` (the arm stands on its last target meanwhile):
- ``--clip-s``: drawing a clip (the video model's job; the oracle is instant, a
  real video model is not -- seconds distilled, tens of seconds full).
- ``--read-s``: one IDM read, (real frame now, next imagined frame) -> k actions.
  The real IDM takes ~1 ms; default 0.

``--horizon full`` draws the whole episode as one clip (one clip cost); ``k``
draws one frame per clip, i.e. a new clip every 10 steps.

    uv run python experiments/latency/scripts/idm_lag.py --horizon full --clip-s 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from dream_robot.core.registry import POLICIES
from dream_robot.core.video import VideoWriter, observation_panel
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import InstructedPickPlace

sys.path.insert(0, str(Path(__file__).parent))
from lag_sweep import overlay  # noqa: E402

CHECKPOINT = Path("experiments/idm/k10/checkpoint.pt")
OUT = Path("experiments/latency")


class LaggedIDM:
    """Wraps an ``IDMPolicy`` (hold only): a read lands ``read + clip`` steps after asked."""

    def __init__(self, policy, *, clip_steps: int, read_steps: int):
        self._p = policy
        self.clip_steps, self.read_steps = clip_steps, read_steps
        self._clear()

    def reset(self) -> None:
        """After env.reset(): the shadow expert latches onto the episode."""
        self._p.reset()
        self._clear()

    def _clear(self) -> None:
        self._queue: list[np.ndarray] = []
        self._staged: tuple[int, list[np.ndarray], dict] | None = None  # (lands at, plan, aim)
        self._last: np.ndarray | None = None
        self.aim_images = None
        self.t = self.plans = self.clips = self.frozen = 0

    @property
    def thinking(self) -> bool:
        return self._staged is not None

    def _think(self, observation) -> None:
        p = self._p
        imagine = not p._clip
        clip = p._future.observe(observation, imagine=imagine)
        if clip is not None:
            p._clip = list(clip)
            self.clips += 1
        future = p._clip.pop(0)
        state = torch.from_numpy(np.asarray(observation.state, np.float32)).unsqueeze(0)
        plan = p._net.predict(p._images(observation), p._images(future), state.to(p._device))
        aim = {name: future.images[name] for name in p._net.cameras}
        cost = self.read_steps + (self.clip_steps if imagine else 0)
        self._staged = (self.t + cost, list(plan[0, : p.execute].cpu().numpy()), aim)
        self.plans += 1

    def __call__(self, observation) -> np.ndarray:
        if self._last is None:
            self._last = np.asarray(observation.state, dtype=np.float32).copy()
        if not self._queue and self._staged is None:
            self._think(observation)                  # observes this step itself
        else:
            self._p._future.observe(observation, imagine=False)  # shadow follows the real step
        if self._staged is not None and self.t >= self._staged[0]:
            _, self._queue, self.aim_images = self._staged
            self._staged = None
        self.t += 1
        if self._queue:
            action = self._queue.pop(0).astype(np.float32)
            g = self._p._net.config.gripper_slice
            action[g] = np.clip(action[g], 0.0, 1.0)
            self._last = action
        else:
            self.frozen += 1
            action = self._last
        return action.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", choices=("k", "full"), default="full")
    parser.add_argument("--clip-s", type=float, default=0.0)
    parser.add_argument("--read-s", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    env = InstructedPickPlace(pairs="seen", wording="template")
    hz = float(env.control_hz)
    policy = POLICIES["oracle_idm"](env, checkpoint=CHECKPOINT,
                                    horizon=None if args.horizon == "full" else "k")
    planner = LaggedIDM(policy, clip_steps=round(args.clip_s * hz),
                        read_steps=round(args.read_s * hz))
    name = f"idm_H{args.horizon}_clip{args.clip_s:g}_read{args.read_s:g}"
    label = f"IDM, H={args.horizon}: clip {args.clip_s:g} s, read {args.read_s:g} s"
    frames = []
    try:
        observation = env.reset(seed=args.seed)
        planner.reset()
        while True:
            action = planner(observation)
            panel = observation_panel(env.render(), observation.images, aim=planner.aim_images)
            frames.append(overlay(panel, planner, hz, label))
            result = env.step(action)
            observation = result.observation
            if result.terminated or result.truncated:
                break
    finally:
        env.close()
    with VideoWriter(OUT / "videos" / f"{name}.mp4", fps=hz) as writer:
        writer.append(frames + [frames[-1]] * 15)
    row = {"config": name, "prompt": env.prompt, "success": bool(result.success),
           "failure_mode": str(result.failure_mode), "seconds": round(planner.t / hz, 2),
           "frozen_fraction": round(planner.frozen / planner.t, 3),
           "reads": planner.plans, "clips": planner.clips}
    print(row, flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
