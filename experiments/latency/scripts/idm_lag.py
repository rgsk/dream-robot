"""The oracle IDM on T5 with thinking time: drawing a clip and reading it cost sim time.

Two costs:
- ``--clip-s``: drawing a clip (the video model's job; the oracle is instant, a
  real video model is not -- seconds distilled, tens of seconds full).
- ``--read-s``: one IDM read, (real frame now, next imagined frame) -> k actions.
  The real IDM takes ~1 ms; default 0.

``--horizon``: ``full`` draws the whole episode as one clip, ``k`` one frame per
clip (a new clip every 10 steps), a number = steps per clip (a multiple of k).

``--mode``:
- ``hold``: the arm stands on its last target while a clip or a read is computed.
- ``async``: the next clip is asked for while the current one is still being
  walked (when what is left of it is no longer than the drawing time); when it
  lands, the frames already in the past are skipped. Only helps if a clip lasts
  longer than it takes to draw. The first clip is always a wait: nothing to walk yet.
No RTC: the IDM is a regression net, nothing to guide; and the oracle's clips
already agree with each other (deterministic expert). RTC returns with a sampled
video model.

    uv run python experiments/latency/scripts/idm_lag.py --horizon 120 --clip-s 3 --mode async
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
    """Wraps an ``IDMPolicy``: clips land ``clip_steps`` late, reads ``read_steps`` late."""

    def __init__(self, policy, *, clip_steps: int, read_steps: int, mode: str = "hold"):
        self._p = policy
        self.k = policy.k
        self.clip_steps, self.read_steps, self.mode = clip_steps, read_steps, mode
        self._clear()

    def reset(self) -> None:
        """After env.reset(): the shadow expert latches onto the episode."""
        self._p.reset()
        self._clear()

    def _clear(self) -> None:
        self._queue: list[np.ndarray] = []
        self._staged: tuple[int, list[np.ndarray], dict] | None = None  # (lands at, plan, aim)
        # Imagined frames as (the step they show, frame); a pending clip as
        # (lands at, frames, frozen-step count when asked).
        self._clip: list[tuple[int, object]] = []
        self._drawing: tuple[int, list, int] | None = None
        self._last: np.ndarray | None = None
        self.aim_images = None
        self.t = self.plans = self.clips = self.frozen = 0

    @property
    def thinking(self) -> bool:
        return self._staged is not None or self._drawing is not None

    def _want_clip(self, need_read: bool) -> bool:
        if self._drawing is not None or not need_read:
            return False
        if self.mode == "hold":
            return not self._clip
        return len(self._clip) * self.k <= self.clip_steps   # what's left won't outlast drawing

    def _read(self, observation) -> None:
        p = self._p
        _, future = self._clip.pop(0)
        state = torch.from_numpy(np.asarray(observation.state, np.float32)).unsqueeze(0)
        plan = p._net.predict(p._images(observation), p._images(future), state.to(p._device))
        aim = {name: future.images[name] for name in p._net.cameras}
        self._staged = (self.t + self.read_steps, list(plan[0, : p.execute].cpu().numpy()), aim)
        self.plans += 1

    def __call__(self, observation) -> np.ndarray:
        if self._last is None:
            self._last = np.asarray(observation.state, dtype=np.float32).copy()
        need_read = not self._queue and self._staged is None
        want = self._want_clip(need_read)
        clip = self._p._future.observe(observation, imagine=want)  # shadow follows every step
        if clip is not None:
            self._drawing = (self.t + self.clip_steps, list(clip), self.frozen)
            self.clips += 1
        if self._drawing is not None and self.t >= self._drawing[0]:
            _, frames, frozen_then = self._drawing
            # The clip starts from the picture it was drawn from. Only the steps the arm
            # actually moved since (walking the old clip) put its first frames behind us;
            # frozen steps don't (hold, or the first clip of an episode).
            moved = self.clip_steps - (self.frozen - frozen_then)
            origin = self.t - moved
            self._clip = [(origin + self.k * (j + 1), f) for j, f in enumerate(frames)]
            self._drawing = None
        self._clip = [(when, f) for when, f in self._clip if when > self.t]
        if need_read and self._clip and (self.mode == "async" or self._drawing is None):
            self._read(observation)
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
    parser.add_argument("--horizon", default="full", help="full, k, or steps per clip")
    parser.add_argument("--mode", choices=("hold", "async"), default="hold")
    parser.add_argument("--clip-s", type=float, default=0.0)
    parser.add_argument("--read-s", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    env = InstructedPickPlace(pairs="seen", wording="template")
    hz = float(env.control_hz)
    horizon = {"full": None, "k": "k"}.get(args.horizon, None)
    if args.horizon not in ("full", "k"):
        horizon = int(args.horizon)
    policy = POLICIES["oracle_idm"](env, checkpoint=CHECKPOINT, horizon=horizon)
    planner = LaggedIDM(policy, clip_steps=round(args.clip_s * hz),
                        read_steps=round(args.read_s * hz), mode=args.mode)
    tag = "" if args.mode == "hold" else f"_{args.mode}"
    name = f"idm_H{args.horizon}_clip{args.clip_s:g}_read{args.read_s:g}{tag}"
    label = f"IDM {args.mode}, H={args.horizon}: clip {args.clip_s:g} s, read {args.read_s:g} s"
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
