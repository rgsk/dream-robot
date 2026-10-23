"""Make planning time visible: the world keeps running while the policy thinks.

In the normal eval the sim waits for every policy call, so a plan that takes 2 s
costs nothing. A real robot does not wait. Here each plan takes ``latency``
seconds of *sim* time, and during those steps the robot does one of two things:

- ``hold``: stands still on its last target until the plan lands. Lag shows as
  freezes; total frozen time = plans x latency, so replanning often costs more.
- ``async``: keeps running what is left of the old plan, and asks for the next
  one ``latency`` early so it lands as the old one runs out. No freezes, but the
  new plan was made from a picture ``latency`` old, so its first ``delay``
  actions are already in the past and are dropped. Needs delay + execute <= chunk.
  The seam jumps: the new plan never knew what the old one was doing.
- ``rtc``: async, but the new plan is drawn *to agree* with the old one (Real-Time
  Chunking, Black et al. 2025): its first ``delay`` actions are pulled onto the
  ones being executed meanwhile, the next few softly onto the old plan's tail, so
  it continues from where the arm will really be. No retraining.

Policy: flow matching on T2-fixed (80/80 normally), plan 16, run ``execute``.
The 20 s episode limit is the real-time budget: freezes eat into it.

    uv run python experiments/latency/scripts/lag_sweep.py --episodes 10
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from dream_robot.core.record import JointNoise
from dream_robot.core.registry import make_env, make_policy
from dream_robot.core.video import VideoWriter, observation_panel

ENV = "robosuite/pick_place_two_bins_fixed"
CHECKPOINTS = {
    "flow_matching": Path("experiments/t2_fixed_fm_half100/checkpoint.pt"),
    "act": Path("experiments/t2_fixed_act_half100/checkpoint.pt"),
}
OUT = Path("experiments/latency")
NOISE_SIGMA = 0.025  # as the T2-fixed FM eval: the scene is fixed, the shake varies it.
# It also makes the arm tremble on screen (fresh noise every step); --noise 0 to watch clean.


class LaggedPlanner:
    """Wraps a ``FlowPolicy``: one call per env step, plans land ``delay`` steps late."""

    def __init__(self, policy, *, execute: int, delay: int, mode: str):
        self._policy = policy
        self._chunk = policy._net.config.chunk
        self._gripper = policy._net.config.gripper_slice
        if mode in ("async", "rtc") and delay + execute > self._chunk:
            raise ValueError(f"{mode} needs delay + execute <= {self._chunk}")
        self.execute, self.delay, self.mode = execute, delay, mode
        self.reset()

    def reset(self) -> None:
        self._policy.reset()
        self._queue: deque[np.ndarray] = deque()
        self._pending: tuple[int, np.ndarray, int] | None = None  # (lands at, plan, origin)
        self._current: tuple[int, np.ndarray] | None = None  # (origin, plan) being executed
        self._last: np.ndarray | None = None
        self.t = 0
        self.plans = 0
        self.frozen = 0
        self.compute_s: list[float] = []

    @property
    def thinking(self) -> bool:
        return self._pending is not None

    def _start(self, observation) -> None:
        start = time.perf_counter()
        if self.mode == "rtc" and self._current is not None:
            plan = self._rtc_plan(observation)
        else:
            plan = self._policy._plan(observation)  # (chunk, dim), from this picture
        self.compute_s.append(time.perf_counter() - start)
        # A plan's index i is meant for step origin + i, origin = the step it was asked at.
        self._pending = (self.t + self.delay, plan, self.t)
        self.plans += 1

    def _rtc_plan(self, observation) -> np.ndarray:
        """Sample the next plan guided toward the one being executed (RTC)."""
        origin, old = self._current
        target = np.zeros_like(old)
        known = np.zeros(self._chunk, dtype=bool)
        for i in range(self._chunk):            # new index i is step t + i; old index j
            j = self.t + i - origin
            if 0 <= j < self._chunk:
                target[i], known[i] = old[j], True
        weights = rtc_weights(self._chunk, self.delay, self.execute) * known
        return rtc_sample(self._policy, observation, target, weights)

    def __call__(self, observation) -> np.ndarray:
        if self._last is None:
            self._last = np.asarray(observation.state, dtype=np.float32).copy()
        early = self.mode in ("async", "rtc")
        trigger = len(self._queue) <= (self.delay if early else 0)
        if self._pending is None and trigger:
            self._start(observation)
        if self._pending is not None and self.t >= self._pending[0]:
            _, plan, origin = self._pending
            # async / rtc: the first `delay` actions were meant for steps already gone
            skip = self.delay if early else 0
            self._queue = deque(plan[skip : skip + self.execute])
            self._current = (origin, plan)
            self._pending = None
        self.t += 1
        if self._queue:
            action = self._queue.popleft().copy()
            action[self._gripper] = np.clip(action[self._gripper], 0.0, 1.0)
            self._last = action
        else:
            self.frozen += 1
            action = self._last
        return action.astype(np.float32)


def rtc_weights(chunk: int, delay: int, execute: int) -> np.ndarray:
    '''How hard each index of the new plan is pulled onto the old plan.

    H = chunk, d = delay, s = execute (steps between plan starts):
        i < d            W = 1                       executed while thinking: must match
        d <= i < H - s   W = c (e^c - 1)/(e - 1),    c = (H - s - i)/(H - s - d + 1)
                                                     old plan's tail: fades 1 -> 0
        i >= H - s       W = 0                       past the old plan's end: free
    '''
    w = np.zeros(chunk, dtype=np.float32)
    w[:delay] = 1.0
    for i in range(delay, chunk - execute):
        c = (chunk - execute - i) / (chunk - execute - delay + 1)
        w[i] = c * (np.exp(c) - 1) / (np.e - 1)
    return w


def _inputs(policy, observation) -> tuple[dict, torch.Tensor]:
    from dream_robot.core.dataset import frame_to_float_chw

    device = policy._device
    images = {name: torch.from_numpy(frame_to_float_chw(observation.images[name]))
              .unsqueeze(0).to(device) for name in policy._net.cameras}
    state = torch.from_numpy(np.asarray(observation.state, np.float32)).unsqueeze(0).to(device)
    return images, state


'''
Guided Euler, flow time tau: 0 = noise, 1 = data, x_tau = tau·A + (1 - tau)·eps.

    Â   = x + (1 - tau)·v(x, tau)            one-jump guess of the final plan
    e   = W ⊙ (Y - Â)                        how far the guess is from the old plan, weighted
    g   = eᵀ ∂Â/∂x                            push x so that Â moves toward Y (one VJP)
    r²  = (1 - tau)² / (tau² + (1 - tau)²)
    v'  = v + min(beta, (1 - tau) / (tau·r²))·g

(ΠGDM-style guidance, as in RTC; beta = 5 caps the pull early on, where tau ~ 0.)
'''
def rtc_sample(policy, observation, target: np.ndarray, weights: np.ndarray,
               *, beta: float = 5.0) -> np.ndarray:
    net = policy._net
    images, state = _inputs(policy, observation)
    device = state.device
    with torch.no_grad():
        obs = net.encode(images, state, None)
        y = net.action_norm.normalize(torch.from_numpy(target).unsqueeze(0).to(device))
    w = torch.from_numpy(weights).to(device).view(1, -1, 1)
    x = torch.randn((1, net.config.chunk, net.config.action_dim),
                    generator=policy._generator, device=device)
    steps = net.config.sample_steps
    dt = 1.0 / steps
    for k in range(steps):
        tau = k * dt
        with torch.enable_grad():
            xg = x.detach().requires_grad_(True)
            v = net.velocity(xg, torch.full((1,), tau, device=device), obs)
            guess = xg + (1 - tau) * v
            (g,) = torch.autograd.grad(guess, xg, grad_outputs=w * (y - guess))
        r2 = (1 - tau) ** 2 / (tau ** 2 + (1 - tau) ** 2)
        scale = beta if tau == 0 else min(beta, (1 - tau) / (tau * r2))
        x = (xg + dt * (v + scale * g)).detach()
    return net.action_norm.denormalize(x)[0].cpu().numpy()

class LaggedEveryCall:
    """Hold-mode lag for a policy that plans on every call (ACT: plan, ensemble, 1 action).

    Each call's action lands ``delay`` steps after the picture it was planned from; the
    arm holds meanwhile. ACT's own step counter only counts calls, which is right here:
    the arm stands still between them, so its plans line up with the actions taken.
    """

    def __init__(self, policy, *, delay: int):
        self._policy, self.delay = policy, delay
        self.reset()

    def reset(self) -> None:
        self._policy.reset()
        self._queue: deque[np.ndarray] = deque()
        self._pending: tuple[int, np.ndarray] | None = None
        self._last: np.ndarray | None = None
        self.t = self.plans = self.frozen = 0
        self.compute_s: list[float] = []

    @property
    def thinking(self) -> bool:
        return self._pending is not None

    def __call__(self, observation) -> np.ndarray:
        if self._last is None:
            self._last = np.asarray(observation.state, dtype=np.float32).copy()
        if self._pending is None:
            start = time.perf_counter()
            self._pending = (self.t + self.delay, self._policy(observation))
            self.compute_s.append(time.perf_counter() - start)
            self.plans += 1
        self.t += 1
        if self.t > self._pending[0]:
            self._last = self._pending[1]
            self._pending = None
            return self._last
        self.frozen += 1
        return self._last


def overlay(frame: np.ndarray, planner: LaggedPlanner, hz: float, label: str) -> np.ndarray:
    frame = np.ascontiguousarray(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, label, (6, 16), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"t = {planner.t / hz:5.2f} s   plans {planner.plans}", (6, 34),
                font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if planner.thinking:
        frozen = not planner._queue
        text, colour = ("THINKING (frozen)", (230, 40, 40)) if frozen else (
            "THINKING (moving on old plan)", (240, 170, 0))
        cv2.rectangle(frame, (0, frame.shape[0] - 24), (frame.shape[1], frame.shape[0]),
                      colour, -1)
        cv2.putText(frame, text, (6, frame.shape[0] - 7), font, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return frame


def run_config(env, policy, *, policy_name, execute, latency, mode, episodes, seed_start,
               film, noise_sigma=NOISE_SIGMA):
    hz = float(env.control_hz)
    delay = int(round(latency * hz))
    if policy_name == "act":  # plans every call; only hold, every step
        planner = LaggedEveryCall(policy, delay=delay)
    else:
        planner = LaggedPlanner(policy, execute=execute, delay=delay, mode=mode)
    prefix = "act_" if policy_name == "act" else ""
    name = f"{prefix}{mode}_exec{execute}_lat{latency:g}" + ("" if noise_sigma else "_clean")
    label = f"{policy_name} {mode}: replan every {execute}, think {latency:g} s"
    writer = VideoWriter(OUT / "videos" / f"{name}.mp4", fps=hz) if film else None
    rows = []
    try:
        for i in range(episodes):
            seed = seed_start + i
            observation = env.reset(seed=seed)
            planner.reset()
            if hasattr(policy, "seed"):
                policy.seed(seed)
            noise = JointNoise(noise_sigma, env.embodiment)
            rng = np.random.default_rng(seed)
            frames = []
            while True:
                action = planner(observation)
                if writer is not None and i == 0:
                    panel = observation_panel(env.render(), observation.images)
                    frames.append(overlay(panel, planner, hz, label))
                result = env.step(noise(action, rng))
                observation = result.observation
                if result.terminated or result.truncated:
                    break
            if frames:
                writer.append(frames + [frames[-1]] * 15)
            rows.append({
                "seed": seed, "success": bool(result.success),
                "failure_mode": str(result.failure_mode), "steps": planner.t,
                "frozen_steps": planner.frozen, "plans": planner.plans,
                "compute_ms": round(1000 * float(np.mean(planner.compute_s)), 2),
            })
    finally:
        if writer is not None:
            writer.close()
    ok = [r for r in rows if r["success"]]
    summary = {
        "config": name, "mode": mode, "execute": execute, "latency_s": latency,
        "delay_steps": delay, "successes": len(ok), "episodes": len(rows),
        "median_seconds_success": (
            round(float(np.median([r["steps"] for r in ok])) / hz, 2) if ok else None),
        "frozen_fraction": round(
            sum(r["frozen_steps"] for r in rows) / sum(r["steps"] for r in rows), 3),
        "plans_per_episode": round(float(np.mean([r["plans"] for r in rows])), 1),
        "rollouts": rows,
    }
    print(f"{name:24s} {len(ok):2d}/{len(rows)}  "
          f"time {summary['median_seconds_success']}  frozen {summary['frozen_fraction']:.0%}  "
          f"plans {summary['plans_per_episode']}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", choices=sorted(CHECKPOINTS), default="flow_matching")
    parser.add_argument("--modes", nargs="+", default=["hold", "async", "rtc"])
    parser.add_argument("--execute", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--latency", nargs="+", type=float, default=[0, 0.16, 0.3, 0.6, 2.0])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=NOISE_SIGMA)
    parser.add_argument("--no-film", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT / "results.json")
    args = parser.parse_args()

    env = make_env(ENV)
    policy = make_policy(args.policy, env, checkpoint=CHECKPOINTS[args.policy])
    chunk = policy._net.config.chunk
    results = []
    try:
        for mode in args.modes:
            for latency in args.latency:
                for execute in args.execute:
                    delay = int(round(latency * env.control_hz))
                    if mode in ("async", "rtc") and (delay + execute > chunk or latency == 0):
                        continue  # can't cover the thinking time / same as hold
                    results.append(run_config(
                        env, policy, policy_name=args.policy, execute=execute, latency=latency, mode=mode,
                        episodes=args.episodes, seed_start=args.seed_start,
                        film=not args.no_film, noise_sigma=args.noise,
                    ))
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
