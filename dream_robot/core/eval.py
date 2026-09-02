"""Rolling a policy out and reporting what happened. The number that counts.

Training loss says whether a policy reproduces the expert's action on frames
drawn from the expert's own trajectories. This says whether it can hold a
trajectory together for two hundred steps of its own. The gap between those two
is **compounding error**: BC is trained on states the expert visited, and at
rollout it visits states its own small mistakes led it to, which are not in the
training distribution, which produces bigger mistakes. A policy can sit
comfortably under the controller's own tracking error and still fail every
episode. That is not a contradiction; it is the reason this module exists.

Sim-agnostic and policy-agnostic: it takes ``core.env_api.Env`` and
``core.record.Policy``, both protocols, and imports neither a simulator nor
torch. The same function evaluates the scripted expert, which is what makes the
expert's success rate a comparable row in the matrix rather than a differently
computed number.

ROADMAP rule 5: every run writes ``results.json`` **and** a video.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import numpy as np

from dream_robot.core.env_api import Env, FailureMode
from dream_robot.core.record import Policy
from dream_robot.core.video import VideoWriter, observation_panel

#: Successes and failures filmed per run, when seeds are not named explicitly.
#:
#: Both, always. Filming "the first three episodes" is what the harness used to
#: do, and on the first BC run all three happened to be failures -- so the only
#: footage of the policy was of it losing, and whether it ever looked competent
#: was unanswerable without re-running by hand. Two of each is enough to see
#: whether the successes are clean or lucky.
VIDEO_PER_OUTCOME = 2

#: Where evaluation seeds start, by default.
#:
#: Far from the recorder's, which count up from 0. Evaluating on the seeds the
#: demonstrations were recorded from would measure whether the policy memorised
#: the twenty-five scenes it was trained on -- the number would be high, and it
#: would mean nothing. Cube placement is what the seed controls, so a disjoint
#: block is the whole of what "unseen scene" means for this task.
EVAL_SEED_START = 1000


@dataclass(frozen=True)
class EpisodeOutcome:
    seed: int
    success: bool
    failure_mode: FailureMode
    steps: int

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "success": self.success,
            "failure_mode": str(self.failure_mode),
            "steps": self.steps,
        }


@dataclass(frozen=True)
class EvalResult:
    """One cell of the results matrix."""

    task: str
    policy: str
    control_hz: float
    episodes: tuple[EpisodeOutcome, ...]
    videos: dict[str, Path] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.success for e in self.episodes) / len(self.episodes)

    @property
    def failure_histogram(self) -> dict[str, int]:
        """Where the failures went, in ``FailureMode``'s closed vocabulary.

        The part of a result that says what to do next. "40% success" is a
        number; "40% success and every failure is NO_GRASP" says the problem is
        in the approach and not the transport, and it is the same set of buckets
        on every task and every simulator, so the column sums.
        """
        counts = Counter(str(e.failure_mode) for e in self.episodes if not e.success)
        return dict(sorted(counts.items()))

    @property
    def cycle_time(self) -> float | None:
        """Median seconds to complete, over successes only.

        Failures have no cycle time. Averaging the horizon in for them would
        make a policy that fails fast look quick, and make cycle time move when
        success rate moves -- two numbers that need to stay independent.
        """
        times = [e.steps / self.control_hz for e in self.episodes if e.success]
        return median(times) if times else None

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "policy": self.policy,
            "episodes": len(self.episodes),
            "success_rate": round(self.success_rate, 4),
            "successes": sum(e.success for e in self.episodes),
            "cycle_time_seconds": (
                round(self.cycle_time, 2) if self.cycle_time is not None else None
            ),
            "mean_steps": round(mean(e.steps for e in self.episodes), 1),
            "failure_histogram": self.failure_histogram,
            "control_hz": self.control_hz,
            "videos": {name: str(path) for name, path in self.videos.items()},
            "rollouts": [e.to_dict() for e in self.episodes],
        }

    def summary(self) -> str:
        parts = [
            f"{self.policy} on {self.task}: "
            f"{self.success_rate:.0%} ({sum(e.success for e in self.episodes)}"
            f"/{len(self.episodes)})"
        ]
        if self.cycle_time is not None:
            parts.append(f"median cycle time {self.cycle_time:.1f}s")
        if self.failure_histogram:
            parts.append(f"failures {self.failure_histogram}")
        return "  |  ".join(parts)


def evaluate(
    env: Env,
    policy: Policy,
    *,
    task: str,
    policy_name: str,
    episodes: int = 20,
    seed_start: int = EVAL_SEED_START,
    video_dir: Path | None = None,
    video_per_outcome: int = VIDEO_PER_OUTCOME,
    video_seeds: Sequence[int] | None = None,
    hold_frames: int = 15,
    max_steps: int | None = None,
) -> EvalResult:
    """Roll ``policy`` out ``episodes`` times and report.

    Every episode runs to the environment's own terminal condition. Nothing is
    retried, nothing is discarded: unlike the recorder, which keeps successes
    because behaviour cloning needs clean targets, an evaluation that dropped
    its failures would be measuring nothing at all.

    **Which episodes get filmed.** By default, up to ``video_per_outcome`` of
    each outcome, into ``successes.mp4`` and ``failures.mp4``. Pass
    ``video_seeds`` to film exactly those seeds instead, into ``selected.mp4``.

    An episode's frames are buffered while it runs -- the outcome is not known
    until it ends -- and streamed out or dropped the moment it does, so memory
    is one episode rather than every filmed one. The alternative, re-running the
    chosen seeds afterwards with rendering on, would be cheaper still and is
    deliberately not what happens here: it only reproduces the same episode for
    a *deterministic* policy, and Diffusion Policy is coming. The footage must
    be of the episode that was actually counted.
    """
    wanted = set(video_seeds) if video_seeds is not None else None
    outcomes: list[EpisodeOutcome] = []
    writers: dict[str, VideoWriter] = {}
    if video_dir is not None:
        names = ["selected"] if wanted is not None else ["successes", "failures"]
        writers = {
            name: VideoWriter(Path(video_dir) / f"{name}.mp4", fps=env.control_hz)
            for name in names
        }
    filmed = Counter()

    try:
        for i in range(episodes):
            seed = seed_start + i
            observation = env.reset(seed=seed)
            policy.reset()
            # Explicit seeds are decided up front. In automatic mode the outcome
            # is unknown until the episode ends, so film speculatively while
            # *either* quota has room and spend one below -- but stop rendering
            # entirely once both are full, since render() is the expensive call.
            if wanted is not None:
                recording = bool(writers) and seed in wanted
            else:
                recording = bool(writers) and any(
                    filmed[name] < video_per_outcome for name in writers
                )
            episode_frames: list[np.ndarray] = []
            steps = 0
            success, failure_mode = False, FailureMode.TIMEOUT

            while max_steps is None or steps < max_steps:
                if recording:
                    # The wide view is for a human and never enters the dataset;
                    # the small panels are the policy's complete input. Seeing
                    # them together is how you notice the wrist camera went blind.
                    episode_frames.append(
                        observation_panel(env.render(), observation.images)
                    )
                result = env.step(policy(observation))
                observation = result.observation
                steps += 1
                success, failure_mode = result.success, result.failure_mode
                if result.terminated or result.truncated:
                    break

            outcomes.append(
                EpisodeOutcome(
                    seed=seed,
                    success=success,
                    failure_mode=FailureMode.NONE if success else failure_mode,
                    steps=steps,
                )
            )

            if episode_frames:
                bucket = "selected" if wanted is not None else (
                    "successes" if success else "failures"
                )
                if wanted is not None or filmed[bucket] < video_per_outcome:
                    filmed[bucket] += 1
                    # Hold the last frame so the outcome is readable at 30 fps.
                    writers[bucket].append(
                        episode_frames + [episode_frames[-1]] * hold_frames
                    )
            print(
                f"  seed {seed}: {'SUCCESS' if success else str(failure_mode).upper()} "
                f"in {steps} steps"
            )
    finally:
        videos = {name: w.close() for name, w in writers.items()}

    return EvalResult(
        task=task,
        policy=policy_name,
        control_hz=float(env.control_hz),
        episodes=tuple(outcomes),
        videos={name: path for name, path in videos.items() if path is not None},
    )


def write_results(result: EvalResult, path: Path, *, extra: dict | None = None) -> Path:
    """``results.json`` for one cell. ROADMAP rule 5.

    ``extra`` carries provenance that the harness cannot know -- which
    checkpoint, which dataset, how many demonstrations. Without it a results
    file records a number with no way to reproduce it, which is the same as not
    having recorded it.
    """
    payload = result.to_dict()
    if extra:
        payload["provenance"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
