"""Turning rollouts into a dataset on disk. Sim-agnostic, like the rest of core.

This is the producer side of ROADMAP's seam. It drives an ``Env`` with a policy,
buffers the episode, decides whether the episode is worth keeping, and writes it
in LeRobotDataset format. Nothing downstream of the directory it writes knows
that robosuite exists.

**The ban on ground truth is enforced by what this file cannot reach.** The
recorder never touches ``env`` except through ``reset`` / ``step`` / ``close``,
and it builds every frame from an ``Observation`` -- which ``schema.py`` already
makes impossible to construct out of a backend's raw dict. So there is no path
by which ``cube_pos`` reaches the disk, and adding one would mean editing three
files that each say in their docstring why not to. That is what spec.md means by
"enforced structurally, not a review checklist". ``PickPlaceCube.cube_pos``
exists and the scripted expert reads it every step; it is on the far side of a
boundary this module has no way to cross.

**Only successful episodes are kept, by default.** Behaviour cloning imitates
what it is shown, so a dataset with failures in it is a dataset that teaches
failing. The noisy-expert work ROADMAP calls for is a different thing that is
easy to confuse with this one: recoveries from perturbed states are *successes*
that start somewhere unusual, and they widen the ribbon of state space the data
covers. Failed episodes narrow nothing and poison the target.

**What gets executed and what gets recorded can differ, on purpose.** A
``perturb`` hook adds noise to the action sent to ``env.step`` while the
dataset keeps the policy's own, clean action (DART, Laskey et al. 2017). The
noise pushes the arm somewhere the expert would never have gone; the clean
label is the expert's correction from there. Recording the noisy action instead
would teach the policy to reproduce the jitter rather than to recover from it.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from dream_robot.core.dataset import (
    ACTION_FEATURE,
    STATE_FEATURE,
    create_dataset,
    image_feature,
)
from dream_robot.core.env_api import Env, FailureMode
from dream_robot.core.schema import Embodiment, Observation, joint_tracking_error

#: A rollout that neither terminates nor truncates is an environment bug, not a
#: long episode. Fail loudly instead of filling the disk overnight.
RUNAWAY_STEPS = 100_000

#: Ceiling on |action[t] - state[t+1]| over the arm joints, in radians.
#:
#: This is a *mode* check, not a precision one. Absolute joint targets keep the
#: error bounded by how far the controller lags one commanded step, so it sits
#: just above ``expert.ik.max_joint_step`` -- the largest move the IK is allowed
#: to command in a step -- saturating on the steps where the arm has not begun
#: moving yet. Over 25 recorded episodes of robosuite pick_place_cube: mean
#: 0.018-0.020, peak 0.062 against a 0.060 clamp. A controller quietly running
#: in delta mode, or a recorder writing the wrong column, puts this in the order
#: of radians instead. 0.25 is ~4x the observed ceiling: loose enough never to
#: fire on a well-tuned task, tight enough that the failure it exists to catch
#: cannot slip past.
TRACKING_TOLERANCE = 0.25


#: ``(action, rng) -> executed action``. Must return a new array; the recorded
#: action is the one passed in. Seeded per episode from the episode seed, so a
#: perturbed recording is exactly as reproducible as a clean one.
Perturb = Callable[[np.ndarray, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class JointNoise:
    """Gaussian noise on the arm joints of the executed action, in radians.

    The gripper is left alone. Its command is an open/close decision, not a
    position the expert converges on, so noise there does not produce a
    recovery -- it produces a gripper that flickers, and a release at the wrong
    moment is a failed episode rather than a wider one.

    Measured on robosuite pick_place_cube: the expert still succeeds on 9/10
    seeds at sigma 0.05 and collapses to 3/10 by 0.1.
    """

    sigma: float
    embodiment: Embodiment

    def __call__(self, action: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        executed = np.array(action, dtype=np.float32, copy=True)
        arm = self.embodiment.arm_slice
        executed[arm] += rng.normal(0.0, self.sigma, self.embodiment.arm_joints)
        return executed

    def to_dict(self) -> dict:
        return {"type": "joint_noise", "sigma": self.sigma}


@runtime_checkable
class Policy(Protocol):
    """What the recorder needs from whatever is producing actions.

    Narrower than the scripted expert's own interface on purpose. The expert
    also reports which phase it is in and whether it gave up, and none of that
    belongs here: an episode is kept or discarded on what the *environment* says
    happened, so a recorder that also knew the expert's internal state would
    have two competing accounts of the same episode. Teleop and a trained
    policy satisfy this same protocol without inventing a phase to report.
    """

    def reset(self) -> None: ...

    def __call__(self, observation: Observation) -> np.ndarray: ...


@dataclass(frozen=True)
class Episode:
    """One rollout, buffered in memory, before the keep/discard decision.

    ``states``, ``actions`` and ``images`` are all length N and index-aligned:
    entry ``t`` is the observation the policy saw and the action it emitted in
    response. That is exactly the pair a dataset frame holds.

    The terminal observation -- the one showing the cube already in the bin --
    is therefore **not** in the episode. There is no action to pair it with, and
    inventing one (repeat the last, emit a zero) would put a command in the
    action column that the policy never issued. The visible consequence is that
    a video reconstructed from the dataset stops one frame before the placement.
    That is the honest picture of what a dataset of (observation, action) pairs
    contains.

    Buffering rather than streaming to ``add_frame`` is what makes the
    keep/discard decision possible at all: the outcome is only known at the end,
    and ~215 steps x 2 cameras x 128x128x3 is about 21 MB.
    """

    seed: int
    states: np.ndarray                       # float32 (N, dim)
    actions: np.ndarray                      # float32 (N, dim)
    images: list[dict[str, np.ndarray]]      # N frames, canonical camera -> uint8
    success: bool
    failure_mode: FailureMode

    @property
    def steps(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class EpisodeRecord:
    """What the summary keeps about an attempt. No pixels."""

    seed: int
    steps: int
    success: bool
    failure_mode: FailureMode
    kept: bool
    tracking_error_mean: float
    tracking_error_max: float

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "success": self.success,
            "failure_mode": str(self.failure_mode),
            "kept": self.kept,
            "tracking_error_mean": round(self.tracking_error_mean, 6),
            "tracking_error_max": round(self.tracking_error_max, 6),
        }


@dataclass(frozen=True)
class RecordingSummary:
    """The provenance of a dataset: which seeds, which outcomes, what was kept.

    Written next to the data because a dataset whose seeds live only in a shell
    history is not reproducible, and "how good is the expert" is a number you
    want before you spend a day wondering why BC plateaued.
    """

    repo_id: str
    root: Path
    task_prompt: str
    episodes: tuple[EpisodeRecord, ...] = field(default_factory=tuple)
    #: How the executed actions were perturbed, or None for a clean recording.
    #: Two datasets with identical seeds and different noise are different data.
    perturbation: dict | None = None

    @property
    def attempted(self) -> int:
        return len(self.episodes)

    @property
    def kept(self) -> int:
        return sum(e.kept for e in self.episodes)

    @property
    def success_rate(self) -> float:
        """The *expert's* success rate, over attempts. Not an eval number.

        core/eval.py reports policy success under the eval protocol. This one
        measures the data generator, and conflating the two is how a demo script
        becomes the thing that reports your results.
        """
        if not self.episodes:
            return 0.0
        return sum(e.success for e in self.episodes) / len(self.episodes)

    @property
    def total_frames(self) -> int:
        return sum(e.steps for e in self.episodes if e.kept)

    @property
    def failure_histogram(self) -> dict[str, int]:
        counts = Counter(
            str(e.failure_mode) for e in self.episodes if not e.success
        )
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "root": str(self.root),
            "task_prompt": self.task_prompt,
            "perturbation": self.perturbation,
            "attempted": self.attempted,
            "kept": self.kept,
            "total_frames": self.total_frames,
            "expert_success_rate": round(self.success_rate, 4),
            "failure_histogram": self.failure_histogram,
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path


def rollout(
    env: Env,
    policy: Policy,
    *,
    seed: int,
    max_steps: int | None = None,
    perturb: Perturb | None = None,
) -> Episode:
    """Drive one episode to its own end and buffer it.

    Stops when the environment says the episode is over -- success, termination
    or truncation -- and never on a step budget of the recorder's own. spec.md
    bans time features from observations precisely because fixed-length episodes
    make "where am I in the episode" predict the action; a recorder that cut
    every episode at the same step would reintroduce that through the back door.

    With ``perturb``, ``env.step`` receives ``perturb(action, rng)`` and the
    episode still records ``action``. See the module docstring.
    """
    limit = RUNAWAY_STEPS if max_steps is None else max_steps
    observation = env.reset(seed=seed)
    policy.reset()
    rng = np.random.default_rng(seed)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    images: list[dict[str, np.ndarray]] = []
    success, failure_mode = False, FailureMode.TIMEOUT

    for _ in range(limit):
        action = np.asarray(policy(observation), dtype=np.float32)
        # Recorded before stepping: this is the pair (what the policy saw, what
        # it did about it). Copied because a backend is free to reuse its render
        # buffer, and a dataset of N identical final frames is a bug that only
        # shows up after the recording has finished.
        states.append(np.array(observation.state, dtype=np.float32, copy=True))
        actions.append(action.copy())
        images.append({k: np.array(v, copy=True) for k, v in observation.images.items()})

        result = env.step(action if perturb is None else perturb(action, rng))
        observation = result.observation
        success, failure_mode = result.success, result.failure_mode
        if result.terminated or result.truncated:
            break
    else:
        if max_steps is None:
            raise RuntimeError(
                f"episode ran {limit} steps without terminating or truncating; "
                "the environment is not enforcing a horizon"
            )

    return Episode(
        seed=seed,
        states=np.stack(states),
        actions=np.stack(actions),
        images=images,
        success=success,
        failure_mode=FailureMode.NONE if success else failure_mode,
    )


def check_tracking(
    episode: Episode, embodiment: Embodiment, *, tolerance: float = TRACKING_TOLERANCE
) -> tuple[float, float]:
    """Recorder integrity check. Raises if the two columns have come apart.

    Run on the arrays that are about to be written, not on anything held
    separately, so it checks the dataset rather than a parallel copy of it.

    Under ``perturb`` that means the clean labels, not the executed actions, and
    it stays tight: a closed-loop expert re-solves from wherever the noise left
    the arm, so its label still lands near the next state. Measured on robosuite
    pick_place_cube at sigma 0.05: clean-label max 0.069 rad, executed-action
    max 0.250 -- the check would sit right on its limit had it read the latter.
    """
    mean, maximum = joint_tracking_error(episode.actions, episode.states, embodiment)
    if maximum > tolerance:
        raise ValueError(
            f"seed {episode.seed}: joint tracking error max {maximum:.4f} rad exceeds "
            f"{tolerance} rad. action[t] and state[t+1] are the same quantity under "
            "spec.md's absolute-joint-target contract, so a gap this size means the "
            "controller is not in absolute mode, or the recorder is writing "
            "misaligned columns."
        )
    return mean, maximum


def record_episodes(
    env: Env,
    policy: Policy,
    *,
    repo_id: str,
    root: Path,
    task_prompt: str,
    episodes: int,
    robot_type: str,
    cameras: Sequence[str] | None = None,
    seed_start: int = 0,
    max_attempts: int | None = None,
    keep_failures: bool = False,
    tracking_tolerance: float = TRACKING_TOLERANCE,
    perturb: Perturb | None = None,
) -> RecordingSummary:
    """Record until ``episodes`` episodes have been kept, and write the dataset.

    Args:
        task_prompt: the natural-language instruction, stored on every frame.
            Required, not optional: ROADMAP wants the task suite
            language-annotated from day one, because a VLA has nothing to
            condition on otherwise and retrofitting the annotation means
            re-recording everything.
        episodes: how many *kept* episodes are wanted. Seeds are consumed in
            order from ``seed_start`` and failures are skipped, so the seed set
            depends on the expert's success rate -- which is why every attempt,
            kept or not, lands in the returned summary.
        cameras: which canonical cameras to record. Defaults to whatever the
            first observation carries.
        perturb: noise on the executed action; the dataset keeps the clean one.
            Described in the summary by its ``to_dict()``, or its ``repr`` if
            it has none -- never by nothing, which would read as a clean dataset.

    Returns the summary, and also writes it to ``root/recording_summary.json``.
    """
    if episodes < 1:
        raise ValueError(f"episodes must be >= 1, got {episodes}")
    if not task_prompt.strip():
        raise ValueError("task_prompt must be a non-empty instruction")
    root = Path(root)
    limit = max_attempts if max_attempts is not None else 2 * episodes + 10

    dataset = None
    camera_names: list[str] = list(cameras) if cameras else []
    records: list[EpisodeRecord] = []
    kept = 0

    for attempt in range(limit):
        if kept >= episodes:
            break
        episode = rollout(env, policy, seed=seed_start + attempt, perturb=perturb)
        mean, maximum = check_tracking(episode, env.embodiment, tolerance=tracking_tolerance)
        keep = episode.success or keep_failures

        if keep:
            if dataset is None:
                # Fixed on the first kept episode and reused verbatim after
                # that: the feature set is a property of the dataset, so a task
                # that started reporting a third camera midway through must fail
                # in add_frame rather than write two incompatible halves.
                camera_names = camera_names or sorted(episode.images[0])
                sample = episode.images[0][camera_names[0]]
                dataset = create_dataset(
                    repo_id=repo_id,
                    root=root,
                    fps=env.control_hz,
                    embodiment=env.embodiment,
                    cameras=camera_names,
                    image_hw=(sample.shape[0], sample.shape[1]),
                    robot_type=robot_type,
                )
            _write_episode(dataset, episode, task_prompt, camera_names)
            kept += 1

        records.append(
            EpisodeRecord(
                seed=episode.seed,
                steps=episode.steps,
                success=episode.success,
                failure_mode=episode.failure_mode,
                kept=keep,
                tracking_error_mean=mean,
                tracking_error_max=maximum,
            )
        )

    if dataset is None:
        raise RuntimeError(
            f"no episode was kept in {len(records)} attempts; nothing was written. "
            "Check the expert against demo.py before recording."
        )
    dataset.finalize()

    summary = RecordingSummary(
        repo_id=repo_id,
        root=root,
        task_prompt=task_prompt,
        episodes=tuple(records),
        perturbation=_describe(perturb),
    )
    if kept < episodes:
        # Not an error: a short dataset that says so beats a run that silently
        # spins through seeds until it finds enough of them.
        print(
            f"warning: kept {kept} of {episodes} requested episodes in {limit} attempts "
            f"(expert success rate {summary.success_rate:.0%})"
        )
    summary.write(root / "recording_summary.json")
    return summary


def _describe(perturb: Perturb | None) -> dict | None:
    """Provenance for the summary. None means clean, so only clean may produce it."""
    if perturb is None:
        return None
    if hasattr(perturb, "to_dict"):
        return perturb.to_dict()
    return {"type": "custom", "repr": repr(perturb)}


def _write_episode(dataset, episode: Episode, task_prompt: str, cameras: Sequence[str]) -> None:
    """Append one buffered episode, frame by frame.

    ``parallel_encoding=False`` is deliberate. LeRobot's default farms video
    encoding to a ``ProcessPoolExecutor``, and Python's forkserver re-imports
    the parent's ``__main__`` in each child -- so an entrypoint without an
    ``if __name__ == "__main__"`` guard re-runs the whole recording recursively,
    and under pytest the children re-import the test session. Serial encoding on
    a ~215-step episode costs well under a second and removes the failure mode.
    """
    for t in range(episode.steps):
        frame = {
            STATE_FEATURE: episode.states[t],
            ACTION_FEATURE: episode.actions[t],
            "task": task_prompt,
        }
        for camera in cameras:
            frame[image_feature(camera)] = episode.images[t][camera]
        dataset.add_frame(frame)
    dataset.save_episode(parallel_encoding=False)
