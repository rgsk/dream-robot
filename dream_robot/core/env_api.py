"""The online interface: what eval and RL need from a task. Nothing else does.

ROADMAP's seam between a policy and a simulator is a **dataset on disk**, not
this protocol. Training never touches an Env; only rollouts do. Keeping that
straight is what stops this file from growing into a second, competing seam --
if a policy ever imports ``Env``, something has gone wrong.

Like schema.py, this module imports numpy and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

from dream_robot.core.schema import (
    CANONICAL_CAMERAS,
    Embodiment,
    Observation,
    validate_action,
    validate_images,
)


class FailureMode(StrEnum):
    """The failure-mode histogram from ROADMAP, as a closed set.

    Closed on purpose. The histogram is only comparable across policies and
    across simulators if every task reports into the same buckets; a free-text
    reason string would produce a different vocabulary per task and a histogram
    that cannot be summed. A task that genuinely needs a new bucket adds it
    here, once, for everyone.
    """

    NONE = "none"                    # succeeded
    NO_GRASP = "no_grasp"            # never got hold of the object
    DROPPED = "dropped"              # had it, lost it in transit
    WRONG_TARGET = "wrong_target"    # placed, but not where it was asked
    KNOCKED_OVER = "knocked_over"    # disturbed the scene
    TIMEOUT = "timeout"              # ran out of steps mid-task


@dataclass(frozen=True)
class StepResult:
    """One environment transition.

    Note what is absent.

    **No ``info`` dict.** schema.py refuses to expose a channel through which a
    backend's raw observation could reach a policy; an untyped ``info`` mapping
    would reintroduce exactly that channel one level up, and "just for
    debugging" is how ground truth ends up in a dataset.

    **No ``reward``.** BC and the eval harness do not consume one, and a field
    nobody reads gets filled with a shaped, task-specific quantity that then
    differs across simulators and quietly becomes part of the comparison. The
    RL rung adds reward as a task-specific extension when it actually needs it.
    """

    observation: Observation
    success: bool
    terminated: bool                              # reached a terminal state
    truncated: bool                               # hit the step limit
    failure_mode: FailureMode = FailureMode.NONE

    def __post_init__(self) -> None:
        if self.success and self.failure_mode is not FailureMode.NONE:
            raise ValueError(
                f"success=True with failure_mode={self.failure_mode!r}; "
                "a successful step has no failure mode"
            )
        if self.terminated and self.truncated:
            raise ValueError("terminated and truncated are mutually exclusive")


@runtime_checkable
class Env(Protocol):
    """What a task must offer for rollouts. Implemented by sim adapters only.

    Deliberately a Protocol rather than a base class: a task satisfies this by
    having the right methods, not by inheriting. Sim adapters may import core,
    but nothing forces them into a class hierarchy that a second backend would
    then have to contort itself to fit.
    """

    @property
    def embodiment(self) -> Embodiment:
        """State/action dimensions. Read from here, never hardcoded."""

    @property
    def control_hz(self) -> float:
        """Control rate. spec.md freezes this at 30 for every task."""

    def reset(self, *, seed: int | None = None) -> Observation:
        """Start an episode. ``seed`` makes the episode reproducible."""

    def step(self, action: np.ndarray) -> StepResult:
        """Apply absolute joint targets for one control step."""

    def render(self) -> np.ndarray:
        """A uint8 (H, W, 3) frame for video.

        Separate from ``observation.images`` on purpose: video wants a
        legible wide view, while the observation cameras are pinned to the
        dataset's contract resolution. Coupling them means either an ugly
        video or a dataset resized to look good.
        """

    def close(self) -> None: ...


def check_env_conformance(env: Env, *, steps: int = 3, seed: int = 0) -> None:
    """Run a task against the contract and raise on the first violation.

    Every sim adapter runs this in its own test. It is what makes "honours the
    contract" a checked property of a backend rather than a claim in its README.
    """
    if not isinstance(env.embodiment, Embodiment):
        raise TypeError(f"env.embodiment must be an Embodiment, got {type(env.embodiment)}")
    if float(env.control_hz) != 30.0:
        raise ValueError(
            f"control_hz is {env.control_hz}; spec.md freezes every task at 30 Hz, "
            "and changing it invalidates every cycle-time number in the matrix"
        )

    obs = env.reset(seed=seed)
    _check_observation(obs, env.embodiment)

    frame = np.asarray(env.render())
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"render() must return uint8 (H, W, 3), got {frame.dtype} {frame.shape}"
        )

    for i in range(steps):
        action = np.zeros(env.embodiment.dim, dtype=np.float32)
        result = env.step(validate_action(action, env.embodiment))
        if not isinstance(result, StepResult):
            raise TypeError(f"step() must return StepResult, got {type(result)} at step {i}")
        _check_observation(result.observation, env.embodiment)


def _check_observation(obs: Observation, embodiment: Embodiment) -> None:
    if not isinstance(obs, Observation):
        raise TypeError(f"expected Observation, got {type(obs)}")
    state = np.asarray(obs.state)
    if state.dtype != np.float32:
        raise ValueError(f"observation.state must be float32, got {state.dtype}")
    if state.shape != (embodiment.dim,):
        raise ValueError(
            f"observation.state must have shape ({embodiment.dim},), got {state.shape}"
        )
    # Re-run the camera checks: an adapter that builds an Observation directly,
    # bypassing build_observation, would otherwise slip a backend camera name in.
    validate_images(obs.images)
    if not set(obs.images) <= set(CANONICAL_CAMERAS):
        raise ValueError(f"non-canonical cameras: {sorted(set(obs.images))}")
