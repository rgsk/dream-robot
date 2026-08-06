"""Canonical observation and action types -- the executable form of spec.md.

This module imports numpy and nothing else. It may never import a simulator: it
is the thing a recorder and a policy agree on, and if it reached into robosuite
the venv split that lets Isaac coexist would be impossible.

Two ideas carry most of the weight here.

1. **Construction, not filtering.** ``build_state`` takes named numeric
   arguments. It has no parameter through which a backend's raw observation dict
   could arrive, so there is no path by which ground-truth object pose can reach
   ``observation.*``. That is what spec.md means by "enforced structurally".

2. **Dimensions are data.** ``Embodiment`` is a value, not a constant. Bimanual
   is a 16-dim action, not a rewrite (ROADMAP rule 2).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

# Every backend names its cameras differently -- agentview, robot0_eye_in_hand,
# cam_high, top. Adapters normalise to these, and nothing downstream ever sees a
# backend's own name. Silent cross-sim breakage lives here if it lives anywhere.
CANONICAL_CAMERAS: tuple[str, ...] = ("top", "wrist")

STATE_DTYPE = np.float32
ACTION_DTYPE = np.float32
IMAGE_DTYPE = np.uint8

# observation.state[gripper] and action[gripper]: 0.0 closed, 1.0 open.
GRIPPER_MIN, GRIPPER_MAX = 0.0, 1.0


@dataclass(frozen=True)
class Embodiment:
    """The shape of one robot's state and action vectors.

    ``arm_joints`` counts every arm joint across every arm: a bimanual 7-DoF
    setup is ``Embodiment(arm_joints=14, grippers=2)`` and yields ``dim == 16``.
    Policies read this from dataset metadata; recorders get it from the task.
    """

    arm_joints: int
    grippers: int = 1

    def __post_init__(self) -> None:
        if self.arm_joints < 1:
            raise ValueError(f"arm_joints must be >= 1, got {self.arm_joints}")
        if self.grippers < 1:
            raise ValueError(f"grippers must be >= 1, got {self.grippers}")

    @property
    def dim(self) -> int:
        return self.arm_joints + self.grippers

    @property
    def arm_slice(self) -> slice:
        return slice(0, self.arm_joints)

    @property
    def gripper_slice(self) -> slice:
        return slice(self.arm_joints, self.dim)


#: robosuite Panda: 7 arm joints + 1 gripper -> dim 8, spec.md's float32[8].
PANDA = Embodiment(arm_joints=7, grippers=1)


@dataclass(frozen=True)
class Observation:
    """What a policy is allowed to see. Nothing else exists."""

    state: np.ndarray                    # float32 (dim,)
    images: Mapping[str, np.ndarray]     # canonical name -> uint8 (H, W, 3)


def _as_1d(name: str, values, dtype) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: {arr}")
    return arr


def build_state(
    *,
    joint_positions,
    gripper_openings,
    embodiment: Embodiment,
) -> np.ndarray:
    """Assemble ``observation.state`` = [joint positions (rad) | gripper opening].

    Args:
        joint_positions: arm joint angles in radians, base to wrist, length
            ``embodiment.arm_joints``.
        gripper_openings: normalised opening per gripper in [0, 1], 0 closed.
            One scalar per gripper even on hardware with two mirrored finger
            joints -- otherwise the action dimension becomes a property of the
            simulator's URDF rather than of the robot.

    Look at what this signature does *not* accept. There is no ``raw_obs``, no
    ``**extra``, no dict. A denylist would have to anticipate every name a
    backend invents for ground truth; robosuite alone offers ``cube_pos``,
    ``gripper_to_cube_pos`` and ``object-state``, and the last survives any
    filter keyed on the word "cube".
    """
    joints = _as_1d("joint_positions", joint_positions, STATE_DTYPE)
    grip = _as_1d("gripper_openings", gripper_openings, STATE_DTYPE)

    if joints.size != embodiment.arm_joints:
        raise ValueError(
            f"expected {embodiment.arm_joints} joint positions, got {joints.size}"
        )
    if grip.size != embodiment.grippers:
        raise ValueError(
            f"expected {embodiment.grippers} gripper openings, got {grip.size}"
        )
    if np.any(grip < GRIPPER_MIN) or np.any(grip > GRIPPER_MAX):
        raise ValueError(
            f"gripper_openings must lie in [{GRIPPER_MIN}, {GRIPPER_MAX}], got {grip}. "
            "Normalise in the adapter -- raw finger-joint positions are backend units."
        )
    return np.concatenate([joints, grip]).astype(STATE_DTYPE)


def validate_images(images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Check camera names are canonical and frames are uint8 (H, W, 3)."""
    if not images:
        raise ValueError(
            "at least one camera is required; state-only observations are a "
            "sanity check, never a recorded dataset (spec.md)"
        )
    unknown = sorted(set(images) - set(CANONICAL_CAMERAS))
    if unknown:
        raise ValueError(
            f"non-canonical camera name(s) {unknown}; expected a subset of "
            f"{list(CANONICAL_CAMERAS)}. Rename in the sim adapter, not downstream."
        )
    checked: dict[str, np.ndarray] = {}
    for name, frame in images.items():
        arr = np.asarray(frame)
        if arr.dtype != IMAGE_DTYPE:
            raise ValueError(
                f"camera {name!r}: expected {np.dtype(IMAGE_DTYPE).name}, "
                f"got {arr.dtype}. Images stay uint8 until the policy's transform; "
                "float frames in a dataset cost 4x the disk for no information."
            )
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"camera {name!r}: expected (H, W, 3), got {arr.shape}"
            )
        checked[name] = arr
    return checked


def build_observation(
    *,
    joint_positions,
    gripper_openings,
    images: Mapping[str, np.ndarray],
    embodiment: Embodiment,
) -> Observation:
    """The only supported way to make an Observation."""
    return Observation(
        state=build_state(
            joint_positions=joint_positions,
            gripper_openings=gripper_openings,
            embodiment=embodiment,
        ),
        images=validate_images(images),
    )


def validate_action(action, embodiment: Embodiment) -> np.ndarray:
    """Check an action is absolute joint targets of the right shape.

    Same layout and units as ``observation.state``: the gripper entry is a
    normalised command in [0, 1], not the +1/-1 open/close convention some
    simulators use. Adapters convert.
    """
    arr = _as_1d("action", action, ACTION_DTYPE)
    if arr.size != embodiment.dim:
        raise ValueError(f"expected action of dim {embodiment.dim}, got {arr.size}")
    grip = arr[embodiment.gripper_slice]
    if np.any(grip < GRIPPER_MIN) or np.any(grip > GRIPPER_MAX):
        raise ValueError(
            f"action gripper entries must lie in [{GRIPPER_MIN}, {GRIPPER_MAX}], got {grip}"
        )
    return arr


def joint_tracking_error(actions: np.ndarray, states: np.ndarray, embodiment: Embodiment):
    """Recorder integrity check: |action[t] - state[t+1]| over arm joints.

    Absolute joint targets make ``action[t]`` and ``state[t+1]`` the same
    quantity in the same units, so they can be diffed directly. A recorder that
    is writing the wrong column, an off-by-one in the step loop, or a controller
    running in delta mode all show up here as a number that will not settle --
    instead of as an unexplained success rate three steps downstream.

    Returns ``(mean, max)`` in radians. Expect a small non-zero value: the PD
    loop lags its commanded target (~0.02 rad at 30 Hz in the robosuite spike).
    """
    a = np.asarray(actions)[:-1, embodiment.arm_slice]
    s = np.asarray(states)[1:, embodiment.arm_slice]
    if a.shape != s.shape:
        raise ValueError(f"shape mismatch: actions {a.shape} vs states {s.shape}")
    err = np.abs(a - s)
    return float(err.mean()), float(err.max())
