"""The seam, as a format: what a dataset on disk looks like, and how to read it back.

ROADMAP's one rule is that a policy and a simulator meet at a directory, never
at an import. This module is that directory's schema. ``record.py`` is the only
thing that writes one; every policy reads one and never learns which simulator
produced it.

**This is the first ``core`` module with a heavy dependency.** ``schema.py`` and
``env_api.py`` import numpy and nothing else, on purpose -- they are what a sim
adapter is allowed to import, and dragging torch into Isaac's venv to describe
an observation would be absurd. Here the dependency is the point: the format
*is* LeRobotDataset, so the module that defines the format imports it.

Two things are decided here rather than at each call site.

1. **Feature names and dtypes.** ``observation.state`` / ``action`` /
   ``observation.images.<canonical camera>``, derived from an ``Embodiment``
   rather than written out, so a bimanual task is a larger ``dim`` and not a
   second copy of this file (ROADMAP rule 2).

2. **Images are stored as video, not PNG frames.** A 215-step episode is ~21 MB
   of raw uint8 across two cameras; AV1 takes it to a few hundred kB. At the
   hundreds-of-episodes scale this ladder needs, PNG frames would make the
   dataset the largest thing in the project for no gain -- the decoder hands
   back the same pixels either way.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dream_robot.core.schema import CANONICAL_CAMERAS, Embodiment

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
IMAGE_PREFIX = "observation.images."

#: Smallest camera side this format will accept, in pixels.
#:
#: Not a taste call. SVT-AV1 dies on very small frames with SIGFPE -- a hard
#: crash inside the encoder, no Python traceback, no partial dataset -- and it
#: happens at ``save_episode``, which is minutes into a recording rather than at
#: the top. Measured on SVT-AV1 3.0.0: 24x24 crashes, 28x28 encodes. 32 is the
#: nearest power of two clear of that, and far below any resolution a policy
#: would actually train on, so the guard costs nothing and converts a core dump
#: into a sentence.
MIN_IMAGE_SIDE = 32


def image_feature(camera: str) -> str:
    """Canonical camera name -> its dataset feature key."""
    if camera not in CANONICAL_CAMERAS:
        raise ValueError(
            f"non-canonical camera {camera!r}; expected one of {list(CANONICAL_CAMERAS)}. "
            "Backend names are renamed in the sim adapter, never here."
        )
    return f"{IMAGE_PREFIX}{camera}"


def vector_names(embodiment: Embodiment) -> list[str]:
    """Per-element labels for the state and action vectors.

    Indexed uniformly -- ``gripper_0`` even when there is one gripper -- so that
    a policy reading dataset metadata parses one naming rule rather than a
    special case that only shows up on bimanual data.
    """
    return (
        [f"joint_{i}" for i in range(embodiment.arm_joints)]
        + [f"gripper_{i}" for i in range(embodiment.grippers)]
    )


def embodiment_from_names(names: Sequence[str]) -> Embodiment:
    """The inverse of ``vector_names``: recover the arm/gripper split from metadata.

    This is what ROADMAP rule 2 means in practice. A policy needs to know which
    entries of an 8-vector are joints and which is a gripper -- to clip the
    gripper to [0, 1] and leave the joints alone, say -- and assuming "the last
    one" is right for a Panda and wrong for a bimanual arm with two grippers,
    silently. The dataset already recorded the answer.
    """
    joints = [n for n in names if n.startswith("joint_")]
    grippers = [n for n in names if n.startswith("gripper_")]
    if len(joints) + len(grippers) != len(names):
        unknown = sorted(set(names) - set(joints) - set(grippers))
        raise ValueError(
            f"unrecognised vector element name(s) {unknown}; this repo's datasets "
            "name them joint_<i> / gripper_<i> (see vector_names)"
        )
    return Embodiment(arm_joints=len(joints), grippers=len(grippers))


def dataset_features(
    *,
    embodiment: Embodiment,
    cameras: Sequence[str],
    image_hw: tuple[int, int],
) -> dict[str, dict]:
    """The full feature dict for ``LeRobotDataset.create``.

    ``observation.state`` and ``action`` share a shape and a set of names on
    purpose: spec.md makes the action absolute joint targets, so ``action[t]``
    and ``state[t+1]`` are the same quantity in the same units. A policy can
    therefore be initialised to predict its own input and still emit a
    physically legal command, and a misaligned recorder shows up as a diff
    between two columns instead of as a bad success rate three stages later.

    Image shapes are declared channel-first ``(3, H, W)`` because that is what
    LeRobot's metadata expects and what its loader returns. Frames handed to
    ``add_frame`` stay ``(H, W, 3)`` uint8 -- it accepts either layout, and
    keeping the writer in the same layout as ``Observation.images`` means no
    transpose sits between the simulator and the disk where it could be
    silently wrong.
    """
    if not cameras:
        raise ValueError(
            "a dataset needs at least one camera; state-only is a pipeline "
            "sanity check, never a recorded dataset (spec.md)"
        )
    height, width = image_hw
    if min(height, width) < MIN_IMAGE_SIDE:
        raise ValueError(
            f"camera frames are {height}x{width}; the video encoder needs both sides "
            f">= {MIN_IMAGE_SIDE}px and crashes the process outright below that"
        )
    names = vector_names(embodiment)
    features: dict[str, dict] = {
        STATE_FEATURE: {
            "dtype": "float32",
            "shape": (embodiment.dim,),
            "names": names,
        },
        ACTION_FEATURE: {
            "dtype": "float32",
            "shape": (embodiment.dim,),
            "names": names,
        },
    }
    for camera in cameras:
        features[image_feature(camera)] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }
    return features


def create_dataset(
    *,
    repo_id: str,
    root: Path,
    fps: float,
    embodiment: Embodiment,
    cameras: Sequence[str],
    image_hw: tuple[int, int],
    robot_type: str,
) -> LeRobotDataset:
    """A fresh, empty dataset ready for ``add_frame``.

    ``fps`` is the task's control rate, not a taste choice: spec.md freezes
    every task at 30 Hz and records one frame per control step with no
    subsampling, so the dataset's fps and the environment's ``control_hz`` are
    the same number by construction. LeRobot wants an int; a task whose control
    rate is not a whole number of hertz would silently round, so refuse it.
    """
    if float(fps) != int(fps):
        raise ValueError(f"fps must be a whole number of hertz, got {fps}")
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=dataset_features(
            embodiment=embodiment, cameras=cameras, image_hw=image_hw
        ),
        root=root,
        robot_type=robot_type,
        use_videos=True,
    )


def open_dataset(repo_id: str, root: Path, *, episode: int | None = None) -> LeRobotDataset:
    """Read a recorded dataset back. Optionally just one episode."""
    return LeRobotDataset(
        repo_id, root=root, episodes=None if episode is None else [episode]
    )


def frame_to_uint8(frame) -> np.ndarray:
    """LeRobot's loader output -> the ``(H, W, 3)`` uint8 the rest of the repo uses.

    The loader returns ``(3, H, W)`` float32 in [0, 1]: the normalisation a
    policy wants, and the wrong thing for a video writer or for comparing
    against what the simulator produced. Converting here, once, keeps a stray
    ``* 255`` out of every read site.

    Round-tripping is lossy in both directions -- 8-bit quantisation on the way
    back, and AV1 is lossy on the way in -- so pixels read out will not equal
    pixels put in. Anything checking the round trip checks shape, dtype and
    gross similarity, never equality.
    """
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3-D frame, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray((np.clip(arr, 0.0, 1.0) * 255.0).round().astype(np.uint8))


def frame_to_float_chw(frame) -> np.ndarray:
    """``(H, W, 3)`` uint8 -> the ``(3, H, W)`` float32 in [0, 1] a model is fed.

    The exact inverse of what LeRobot's loader already does, and it exists so
    that both halves of a policy's life go through one function. Training reads
    frames from the dataset, already channel-first floats; a policy running in a
    simulator gets ``Observation.images``, which is channel-last uint8. If those
    two paths disagree -- a missing divide by 255, a transpose in one and not
    the other -- the policy trains fine and then sees different pixels at
    rollout, which looks like a policy that does not generalise rather than like
    a bug.
    """
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {arr.dtype}")
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0)


def episode_frames(
    dataset: LeRobotDataset, camera: str
) -> list[np.ndarray]:
    """Every frame of one camera, in order, as ``(H, W, 3)`` uint8.

    Expects a dataset opened with ``episode=`` -- LeRobot renumbers a
    single-episode view to indices ``0..len-1``, so iteration is the whole of it.
    """
    key = image_feature(camera)
    return [frame_to_uint8(dataset[i][key]) for i in range(len(dataset))]
