import numpy as np
import pytest

from dream_robot.core.dataset import (
    ACTION_FEATURE,
    STATE_FEATURE,
    dataset_features,
    frame_to_uint8,
    image_feature,
    vector_names,
)
from dream_robot.core.schema import PANDA, Embodiment

BIMANUAL = Embodiment(arm_joints=14, grippers=2)


# --- feature names ----------------------------------------------------------

def test_image_feature_rejects_backend_camera_names():
    """The rename happens in the sim adapter; by here it is already too late."""
    with pytest.raises(ValueError, match="non-canonical camera"):
        image_feature("agentview")


def test_vector_names_are_indexed_uniformly():
    # gripper_0 even with one gripper: one naming rule for a policy to parse,
    # not a special case that only surfaces on bimanual data.
    assert vector_names(PANDA) == [f"joint_{i}" for i in range(7)] + ["gripper_0"]
    assert len(vector_names(BIMANUAL)) == BIMANUAL.dim == 16


# --- the feature dict -------------------------------------------------------

def test_state_and_action_share_shape_and_names():
    """spec.md makes action[t] and state[t+1] the same quantity in the same units."""
    features = dataset_features(embodiment=PANDA, cameras=["top"], image_hw=(128, 128))
    state, action = features[STATE_FEATURE], features[ACTION_FEATURE]
    assert state["shape"] == action["shape"] == (8,)
    assert state["names"] == action["names"]
    assert state["dtype"] == action["dtype"] == "float32"


def test_dimensions_follow_the_embodiment():
    """Bimanual is a bigger number here, not a second copy of this module."""
    features = dataset_features(embodiment=BIMANUAL, cameras=["top"], image_hw=(96, 96))
    assert features[STATE_FEATURE]["shape"] == (16,)
    assert features[ACTION_FEATURE]["shape"] == (16,)


def test_images_are_declared_channel_first_video():
    features = dataset_features(
        embodiment=PANDA, cameras=["top", "wrist"], image_hw=(128, 96)
    )
    top = features["observation.images.top"]
    assert top["dtype"] == "video"
    assert top["shape"] == (3, 128, 96)          # (C, H, W), height before width
    assert set(features) == {
        STATE_FEATURE, ACTION_FEATURE,
        "observation.images.top", "observation.images.wrist",
    }


def test_a_dataset_needs_at_least_one_camera():
    """State-only is a pipeline sanity check, never a recorded dataset."""
    with pytest.raises(ValueError, match="at least one camera"):
        dataset_features(embodiment=PANDA, cameras=[], image_hw=(128, 128))


# --- reading frames back ----------------------------------------------------

def test_frame_to_uint8_converts_channel_first_floats():
    """LeRobot hands back (3, H, W) float32 in [0, 1]; the repo speaks (H, W, 3) uint8."""
    chw = np.zeros((3, 4, 5), dtype=np.float32)
    chw[0] = 1.0
    out = frame_to_uint8(chw)
    assert out.shape == (4, 5, 3) and out.dtype == np.uint8
    assert np.array_equal(out[..., 0], np.full((4, 5), 255, dtype=np.uint8))
    assert out[..., 1].max() == 0


def test_frame_to_uint8_passes_through_hwc_uint8():
    img = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(frame_to_uint8(img), img)


def test_frame_to_uint8_keeps_a_three_wide_image_channel_last():
    """A (3, H, 3) frame is ambiguous; channel-last wins because that is the contract."""
    hwc = np.zeros((3, 3, 3), dtype=np.uint8)
    assert frame_to_uint8(hwc).shape == (3, 3, 3)


def test_frames_too_small_for_the_encoder_are_refused():
    """Below MIN_IMAGE_SIDE the AV1 encoder aborts the process; catch it early."""
    with pytest.raises(ValueError, match="crashes the process"):
        dataset_features(embodiment=PANDA, cameras=["top"], image_hw=(16, 16))
