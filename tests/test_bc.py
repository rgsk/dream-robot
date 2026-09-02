import numpy as np
import pytest
import torch

from dream_robot.core.dataset import (
    embodiment_from_names,
    frame_to_float_chw,
    frame_to_uint8,
    vector_names,
)
from dream_robot.core.schema import PANDA, Embodiment, build_observation
from dream_robot.policies.bc.data import split_episodes
from dream_robot.policies.bc.model import (
    BCPolicyNet,
    ModelConfig,
    NormalizationStats,
    Normalizer,
    SpatialSoftmax,
)
from dream_robot.policies.bc.policy import BCPolicy

SMALL = ModelConfig(
    state_dim=8, action_dim=8, arm_joints=7, grippers=1,
    cameras=("top", "wrist"), image_hw=(32, 32),
    conv_channels=(8, 16, 16), keypoints=8, hidden=(32, 32), dropout=0.0, groups=4,
)


def unit_stats(dim: int = 8) -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(dim, np.float32), state_std=np.ones(dim, np.float32),
        action_mean=np.zeros(dim, np.float32), action_std=np.ones(dim, np.float32),
        image_mean=np.full(3, 0.5, np.float32), image_std=np.full(3, 0.25, np.float32),
    )


# --- spatial softmax --------------------------------------------------------

def test_spatial_softmax_finds_the_spike():
    """A single hot pixel should come back as its own normalised coordinate."""
    layer = SpatialSoftmax(9, 9)
    features = torch.zeros(1, 1, 9, 9)
    features[0, 0, 0, 8] = 50.0            # top-right: row 0, column 8
    x, y = layer(features)[0].tolist()
    assert x == pytest.approx(1.0, abs=1e-3)      # column 8 of 8 -> u = +1
    assert y == pytest.approx(-1.0, abs=1e-3)     # row 0 -> v = -1


def test_spatial_softmax_averages_two_blobs_to_a_point_between_them():
    """The documented failure mode: an expectation over a bimodal map."""
    layer = SpatialSoftmax(9, 9)
    features = torch.zeros(1, 1, 9, 9)
    features[0, 0, 4, 0] = 50.0
    features[0, 0, 4, 8] = 50.0
    x, _ = layer(features)[0].tolist()
    assert x == pytest.approx(0.0, abs=1e-3)      # midpoint, where nothing is


def test_spatial_softmax_output_is_two_numbers_per_channel():
    out = SpatialSoftmax(4, 4)(torch.randn(3, 6, 4, 4))
    assert out.shape == (3, 12)
    assert out.abs().max() <= 1.0                 # coordinates stay in [-1, 1]


def test_spatial_softmax_gradient_flows_to_position():
    """Differentiable in position is the whole reason it is not an argmax."""
    layer = SpatialSoftmax(5, 5)
    features = torch.zeros(1, 1, 5, 5, requires_grad=True)
    layer(features).sum().backward()
    assert features.grad.abs().sum() > 0


# --- normalisation ----------------------------------------------------------

def test_normalizer_round_trips():
    norm = Normalizer(np.array([1.0, -2.0]), np.array([0.5, 4.0]))
    x = torch.tensor([[3.0, 6.0]])
    assert torch.allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-6)


def test_normalizer_floors_a_constant_dimension():
    """A joint that never moved would otherwise divide by ~0."""
    norm = Normalizer(np.zeros(2), np.array([1.0, 0.0]))
    assert float(norm.std[1]) > 0
    assert torch.isfinite(norm.normalize(torch.ones(1, 2))).all()


def test_normalization_constants_live_in_the_state_dict():
    """A checkpoint without them would load cleanly and predict nonsense."""
    net = BCPolicyNet(SMALL, unit_stats())
    keys = net.state_dict()
    assert "action_norm.mean" in keys and "image_norm.std" in keys


# --- the network ------------------------------------------------------------

def test_forward_shape_follows_the_config():
    net = BCPolicyNet(SMALL, unit_stats())
    images = {name: torch.rand(4, 3, 32, 32) for name in SMALL.cameras}
    assert net(images, torch.randn(4, 8)).shape == (4, 8)


def test_batch_size_does_not_change_the_answer():
    """The GroupNorm claim: a rollout at batch 1 sees the trained function."""
    net = BCPolicyNet(SMALL, unit_stats()).eval()
    images = {name: torch.rand(4, 3, 32, 32) for name in SMALL.cameras}
    state = torch.randn(4, 8)
    batched = net(images, state)
    single = net({k: v[:1] for k, v in images.items()}, state[:1])
    assert torch.allclose(batched[:1], single, atol=1e-6)


def test_a_missing_camera_is_an_error_not_a_silent_zero():
    net = BCPolicyNet(SMALL, unit_stats())
    with pytest.raises(ValueError, match="missing camera"):
        net({"top": torch.rand(1, 3, 32, 32)}, torch.randn(1, 8))


def test_dimensions_come_from_the_config_not_from_a_constant():
    """Bimanual: a bigger action_dim and no code change (ROADMAP rule 2)."""
    bimanual = ModelConfig(
        state_dim=16, action_dim=16, arm_joints=14, grippers=2,
        cameras=("top",), image_hw=(32, 32),
        conv_channels=(8, 16), keypoints=4, hidden=(16, 16), dropout=0.0, groups=4,
    )
    net = BCPolicyNet(bimanual, unit_stats(16))
    assert net({"top": torch.rand(2, 3, 32, 32)}, torch.randn(2, 16)).shape == (2, 16)
    assert bimanual.gripper_slice == slice(14, 16)


def test_gripper_slice_is_read_from_the_dataset_naming():
    assert embodiment_from_names(vector_names(PANDA)) == PANDA
    bimanual = Embodiment(arm_joints=14, grippers=2)
    assert embodiment_from_names(vector_names(bimanual)) == bimanual


# --- the episode split ------------------------------------------------------

def test_split_holds_out_whole_episodes_with_no_overlap():
    train, val = split_episodes(25, validation=5, seed=0)
    assert len(train) == 20 and len(val) == 5
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(25))


def test_split_is_reproducible_and_seed_dependent():
    assert split_episodes(25, validation=5, seed=0) == split_episodes(25, validation=5, seed=0)
    assert split_episodes(25, validation=5, seed=1) != split_episodes(25, validation=5, seed=0)


def test_split_refuses_to_leave_nothing_to_train_on():
    with pytest.raises(ValueError, match="nothing would be left"):
        split_episodes(5, validation=5)


# --- checkpoint and rollout path --------------------------------------------

def save_checkpoint(path, config=SMALL):
    net = BCPolicyNet(config, unit_stats(config.action_dim))
    torch.save(
        {
            "state_dict": net.state_dict(),
            "model_config": config.to_dict(),
            "normalization": unit_stats(config.action_dim).to_dict(),
            "dataset": {"repo_id": "x/y", "train_episodes": [0, 1]},
            "train_config": {}, "epoch": 3, "val_l1": 0.1, "val_l1_rad": 0.01,
        },
        path,
    )
    return net


def test_checkpoint_round_trips_to_the_same_prediction(tmp_path):
    path = tmp_path / "checkpoint.pt"
    net = save_checkpoint(path)
    loaded = BCPolicy.load(path, device="cpu")

    images = {name: torch.rand(1, 3, 32, 32) for name in SMALL.cameras}
    state = torch.randn(1, 8)
    expected = net.eval().predict(images, state)[0].numpy()
    actual = loaded._net.predict(images, state)[0].numpy()
    assert np.allclose(expected, actual, atol=1e-6)
    assert loaded.metadata["epoch"] == 3


def test_rollout_path_matches_the_training_path(tmp_path):
    """The bug this guards against has no error message and no bad loss.

    Training reads channel-first floats out of the dataset; a rollout gets
    channel-last uint8 out of an Observation. If those disagree -- a missing
    /255, a transpose applied on one side only -- validation looks excellent and
    the policy fails in the simulator, which reads as "BC does not generalise".
    Measured on the real checkpoint, the arm joints agree exactly.
    """
    path = tmp_path / "checkpoint.pt"
    net = save_checkpoint(path).eval()
    policy = BCPolicy.load(path, device="cpu")

    rng = np.random.default_rng(0)
    frames = {
        name: rng.integers(0, 255, (32, 32, 3), dtype=np.uint8) for name in SMALL.cameras
    }
    state = np.concatenate([rng.normal(size=7), [0.5]]).astype(np.float32)
    observation = build_observation(
        joint_positions=state[:7], gripper_openings=[state[7]],
        images=frames, embodiment=PANDA,
    )

    via_training_path = net.predict(
        {n: torch.from_numpy(frame_to_float_chw(f)).unsqueeze(0) for n, f in frames.items()},
        torch.from_numpy(state).unsqueeze(0),
    )[0].numpy()
    via_rollout_path = policy(observation)
    # Arm joints must match bit for bit; only the gripper is deliberately clipped.
    assert np.array_equal(via_training_path[:7], via_rollout_path[:7])


def test_frame_conversion_is_exact_for_decoded_video():
    """Dataset floats are k/255 exactly, so the uint8 round trip loses nothing."""
    original = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(frame_to_uint8(frame_to_float_chw(original)), original)


def test_policy_clips_the_gripper_into_the_contract(tmp_path):
    """A regression head overshoots [0, 1]; validate_action refuses that."""
    path = tmp_path / "checkpoint.pt"
    net = save_checkpoint(path)
    # Force a large positive bias on the final layer so the raw output leaves [0, 1].
    with torch.no_grad():
        net.head[-1].bias.fill_(5.0)
    torch.save(
        {
            "state_dict": net.state_dict(), "model_config": SMALL.to_dict(),
            "normalization": unit_stats().to_dict(), "dataset": {}, "train_config": {},
        },
        path,
    )
    policy = BCPolicy.load(path, device="cpu")
    frames = {n: np.zeros((32, 32, 3), np.uint8) for n in SMALL.cameras}
    observation = build_observation(
        joint_positions=np.zeros(7), gripper_openings=[0.5],
        images=frames, embodiment=PANDA,
    )
    action = policy(observation)
    assert 0.0 <= action[7] <= 1.0
    assert action[0] > 1.0          # arm joints are deliberately NOT clipped


def test_policy_rejects_an_environment_missing_its_camera(tmp_path):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path)
    policy = BCPolicy.load(path, device="cpu")
    observation = build_observation(
        joint_positions=np.zeros(7), gripper_openings=[0.5],
        images={"top": np.zeros((32, 32, 3), np.uint8)}, embodiment=PANDA,
    )
    with pytest.raises(ValueError, match="trained on camera"):
        policy(observation)
