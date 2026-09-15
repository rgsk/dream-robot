import numpy as np
import pytest
import torch
from fakes import FAKE_EMBODIMENT, FakeEnv, FakePolicy

from dream_robot.core.record import record_episodes
from dream_robot.core.registry import POLICIES
from dream_robot.core.schema import build_observation
from dream_robot.policies.act.data import ChunkDataset, compute_stats
from dream_robot.policies.act.model import ACTConfig, ACTPolicyNet, kl_divergence, masked_l1
from dream_robot.policies.act.policy import ACTPolicy
from dream_robot.policies.bc.model import NormalizationStats

SMALL = ACTConfig(
    state_dim=8,
    action_dim=8,
    arm_joints=7,
    grippers=1,
    cameras=("top", "wrist"),
    image_hw=(32, 32),
    chunk=6,
    dim=32,
    heads=4,
    feedforward=64,
    encoder_layers=1,
    decoder_layers=1,
    vae_layers=1,
    latent=4,
    conv_channels=(8, 16, 16),
    groups=4,
    dropout=0.0,
)


def unit_stats(dim: int = 8) -> NormalizationStats:
    return NormalizationStats(
        state_mean=np.zeros(dim, np.float32),
        state_std=np.ones(dim, np.float32),
        action_mean=np.zeros(dim, np.float32),
        action_std=np.ones(dim, np.float32),
        image_mean=np.full(3, 0.5, np.float32),
        image_std=np.full(3, 0.25, np.float32),
    )


def batch(n: int, config: ACTConfig = SMALL):
    torch.manual_seed(0)
    images = {c: torch.rand(n, 3, *config.image_hw) for c in config.cameras}
    return images, torch.randn(n, config.state_dim), torch.randn(n, config.chunk, config.action_dim)


# --- the network ------------------------------------------------------------


def test_predicts_a_whole_chunk():
    net = ACTPolicyNet(SMALL, unit_stats()).eval()
    images, state, _ = batch(3)
    chunk, mu, logvar = net(images, state)
    assert chunk.shape == (3, SMALL.chunk, SMALL.action_dim)
    assert mu is None and logvar is None  # rollout: no true chunk, latent at the prior


def test_training_forward_returns_the_latent_distribution():
    net = ACTPolicyNet(SMALL, unit_stats())
    images, state, actions = batch(3)
    chunk, mu, logvar = net(images, state, actions)
    assert chunk.shape == (3, SMALL.chunk, SMALL.action_dim)
    assert mu.shape == logvar.shape == (3, SMALL.latent)


def test_batch_size_does_not_change_the_rollout_answer():
    """GroupNorm and a zero latent: one observation gets the same plan alone or in a batch."""
    net = ACTPolicyNet(SMALL, unit_stats()).eval()
    images, state, _ = batch(4)
    together = net.predict(images, state)
    alone = net.predict({c: v[1:2] for c, v in images.items()}, state[1:2])
    assert torch.allclose(together[1:2], alone, atol=1e-5)


def test_padded_steps_are_invisible_to_the_latent_encoder():
    """Steps past the episode end are repeats; what they contain must not move the latent."""
    net = ACTPolicyNet(SMALL, unit_stats())
    images, state, actions = batch(2)
    pad = torch.zeros(2, SMALL.chunk, dtype=torch.bool)
    pad[:, 4:] = True
    changed = actions.clone()
    changed[:, 4:] += 100.0
    _, mu_a, _ = net(images, state, actions, pad)
    _, mu_b, _ = net(images, state, changed, pad)
    assert torch.allclose(mu_a, mu_b, atol=1e-5)


def test_masked_l1_ignores_padded_steps():
    predicted = torch.zeros(1, 4, 2)
    target = torch.tensor([[[1.0, 1.0], [1.0, 1.0], [50.0, 50.0], [50.0, 50.0]]])
    pad = torch.tensor([[False, False, True, True]])
    assert float(masked_l1(predicted, target, pad)) == pytest.approx(1.0)


def test_kl_is_zero_at_the_prior():
    assert float(kl_divergence(torch.zeros(3, 4), torch.zeros(3, 4))) == pytest.approx(0.0)


def test_config_round_trips():
    assert ACTConfig.from_dict(SMALL.to_dict()) == SMALL


# --- the chunk dataset ------------------------------------------------------


@pytest.fixture
def recorded(tmp_path):
    """Two 5-frame episodes from FakeEnv, written through the real recorder."""
    root = tmp_path / "ds"
    record_episodes(
        FakeEnv(),
        FakePolicy(),
        repo_id="dream_robot/fake",
        root=root,
        task_prompt="put the cube in the bin",
        episodes=2,
        robot_type="fake",
    )
    return root


def test_chunks_are_the_next_actions_and_stop_at_the_episode_end(recorded):
    data = ChunkDataset("dream_robot/fake", recorded, ("top", "wrist"), [0, 1], chunk=3)
    assert len(data) == 10
    # Row 1 of episode 0: rows 1, 2, 3, nothing padded.
    actions, pad = data.chunk_at(1)
    assert np.array_equal(actions, data.actions[1:4]) and not pad.any()
    # Row 4 is episode 0's last frame: one real step, two repeats of it, never episode 1's row 5.
    actions, pad = data.chunk_at(4)
    assert pad.tolist() == [False, True, True]
    assert np.array_equal(actions, np.repeat(data.actions[4:5], 3, axis=0))


def test_a_sample_is_images_state_chunk_and_pad(recorded):
    data = ChunkDataset("dream_robot/fake", recorded, ("top", "wrist"), [1], chunk=3)
    images, state, actions, pad = data[0]
    assert set(images) == {"top", "wrist"} and images["top"].shape == (3, 32, 32)
    assert state.shape == (FAKE_EMBODIMENT.dim,)
    assert actions.shape == (3, FAKE_EMBODIMENT.dim) and pad.shape == (3,)


def test_stats_come_from_the_chosen_episodes_only(recorded):
    data = ChunkDataset("dream_robot/fake", recorded, ("top", "wrist"), [0], chunk=3)
    stats = compute_stats(data, ("top", "wrist"), samples=5)
    assert np.allclose(stats.action_mean, data.actions[:5].mean(0), atol=1e-6)


# --- the rollout policy -----------------------------------------------------


class ScriptedNet(torch.nn.Module):
    """Stands in for ACTPolicyNet: plan k is filled with the value k, so ensembling is checkable."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        self.cameras = config.cameras
        self.calls = 0

    def predict(self, images, state):
        plan = torch.full((1, self.config.chunk, self.config.action_dim), float(self.calls))
        self.calls += 1
        return plan


def observation(config: ACTConfig = SMALL):
    return build_observation(
        joint_positions=np.zeros(config.arm_joints),
        gripper_openings=[0.5],
        images={c: np.zeros((*config.image_hw, 3), np.uint8) for c in config.cameras},
        embodiment=_panda_like(config),
    )


def _panda_like(config):
    from dream_robot.core.schema import Embodiment

    return Embodiment(arm_joints=config.arm_joints, grippers=config.grippers)


def test_temporal_ensembling_averages_every_plan_that_covers_this_step():
    policy = ACTPolicy(
        ScriptedNet(SMALL), device=torch.device("cpu"), metadata={}, ensemble_decay=0.0
    )
    first = policy(observation())
    second = policy(observation())
    third = policy(observation())
    assert first[0] == pytest.approx(0.0)  # only plan 0
    assert second[0] == pytest.approx(0.5)  # plans 0 and 1
    assert third[0] == pytest.approx(1.0)  # plans 0, 1, 2


def test_plans_older_than_a_chunk_drop_out():
    policy = ACTPolicy(
        ScriptedNet(SMALL), device=torch.device("cpu"), metadata={}, ensemble_decay=0.0
    )
    for _ in range(SMALL.chunk + 3):
        action = policy(observation())
    # Step chunk+2 is covered by plans chunk-3 .. chunk+2, i.e. the last `chunk` plans.
    last = SMALL.chunk + 2
    assert action[0] == pytest.approx(np.mean(range(last - SMALL.chunk + 1, last + 1)))


def test_decay_favours_older_plans():
    policy = ACTPolicy(
        ScriptedNet(SMALL), device=torch.device("cpu"), metadata={}, ensemble_decay=1.0
    )
    policy(observation())
    second = policy(observation())
    weights = np.exp(-np.arange(2))
    assert second[0] == pytest.approx((weights * [0, 1]).sum() / weights.sum())


def test_reset_forgets_the_previous_episode():
    policy = ACTPolicy(
        ScriptedNet(SMALL), device=torch.device("cpu"), metadata={}, ensemble_decay=0.0
    )
    for _ in range(4):
        policy(observation())
    policy.reset()
    assert policy(observation())[0] == pytest.approx(4.0)  # only the new plan, no old ones mixed in


def test_gripper_is_clipped_into_the_contract():
    net = ScriptedNet(SMALL)
    net.calls = 7  # every plan value 7, far outside [0, 1]
    action = ACTPolicy(net, device=torch.device("cpu"), metadata={})(observation())
    assert action[SMALL.gripper_slice][0] == 1.0
    assert action[0] == pytest.approx(7.0)  # arm joints are left alone


def test_checkpoint_round_trips_to_the_same_plan(tmp_path):
    torch.manual_seed(0)
    net = ACTPolicyNet(SMALL, unit_stats()).eval()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "state_dict": net.state_dict(),
            "model_config": SMALL.to_dict(),
            "normalization": unit_stats().to_dict(),
            "epoch": 1,
        },
        path,
    )
    loaded = ACTPolicy.load(path, device="cpu")
    images, state, _ = batch(1)
    assert torch.allclose(loaded._net.predict(images, state), net.predict(images, state), atol=1e-6)


def test_act_is_registered():
    assert "act" in POLICIES
