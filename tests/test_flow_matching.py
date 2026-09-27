import numpy as np
import pytest
import torch

from dream_robot.core.registry import POLICIES
from dream_robot.core.schema import Embodiment, build_observation
from dream_robot.policies.bc.model import NormalizationStats
from dream_robot.policies.flow_matching.model import FlowConfig, FlowPolicyNet, masked_mse
from dream_robot.policies.flow_matching.policy import FlowPolicy

SMALL = FlowConfig(
    state_dim=8,
    action_dim=8,
    arm_joints=7,
    grippers=1,
    cameras=("top", "wrist"),
    image_hw=(32, 32),
    chunk=6,
    dim=32,
    heads=4,
    layers=2,
    feedforward=64,
    conv_channels=(8, 16, 16),
    keypoints=8,
    groups=4,
    sample_steps=4,
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


def batch(n: int, config: FlowConfig = SMALL):
    torch.manual_seed(0)
    images = {c: torch.rand(n, 3, *config.image_hw) for c in config.cameras}
    return images, torch.randn(n, config.state_dim), torch.randn(n, config.chunk, config.action_dim)


def trained_ish(config: FlowConfig = SMALL) -> FlowPolicyNet:
    """A net with every weight random, so the zero-initialised output is not the identity."""
    torch.manual_seed(1)
    net = FlowPolicyNet(config, unit_stats())
    with torch.no_grad():
        for p in net.parameters():
            if p.abs().sum() == 0:
                p.normal_(0, 0.1)
    return net.eval()


# --- the network ------------------------------------------------------------


def test_velocity_has_the_chunk_shape():
    net = FlowPolicyNet(SMALL, unit_stats())
    images, state, actions = batch(3)
    v = net.velocity(actions, torch.rand(3), net.encode(images, state))
    assert v.shape == (3, SMALL.chunk, SMALL.action_dim)


def test_a_fresh_net_predicts_zero_velocity_so_sampling_returns_the_noise():
    """adaLN-zero and a zero output layer: at init the flow does nothing."""
    net = FlowPolicyNet(SMALL, unit_stats()).eval()
    images, state, _ = batch(2)
    plan = net.sample(images, state, generator=torch.Generator().manual_seed(0))
    noise = torch.randn(
        (2, SMALL.chunk, SMALL.action_dim), generator=torch.Generator().manual_seed(0)
    )
    assert torch.allclose(plan, noise, atol=1e-6)


def test_same_generator_seed_same_plan_different_seed_different_plan():
    net = trained_ish()
    images, state, _ = batch(1)
    a = net.sample(images, state, generator=torch.Generator().manual_seed(5))
    b = net.sample(images, state, generator=torch.Generator().manual_seed(5))
    c = net.sample(images, state, generator=torch.Generator().manual_seed(6))
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_loss_is_a_finite_scalar():
    net = trained_ish()
    images, state, actions = batch(2)
    pad = torch.zeros(2, SMALL.chunk, dtype=torch.bool)
    pad[:, 4:] = True
    loss = net.loss(images, state, actions, pad)
    assert loss.shape == () and torch.isfinite(loss)


def test_masked_mse_averages_real_steps_only():
    predicted = torch.zeros(1, 4, 2)
    target = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [50.0, 50.0], [50.0, 50.0]]])
    pad = torch.tensor([[False, False, True, True]])
    assert float(masked_mse(predicted, target, pad)) == pytest.approx(2.5)


def test_config_round_trips():
    assert FlowConfig.from_dict(SMALL.to_dict()) == SMALL


# --- the rollout policy -----------------------------------------------------


class ScriptedNet(torch.nn.Module):
    """Stands in for FlowPolicyNet: plan k is k*100 + step index, so the queue is checkable."""

    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        self.cameras = config.cameras
        self.calls = 0

    def sample(self, images, state, *, steps=None, generator=None, condition=None):
        plan = torch.zeros(1, self.config.chunk, self.config.action_dim)
        plan[0, :, 0] = self.calls * 100 + torch.arange(self.config.chunk, dtype=torch.float32)
        self.calls += 1
        return plan


def observation(config: FlowConfig = SMALL):
    return build_observation(
        joint_positions=np.zeros(config.arm_joints),
        gripper_openings=[0.5],
        images={c: np.zeros((*config.image_hw, 3), np.uint8) for c in config.cameras},
        embodiment=Embodiment(arm_joints=config.arm_joints, grippers=config.grippers),
    )


def policy_with(net, execute: int = 3) -> FlowPolicy:
    return FlowPolicy(net, device=torch.device("cpu"), metadata={}, execute=execute)


def test_runs_the_first_execute_actions_then_replans():
    net = ScriptedNet(SMALL)
    policy = policy_with(net, execute=3)
    firsts = [policy(observation())[0] for _ in range(7)]
    assert firsts == [0, 1, 2, 100, 101, 102, 200]
    assert net.calls == 3


def test_reset_drops_the_rest_of_the_plan():
    net = ScriptedNet(SMALL)
    policy = policy_with(net, execute=3)
    policy(observation())
    policy.reset()
    assert policy(observation())[0] == 100  # a fresh plan, not step 1 of the old one


def test_execute_must_fit_in_the_chunk():
    with pytest.raises(ValueError):
        policy_with(ScriptedNet(SMALL), execute=SMALL.chunk + 1)


def test_gripper_is_clipped_into_the_contract():
    net = ScriptedNet(SMALL)
    net.calls = 7
    policy = policy_with(net)
    action = policy(observation())
    assert action[0] == pytest.approx(700.0)  # arm joints are left alone
    assert 0.0 <= action[SMALL.gripper_slice][0] <= 1.0


def test_policy_seed_makes_rollouts_repeatable():
    policy = FlowPolicy(trained_ish(), device=torch.device("cpu"), metadata={}, seed=3)
    first = policy(observation())
    policy.reset()
    policy.seed(3)
    assert np.array_equal(policy(observation()), first)


def test_checkpoint_round_trips_to_the_same_plan(tmp_path):
    net = trained_ish()
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
    loaded = FlowPolicy.load(path, device="cpu")
    images, state, _ = batch(1)
    g = lambda: torch.Generator().manual_seed(0)  # noqa: E731
    assert torch.allclose(
        loaded._net.sample(images, state, generator=g()),
        net.sample(images, state, generator=g()),
        atol=1e-6,
    )


def test_flow_matching_is_registered():
    assert "flow_matching" in POLICIES
