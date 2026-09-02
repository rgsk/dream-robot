import json

import numpy as np
import pytest
from fakes import FAKE_EMBODIMENT, FakeEnv, FakePolicy

from dream_robot.core import record as record_mod
from dream_robot.core.dataset import (
    ACTION_FEATURE,
    STATE_FEATURE,
    episode_frames,
    open_dataset,
)
from dream_robot.core.env_api import FailureMode
from dream_robot.core.record import (
    Policy,
    check_tracking,
    record_episodes,
    rollout,
)

REPO_ID = "dream_robot/fake_task"
PROMPT = "put the cube in the bin"


def record(tmp_path, env=None, policy=None, **kwargs):
    kwargs.setdefault("episodes", 2)
    return record_episodes(
        env or FakeEnv(),
        policy or FakePolicy(),
        repo_id=REPO_ID,
        root=tmp_path / "ds",
        task_prompt=PROMPT,
        robot_type="fake",
        **kwargs,
    )


# --- the protocol -----------------------------------------------------------

def test_fake_policy_satisfies_the_policy_protocol():
    assert isinstance(FakePolicy(), Policy)


def test_a_policy_that_only_acts_is_not_enough():
    """reset() is in the protocol because a stateful policy must be rewound per episode."""
    class ActionOnly:
        def __call__(self, observation):
            return np.zeros(4, dtype=np.float32)

    assert not isinstance(ActionOnly(), Policy)


# --- rollout alignment ------------------------------------------------------

def test_rollout_arrays_are_index_aligned(fake_env, fake_policy):
    """states[t], actions[t] and images[t] are one (saw this, did that) pair."""
    episode = rollout(fake_env, fake_policy, seed=0)
    assert episode.steps == len(episode.states) == len(episode.images) == 5

    arm = FAKE_EMBODIMENT.arm_slice
    # FakePolicy commands state + delta, so this holds only if each action was
    # paired with the observation it was actually computed from.
    for t in range(episode.steps):
        assert np.allclose(
            episode.actions[t][arm], episode.states[t][arm] + fake_policy.delta, atol=1e-6
        )


def test_rollout_records_the_observation_before_the_action(fake_env, fake_policy):
    """FakeEnv paints the step index into every frame, so alignment is visible."""
    episode = rollout(fake_env, fake_policy, seed=0)
    for t in range(episode.steps):
        assert episode.images[t]["top"][0, 0, 0] == t


def test_rollout_excludes_the_terminal_observation(fake_env, fake_policy):
    """No action was ever taken in response to it, and inventing one would lie."""
    episode = rollout(fake_env, fake_policy, seed=0)
    # FakeEnv succeeds on step 5, so the successful frame is index 5 and absent.
    assert max(img["top"][0, 0, 0] for img in episode.images) == 4


def test_rollout_copies_frames_out_of_the_render_buffer(fake_env, fake_policy):
    """A backend reusing one buffer would otherwise give N identical frames."""
    episode = rollout(fake_env, fake_policy, seed=0)
    values = [int(img["wrist"][0, 0, 0]) for img in episode.images]
    assert len(set(values)) == len(values)


def test_rollout_seeds_the_environment(fake_env, fake_policy):
    rollout(fake_env, fake_policy, seed=17)
    assert fake_env.seeds == [17]
    assert fake_policy.resets == 1


def test_rollout_stops_on_truncation_not_on_a_step_budget(fake_policy):
    """Episode length is the environment's call. A fixed budget would put the
    clock back into the data that spec.md bans time features to keep out."""
    env = FakeEnv(succeed_after=None, horizon=6)
    episode = rollout(env, fake_policy, seed=0)
    assert episode.steps == 6
    assert not episode.success
    assert episode.failure_mode is FailureMode.NO_GRASP


def test_an_env_that_never_ends_is_an_error_not_a_long_episode(monkeypatch, fake_policy):
    monkeypatch.setattr(record_mod, "RUNAWAY_STEPS", 4)
    env = FakeEnv(succeed_after=None, horizon=10_000)
    with pytest.raises(RuntimeError, match="not enforcing a horizon"):
        rollout(env, fake_policy, seed=0)


def test_max_steps_caps_without_complaining(fake_policy):
    env = FakeEnv(succeed_after=None, horizon=10_000)
    assert rollout(env, fake_policy, seed=0, max_steps=3).steps == 3


# --- the integrity check ----------------------------------------------------

def test_tracking_check_passes_for_an_absolute_position_controller(fake_env, fake_policy):
    episode = rollout(fake_env, fake_policy, seed=0)
    mean, maximum = check_tracking(episode, FAKE_EMBODIMENT)
    # lag=0.1 of a 0.05 delta: the controller covers 90% of each command.
    assert maximum == pytest.approx(0.005, abs=1e-3)
    assert mean <= maximum


def test_tracking_check_catches_a_controller_ignoring_its_target():
    """What delta mode, or a misaligned column, looks like from outside."""
    env = FakeEnv(lag=1.0)                       # state never moves toward the action
    episode = rollout(env, FakePolicy(delta=0.5), seed=0)
    with pytest.raises(ValueError, match="exceeds"):
        check_tracking(episode, FAKE_EMBODIMENT)


# --- recording --------------------------------------------------------------

def test_only_successful_episodes_are_kept(tmp_path):
    """Behaviour cloning imitates what it is shown, so failures are poison."""
    # Succeeds on even seeds only.
    class Flaky(FakeEnv):
        def reset(self, *, seed=None):
            self._succeed_after = 5 if seed % 2 == 0 else None
            return super().reset(seed=seed)

    summary = record(tmp_path, env=Flaky(succeed_after=None, horizon=6), episodes=2)
    assert summary.kept == 2
    assert [e.seed for e in summary.episodes if e.kept] == [0, 2]
    assert [e.seed for e in summary.episodes] == [0, 1, 2]
    assert summary.success_rate == pytest.approx(2 / 3)
    assert summary.failure_histogram == {"no_grasp": 1}


def test_keep_failures_records_them_anyway(tmp_path):
    env = FakeEnv(succeed_after=None, horizon=4)
    summary = record(tmp_path, env=env, episodes=2, keep_failures=True)
    assert summary.kept == 2
    assert all(not e.success and e.kept for e in summary.episodes)


def test_recording_nothing_is_an_error_not_an_empty_dataset(tmp_path):
    env = FakeEnv(succeed_after=None, horizon=3)
    with pytest.raises(RuntimeError, match="no episode was kept"):
        record(tmp_path, env=env, episodes=1, max_attempts=2)


def test_a_short_run_says_so_and_still_writes(tmp_path, capsys):
    class Flaky(FakeEnv):
        def reset(self, *, seed=None):
            self._succeed_after = 5 if seed == 0 else None
            return super().reset(seed=seed)

    summary = record(
        tmp_path, env=Flaky(succeed_after=None, horizon=6), episodes=3, max_attempts=4
    )
    assert summary.kept == 1
    assert "kept 1 of 3" in capsys.readouterr().out


def test_a_prompt_is_required(tmp_path):
    """Language annotation from day one -- retrofitting means re-recording."""
    with pytest.raises(ValueError, match="non-empty instruction"):
        record_episodes(
            FakeEnv(), FakePolicy(), repo_id=REPO_ID, root=tmp_path / "ds",
            task_prompt="   ", episodes=1, robot_type="fake",
        )


# --- the round trip ---------------------------------------------------------

def test_dataset_round_trips_through_disk(tmp_path):
    summary = record(tmp_path, episodes=2)
    assert summary.total_frames == 10

    dataset = open_dataset(REPO_ID, tmp_path / "ds")
    assert dataset.num_episodes == 2
    assert dataset.num_frames == 10
    assert dataset.meta.fps == 30
    assert dataset.meta.robot_type == "fake"

    item = dataset[0]
    assert item[STATE_FEATURE].shape == (FAKE_EMBODIMENT.dim,)
    assert item[ACTION_FEATURE].shape == (FAKE_EMBODIMENT.dim,)
    assert item["task"] == PROMPT


def test_nothing_but_state_action_and_images_reaches_the_disk(tmp_path):
    """The structural form of spec.md's ban on ground truth in observations.

    The recorder builds frames out of an ``Observation``, which cannot be
    constructed from a backend's raw dict in the first place -- so the check
    that matters is that the written feature set is exactly the allowlist, with
    no room for a cube pose to have been smuggled alongside it.
    """
    record(tmp_path, episodes=1)
    features = set(open_dataset(REPO_ID, tmp_path / "ds").meta.features)
    bookkeeping = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    assert features - bookkeeping == {
        STATE_FEATURE, ACTION_FEATURE,
        "observation.images.top", "observation.images.wrist",
    }


def test_frames_survive_the_video_codec(tmp_path):
    """AV1 is lossy, so this checks the picture is recognisably the same one."""
    record(tmp_path, episodes=1)
    dataset = open_dataset(REPO_ID, tmp_path / "ds", episode=0)
    frames = episode_frames(dataset, "top")
    assert len(frames) == 5
    assert all(f.shape == (32, 32, 3) and f.dtype == np.uint8 for f in frames)
    # FakeEnv paints frame t with the value t, in order.
    recovered = [float(np.median(f)) for f in frames]
    assert recovered == sorted(recovered)
    assert np.allclose(recovered, [0, 1, 2, 3, 4], atol=3)


def test_summary_is_written_next_to_the_data(tmp_path):
    """A dataset whose seeds live only in a shell history is not reproducible."""
    summary = record(tmp_path, episodes=2)
    written = json.loads((tmp_path / "ds" / "recording_summary.json").read_text())
    assert written == summary.to_dict()
    assert written["task_prompt"] == PROMPT
    assert [e["seed"] for e in written["episodes"]] == [0, 1]


# --- against the real simulator ---------------------------------------------

@pytest.mark.slow
def test_recording_pick_place_cube_produces_a_readable_dataset(tmp_path):
    """The seam, end to end: robosuite in, a directory out, frames back.

    The only test in the file that needs a simulator. Everything above it runs
    against FakeEnv precisely because none of it should care.
    """
    from dream_robot.core.dataset import image_feature
    from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube, TaskConfig
    from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertActor, ExpertPolicy

    cfg = TaskConfig.load()
    env = PickPlaceCube(cfg)
    try:
        summary = record_episodes(
            env,
            ExpertActor(ExpertPolicy(env)),
            repo_id=REPO_ID,
            root=tmp_path / "ds",
            task_prompt=cfg.prompt,
            episodes=1,
            robot_type="panda",
        )
    finally:
        env.close()

    assert summary.kept == 1
    # Absolute joint targets: action[t] and state[t+1] are the same quantity, so
    # the gap is controller lag and stays bounded by the IK's per-step clamp.
    assert summary.episodes[0].tracking_error_max <= 0.07

    dataset = open_dataset(REPO_ID, tmp_path / "ds", episode=0)
    assert set(dataset.meta.camera_keys) == {image_feature("top"), image_feature("wrist")}
    assert dataset.meta.fps == cfg.control_hz
    assert dataset[0]["task"] == cfg.prompt

    height, width = cfg.camera_hw
    frames = episode_frames(dataset, "top")
    assert len(frames) == summary.episodes[0].steps
    assert all(f.shape == (height, width, 3) for f in frames)
    # Not a uniform buffer: the scene actually got rendered into these.
    assert frames[0].std() > 5
