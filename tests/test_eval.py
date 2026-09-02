import json

import numpy as np
import pytest
from fakes import FakeEnv, FakePolicy

from dream_robot.core.env_api import FailureMode
from dream_robot.core.eval import EVAL_SEED_START, evaluate, write_results


def run(env=None, policy=None, **kwargs):
    kwargs.setdefault("episodes", 4)
    kwargs.setdefault("video_per_outcome", 0)
    return evaluate(
        env or FakeEnv(),
        policy or FakePolicy(),
        task="fake/task",
        policy_name="fake",
        **kwargs,
    )


class Flaky(FakeEnv):
    """Succeeds on even seeds, times out on odd ones."""

    def reset(self, *, seed=None):
        self._succeed_after = 5 if seed % 2 == 0 else None
        return super().reset(seed=seed)


# --- what gets counted ------------------------------------------------------

def test_success_rate_counts_every_episode():
    """Unlike the recorder, evaluation may not discard its failures."""
    result = run(env=Flaky(succeed_after=None, horizon=8), episodes=4, seed_start=0)
    assert len(result.episodes) == 4
    assert result.success_rate == 0.5
    assert [e.seed for e in result.episodes] == [0, 1, 2, 3]


def test_failure_histogram_uses_the_closed_vocabulary():
    result = run(
        env=Flaky(succeed_after=None, horizon=8, failure_mode=FailureMode.DROPPED),
        episodes=4, seed_start=0,
    )
    assert result.failure_histogram == {"dropped": 2}
    # Successes carry no failure mode.
    assert all(
        e.failure_mode is FailureMode.NONE for e in result.episodes if e.success
    )


def test_cycle_time_is_measured_over_successes_only():
    """Otherwise a policy that fails fast looks quick, and the two numbers couple."""
    result = run(env=Flaky(succeed_after=None, horizon=40), episodes=4, seed_start=0)
    # Successes take 5 steps at 30 Hz; failures take 40 and must not count.
    assert result.cycle_time == pytest.approx(5 / 30)


def test_cycle_time_is_none_when_nothing_succeeded():
    result = run(env=FakeEnv(succeed_after=None, horizon=6), episodes=2)
    assert result.cycle_time is None
    assert result.success_rate == 0.0


def test_evaluation_seeds_default_away_from_the_recorder_block():
    """Evaluating on the recorded seeds would measure memorisation of 25 scenes."""
    env = FakeEnv()
    run(env=env, episodes=3)
    assert env.seeds == [EVAL_SEED_START, EVAL_SEED_START + 1, EVAL_SEED_START + 2]


def test_the_policy_is_reset_once_per_episode(fake_policy):
    run(policy=fake_policy, episodes=3)
    assert fake_policy.resets == 3


# --- reporting --------------------------------------------------------------

def test_summary_reports_rate_and_failures():
    result = run(env=Flaky(succeed_after=None, horizon=8), episodes=4, seed_start=0)
    text = result.summary()
    assert "50%" in text and "2/4" in text and "no_grasp" in text


def test_results_json_carries_provenance(tmp_path):
    """A number with no way to reproduce it is the same as no number."""
    result = run(episodes=2)
    path = write_results(
        result, tmp_path / "results.json",
        extra={"checkpoint": "x.pt", "demonstrations": 20},
    )
    payload = json.loads(path.read_text())
    assert payload["task"] == "fake/task"
    assert payload["success_rate"] == 1.0
    assert payload["provenance"]["demonstrations"] == 20
    assert len(payload["rollouts"]) == 2


# --- video ------------------------------------------------------------------

def test_successes_and_failures_are_filmed_separately(tmp_path):
    """The bug this fixes: the first BC run filmed three episodes, all failures,
    so there was no footage of the policy ever succeeding."""
    result = run(
        env=Flaky(succeed_after=None, horizon=8), episodes=4, seed_start=0,
        video_dir=tmp_path, video_per_outcome=2, hold_frames=0,
    )
    assert set(result.videos) == {"successes", "failures"}

    import imageio.v3 as iio
    # Two successes of 5 steps; two failures of 8.
    assert len(iio.imread(result.videos["successes"])) == 2 * 5
    assert len(iio.imread(result.videos["failures"])) == 2 * 8


def test_the_per_outcome_quota_is_respected(tmp_path):
    result = run(
        env=Flaky(succeed_after=None, horizon=8), episodes=8, seed_start=0,
        video_dir=tmp_path, video_per_outcome=1, hold_frames=0,
    )
    import imageio.v3 as iio
    assert len(iio.imread(result.videos["successes"])) == 5
    assert len(iio.imread(result.videos["failures"])) == 8


def test_an_outcome_that_never_happened_leaves_no_file(tmp_path):
    """A policy that never fails should not produce an empty failures.mp4."""
    result = run(episodes=2, video_dir=tmp_path, video_per_outcome=2, hold_frames=0)
    assert set(result.videos) == {"successes"}
    assert not (tmp_path / "failures.mp4").exists()


def test_explicit_seeds_film_exactly_those_episodes(tmp_path):
    result = run(
        env=Flaky(succeed_after=None, horizon=8), episodes=6, seed_start=0,
        video_dir=tmp_path, video_seeds=[3, 4], hold_frames=0,
    )
    assert set(result.videos) == {"selected"}
    import imageio.v3 as iio
    # Seed 3 fails (8 steps), seed 4 succeeds (5) -- filmed in seed order.
    assert len(iio.imread(result.videos["selected"])) == 8 + 5


def test_no_video_requested_means_no_file(tmp_path):
    result = run(episodes=2, video_dir=None)
    assert result.videos == {}
    assert not list(tmp_path.iterdir())


def test_rendering_stops_once_both_quotas_are_full(tmp_path):
    """render() is the expensive call; filming must not cost the whole run."""
    class Counting(Flaky):
        renders = 0

        def render(self):
            type(self).renders += 1
            return super().render()

    env = Counting(succeed_after=None, horizon=8)
    run(env=env, episodes=8, seed_start=0, video_dir=tmp_path,
        video_per_outcome=1, hold_frames=0)
    # Quotas fill on seeds 0 and 1; the remaining six episodes render nothing.
    assert Counting.renders == 5 + 8


def test_panel_shows_the_wide_view_beside_the_policy_input(tmp_path):
    """The left half is for a human; the right half is the complete policy input."""
    result = run(episodes=1, video_dir=tmp_path, video_per_outcome=1, hold_frames=0)
    import imageio.v3 as iio
    frame = np.asarray(iio.imread(result.videos["successes"]))[0]
    # FakeEnv renders 64x64; two 32x32 cameras stack into a 32-wide column.
    assert frame.shape == (64, 64 + 32, 3)
    # The wide view is a constant 7; h.264 is lossy, so compare with tolerance.
    assert abs(int(frame[0, 0, 0]) - 7) <= 3
