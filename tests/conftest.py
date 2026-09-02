"""Shared fixtures. The fakes themselves live in ``fakes.py`` -- see the note there."""

import pytest
from fakes import FakeEnv, FakePolicy


@pytest.fixture
def fake_env():
    return FakeEnv()


@pytest.fixture
def fake_policy():
    return FakePolicy()
