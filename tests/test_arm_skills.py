"""pick/place on the real T5 sim, aimed by perception, through the MCP server."""

import asyncio
import json

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("robosuite")

from mcp import Client  # noqa: E402

from dream_robot.agent.server import build_server  # noqa: E402
from dream_robot.core.registry import make_env  # noqa: E402


def _drive(calls):
    """Run (tool, args) calls in order on one fresh server; return each result."""

    async def go():
        async with Client(build_server(lambda: make_env("robosuite/instructed_pick_place"))) as c:
            out = []
            for name, args in calls:
                r = await c.call_tool(name, args(out) if callable(args) else args)
                out.append(r.content[0].text if r.is_error else json.loads(r.content[0].text))
            return out

    return asyncio.run(go())


def test_find_pick_place_completes_seed_1000():
    # Seed 1000: "put the yellow cube in the right box".
    reset, found, picked, placed, status = _drive([
        ("reset", {"seed": 1000}),
        ("find_by_colour", {"colour": "yellow"}),
        ("pick", lambda out: {"point": out[1]["point"]}),
        ("place", {"bin": "right"}),
        ("status", {}),
    ])
    assert reset["instruction"] == "put the yellow cube in the right box"
    assert picked["ok"] and picked["holding"]
    assert placed["ok"]
    assert status["success"]


def test_failures_are_reported_not_raised():
    placed, picked = _drive([
        ("place", {"bin": "left"}),                                 # nothing held
        ("pick", {"point": [0.0, 0.1, 0.83]}),                  # empty table
    ])
    assert placed == {**placed, "ok": False, "failure": "no_grasp", "steps": 0}
    assert picked == {**picked, "ok": False, "failure": "no_grasp", "holding": False}
