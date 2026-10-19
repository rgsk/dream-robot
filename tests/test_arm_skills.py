"""pick/place_at on the real T5 and T7 sims, aimed by perception, through the MCP server."""

import asyncio
import json

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("robosuite")

from mcp import Client  # noqa: E402

from dream_robot.agent.server import build_scored_server  # noqa: E402
from dream_robot.core.registry import make_env  # noqa: E402


def _drive(calls, task="robosuite/instructed_pick_place"):
    """Run (tool, args) calls in order on one fresh server; return each result, then the
    referee's verdict."""
    server, score = build_scored_server(lambda: make_env(task))

    async def go():
        async with Client(server) as c:
            out = []
            for name, args in calls:
                r = await c.call_tool(name, args(out) if callable(args) else args)
                out.append(r.content[0].text if r.is_error else json.loads(r.content[0].text))
            return out

    return [*asyncio.run(go()), score()]


def _container(things, side):
    """The container on ``side``, as the top camera sees it: entries come left to right."""
    bins = [t for t in things["things"] if t["kind"] == "container"]
    return bins[0] if side == "left" else bins[-1]


def test_t5_find_by_colour_pick_place_at_completes_seed_1000():
    # Seed 1000: "put the yellow cube in the right box".
    reset, found, things, picked, placed, status = _drive([
        ("reset", {"seed": 1000}),
        ("find_by_colour", {"colour": "yellow"}),
        ("find_objects", {}),
        ("pick", lambda out: {"grasp": out[1]["grasp"]}),
        ("place_at", lambda out: {"point": _container(out[2], "right")["drop_point"]}),
    ])
    assert reset["instruction"] == "put the yellow cube in the right box"
    assert picked["ok"] and picked["holding"]
    assert placed["ok"]
    assert status["success"]


def test_t7_find_objects_pick_place_at_completes_seed_1000():
    # Seed 1000: "put the red box in the left bin"; the red box is the only red
    # rectangular-footprint object.
    def red_box(out):
        (box,) = [t for t in out[1]["things"]
                  if t["kind"] == "object" and t["colour"] == "red" and t["footprint"] == "rectangular"
                  and t["size"]["height"] < 0.06]
        return {"grasp": box["grasp"]}

    reset, things, picked, placed, status = _drive([
        ("reset", {"seed": 1000}),
        ("find_objects", {}),
        ("pick", red_box),
        ("place_at", lambda out: {"point": _container(out[1], "left")["drop_point"]}),
    ], task="robosuite/assorted_pick_place")
    assert reset["instruction"] == "put the red box in the left bin"
    assert sum(t["kind"] == "container" for t in things["things"]) == 2
    assert picked["ok"] and picked["holding"]
    assert placed["ok"]
    assert status["success"]


def test_failures_are_reported_not_raised():
    placed, picked, _ = _drive([
        ("place_at", {"point": [0.0, -0.2, 0.85]}),                 # nothing held
        ("pick", {"grasp": {"point": [0.0, 0.1, 0.83]}}),           # empty table
    ])
    assert placed == {**placed, "ok": False, "failure": "no_grasp", "steps": 0}
    assert picked == {**picked, "ok": False, "failure": "no_grasp", "holding": False}
