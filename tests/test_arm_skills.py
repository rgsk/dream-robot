"""The arm primitives on the real T7 sim, aimed by perception, through the MCP server."""

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


def _pick_place(grasp, drop, hang):
    """The driver prompt's pick and place recipes, as (tool, args) calls."""
    x, y, z = grasp["point"]
    dx, dy, dz = drop
    return [
        ("move_to", {"point": [x, y, z + 0.10], "yaw_deg": grasp["yaw_deg"]}),
        ("move_to", {"point": [x, y, z]}),
        ("gripper", {"action": "close"}),
        ("move_by", {"dz": 0.2}),
        ("move_to", {"point": [dx, dy, dz + hang + 0.15]}),
        ("move_to", {"point": [dx, dy, dz + hang + 0.02]}),
        ("gripper", {"action": "open"}),
        ("move_by", {"dz": 0.12}),
    ]


def test_t7_primitives_pick_place_completes_seed_2009():
    # Seed 2009: "put the yellow box in the right bin"; the only yellow object.
    server, score = build_scored_server(lambda: make_env("robosuite/assorted_pick_place"))

    async def go():
        async with Client(server) as c:
            reset = json.loads((await c.call_tool("reset", {"seed": 2009})).content[0].text)
            things = json.loads((await c.call_tool("find_objects", {})).content[0].text)
            (box,) = [t for t in things["things"] if t["kind"] == "object" and t["colour"] == "yellow"]
            calls = _pick_place(box["grasp"], _container(things, "right")["drop_point"],
                                box["grasp"]["above_table"])
            out = []
            for name, args in calls:
                out.append(json.loads((await c.call_tool(name, args)).content[0].text))
            return reset, out

    reset, out = asyncio.run(go())
    assert reset["instruction"] == "put the yellow box in the right bin"
    assert out[2]["hand"]["holding"]                     # closed on the box
    assert not out[6]["hand"]["holding"]                 # let go
    assert score()["success"]


def test_failures_are_reported_not_raised():
    closed, blocked, _ = _drive([
        ("gripper", {"action": "close"}),                         # on nothing
        ("move_to", {"point": [0.05, 0.0, 0.79]}),                # into the table
    ])
    assert closed["ok"] and not closed["hand"]["holding"]
    assert blocked == {**blocked, "ok": False, "failure": "blocked"}
