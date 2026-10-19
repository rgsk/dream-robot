"""The robot MCP server, driven through a real MCP client in-process.

Uses FakeEnv, so no simulator: what is under test is the tool surface -- names,
arguments, return shapes -- which is what every planner depends on.
"""

import asyncio

import imageio.v3 as iio
import numpy as np
from fakes import FAKE_HW, FAKE_RENDER_HW, FakeEnv, overhead_rgbd
from mcp import Client

from dream_robot.agent.server import build_server


def _run(coro_fn):
    async def go():
        async with Client(build_server(FakeEnv)) as client:
            return await coro_fn(client)

    return asyncio.run(go())


def test_lists_the_three_tools():
    async def go(client):
        return {t.name for t in (await client.list_tools()).tools}

    assert _run(go) == {"reset", "look", "status"}


def test_reset_then_status_round_trips():
    async def go(client):
        reset = await client.call_tool("reset", {"seed": 3})
        status = await client.call_tool("status", {})
        return reset.structured_content, status.structured_content

    reset, status = _run(go)
    assert reset == status
    assert status["seed"] == 3
    assert status["steps"] == 0
    assert status["success"] is False
    assert status["failure_mode"] == "none"


def test_look_returns_a_png_per_camera():
    async def go(client):
        await client.call_tool("reset", {"seed": 0})
        out = {}
        for cam in ("scene", "top", "wrist"):
            (content,) = (await client.call_tool("look", {"camera": cam})).content
            out[cam] = content
        return out

    import base64

    for cam, content in _run(go).items():
        assert content.type == "image" and content.mime_type == "image/png"
        frame = iio.imread(base64.b64decode(content.data))
        assert frame.shape[:2] == (FAKE_RENDER_HW if cam == "scene" else FAKE_HW)


def test_camera_is_an_enum_of_the_envs_cameras():
    """An enum is a dropdown in the Inspector and a closed choice for an LLM."""

    async def go(client):
        (look,) = [t for t in (await client.list_tools()).tools if t.name == "look"]
        return look.input_schema["properties"]["camera"]

    camera = _run(go)
    assert camera["enum"] == ["scene", "top", "wrist"]
    assert camera["default"] == "scene"


def test_unknown_camera_is_rejected_before_the_sim_sees_it():
    async def go(client):
        return await client.call_tool("look", {"camera": "side"})

    result = _run(go)
    assert result.is_error
    assert "scene" in result.content[0].text


def test_a_crash_reaches_the_caller_with_its_text():
    class BrokenRender(FakeEnv):
        def render(self):
            raise RuntimeError("GL context lost")

    async def go():
        async with Client(build_server(BrokenRender)) as client:
            return await client.call_tool("look", {"camera": "scene"})

    result = asyncio.run(go())
    assert result.is_error
    assert "RuntimeError: GL context lost" in result.content[0].text


def test_every_env_call_runs_on_one_thread():
    """MuJoCo's GL context is bound to the thread that created it; a render on
    another thread returns noise. MCP's own worker pool would scatter calls."""
    import threading

    threads = set()

    class ThreadRecordingEnv(FakeEnv):
        def __init__(self):
            threads.add(threading.get_ident())
            super().__init__()

        def reset(self, **kw):
            threads.add(threading.get_ident())
            return super().reset(**kw)

        def render(self):
            threads.add(threading.get_ident())
            return super().render()

    async def go():
        async with Client(build_server(ThreadRecordingEnv)) as client:
            for seed in range(5):
                await client.call_tool("reset", {"seed": seed})
                await client.call_tool("look", {"camera": "scene"})

    asyncio.run(go())
    assert len(threads) == 1


class DepthFakeEnv(FakeEnv):
    """FakeEnv plus a depth camera that sees one blue cube top at (0.1, 0.2, 0.05)."""

    def rgbd(self, camera="top"):
        return overhead_rgbd(hw=128, squares=[((25, 60, 230), (0.1, 0.2), 0.05, 0.05)])


def _call_find(colour, env=DepthFakeEnv):
    async def go():
        async with Client(build_server(env)) as client:
            return await client.call_tool("find_by_colour", {"colour": colour})

    return asyncio.run(go())


def test_find_by_colour_is_offered_only_when_the_env_has_a_depth_camera():
    async def tools(env):
        async with Client(build_server(env)) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert "find_by_colour" not in asyncio.run(tools(FakeEnv))
    assert "find_by_colour" in asyncio.run(tools(DepthFakeEnv))


def test_find_returns_the_point_and_a_marked_image():
    import json

    text, image = _call_find("blue").content
    result = json.loads(text.text)
    assert result["found"] is True and result["ambiguous"] is False
    np.testing.assert_allclose(result["point"], [0.1, 0.2, 0.05], atol=0.01)
    assert [c["id"] for c in result["candidates"]] == [1]
    assert image.type == "image"


def test_find_of_an_absent_object_says_so():
    import json

    text, _ = _call_find("red").content
    assert json.loads(text.text) == {
        "colour": "red", "found": False, "ambiguous": False,
        "point": None, "candidates": [],
    }


class TwoBlueFakeEnv(FakeEnv):
    def rgbd(self, camera="top"):
        blue = (25, 60, 230)
        return overhead_rgbd(hw=128, squares=[
            (blue, (0.2, 0.0), 0.05, 0.05), (blue, (-0.2, 0.0), 0.05, 0.05),
        ])


def test_two_matches_are_ambiguous_with_no_point_and_both_candidates():
    import json

    text, _ = _call_find("blue", env=TwoBlueFakeEnv).content
    result = json.loads(text.text)
    assert result["found"] is False and result["ambiguous"] is True
    assert result["point"] is None
    xs = sorted(c["point"][0] for c in result["candidates"])
    np.testing.assert_allclose(xs, [-0.2, 0.2], atol=0.01)


def test_colour_is_an_enum_and_anything_else_is_rejected():
    async def schema():
        async with Client(build_server(DepthFakeEnv)) as client:
            (t,) = [t for t in (await client.list_tools()).tools if t.name == "find_by_colour"]
            return t.input_schema["properties"]["colour"]

    assert asyncio.run(schema())["enum"] == ["red", "yellow", "green", "blue"]
    result = _call_find("the mug")
    assert result.is_error


class PaintableFakeEnv(FakeEnv):
    palette = ("red", "blue")

    def paint(self, colors):
        self.painted = colors
        return self._observe()


def test_debug_tools_exist_only_on_a_debug_server():
    """A planner that can repaint the scene can succeed by changing the task."""

    async def tools(debug):
        async with Client(build_server(PaintableFakeEnv, debug=debug)) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert "debug_paint" not in asyncio.run(tools(False))
    assert "debug_paint" in asyncio.run(tools(True))
