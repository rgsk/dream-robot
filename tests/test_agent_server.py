"""The robot MCP server, driven through a real MCP client in-process.

Uses FakeEnv, so no simulator: what is under test is the tool surface -- names,
arguments, return shapes -- which is what every planner depends on.
"""

import asyncio

import imageio.v3 as iio
from fakes import FAKE_HW, FAKE_RENDER_HW, FakeEnv
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
        for cam in ("front", "top", "wrist"):
            (content,) = (await client.call_tool("look", {"camera": cam})).content
            out[cam] = content
        return out

    import base64

    for cam, content in _run(go).items():
        assert content.type == "image" and content.mime_type == "image/png"
        frame = iio.imread(base64.b64decode(content.data))
        assert frame.shape[:2] == (FAKE_RENDER_HW if cam == "front" else FAKE_HW)


def test_tools_before_reset_report_an_error_not_a_crash():
    async def go(client):
        return await client.call_tool("status", {})

    result = _run(go)
    assert result.is_error
    assert "reset" in result.content[0].text


def test_unknown_camera_is_an_error_naming_the_real_ones():
    async def go(client):
        await client.call_tool("reset", {})
        return await client.call_tool("look", {"camera": "side"})

    result = _run(go)
    assert result.is_error
    assert "front" in result.content[0].text


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
                await client.call_tool("look", {"camera": "front"})

    asyncio.run(go())
    assert len(threads) == 1
