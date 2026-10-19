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
    assert "success" not in status                    # no tool reports success


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


def test_arm_tools_and_find_objects_are_offered_only_for_a_robosuite_arm():
    async def tools():
        async with Client(build_server(FakeEnv)) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert not {"move_to", "move_by", "gripper", "find_objects"} & asyncio.run(tools())
