"""The robot as an MCP server. ROADMAP rung 9's executor side, reached only through tools.

Everything that drives the robot -- Claude Code from the terminal, our own LLM
planner, a scripted test -- goes through these tools and nothing else. That is
the planner/executor boundary from ROADMAP made literal: if a planner can do it,
there is a tool for it, and a tool's return value is the whole return channel.

**One env per server process, built on first reset.** A sim takes seconds to
build, and an MCP client gives up on a server that is slow to answer the
handshake. So startup is instant and the first ``reset`` pays for the build.

**Tools return plain dicts and images.** A dict is what both MCP clients and
OpenAI function calling hand straight to the model, so the tool layer needs no
second adapter per planner.

Run it the way a client launches it::

    MUJOCO_GL=egl uv run python -m dream_robot.agent.server --env robosuite/instructed_pick_place

``.mcp.json`` at the repo root registers exactly this command for Claude Code.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import functools
import io
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import imageio.v3 as iio
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from dream_robot.core.env_api import Env, FailureMode, StepResult

INSTRUCTIONS = """\
A simulated robot arm (Franka Panda) at a table. Call `reset` to start an episode;
it returns the task instruction. `look` shows the scene from a camera. `status`
reports progress and, once the episode ends, how it failed.
"""


class RobotSession:
    """The state behind the tools: one env, one episode at a time."""

    def __init__(self, make_env: Callable[[], Env]):
        self._make_env = make_env
        self._env: Env | None = None
        self._obs = None
        self._last: StepResult | None = None
        self._steps = 0
        self._seed: int | None = None

    @property
    def env(self) -> Env:
        if self._env is None:
            raise ToolError("no episode yet: call reset first")
        return self._env

    def reset(self, seed: int | None) -> dict[str, Any]:
        with _quiet():
            if self._env is None:
                self._env = self._make_env()
            self._obs = self._env.reset(seed=seed)
        self._last, self._steps, self._seed = None, 0, seed
        return self.status()

    def frame(self, camera: str) -> np.ndarray:
        if camera == "front":
            return np.asarray(self.env.render())
        images = self._obs.images
        if camera not in images:
            raise ToolError(f"no camera {camera!r}; have front, {', '.join(images)}")
        return np.asarray(images[camera])

    def status(self) -> dict[str, Any]:
        last = self._last
        return {
            "instruction": getattr(self.env, "prompt", None),
            "seed": self._seed,
            "steps": self._steps,
            "success": bool(last and last.success),
            "done": bool(last and (last.terminated or last.truncated)),
            "failure_mode": str(last.failure_mode if last else FailureMode.NONE),
        }


def build_server(make_env: Callable[[], Env]) -> MCPServer:
    session = RobotSession(make_env)
    server = MCPServer("dream-robot", instructions=INSTRUCTIONS)
    on_sim = _on_sim_thread(ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim"))

    @server.tool()
    @on_sim
    def reset(seed: int | None = None) -> dict[str, Any]:
        """Start a new episode. Same seed, same scene. Returns the instruction and status."""
        return session.reset(seed)

    @server.tool()
    @on_sim
    def look(camera: str = "front") -> Image:
        """An image of the scene. Cameras: "front" (wide view), "top", "wrist" (the robot's own)."""
        return Image(data=iio.imwrite("<bytes>", session.frame(camera), extension=".png"))

    @server.tool()
    @on_sim
    def status() -> dict[str, Any]:
        """The instruction, steps taken, and whether the episode succeeded or how it failed."""
        return session.status()

    return server


def _on_sim_thread(sim: ThreadPoolExecutor):
    """Run a tool on the one sim thread, and pass its crash text to the caller.

    **One thread, always the same one.** MCP runs a plain function tool on any
    free worker thread, but MuJoCo's GL context belongs to the thread that made
    it. A ``look`` that lands on a different thread from the ``reset`` that built
    the sim reads a framebuffer with no context and returns noise -- which is
    what the Inspector showed, while Claude Code happened to hit one thread.

    **Crash text, not "Error executing tool".** MCP withholds an unexpected
    exception's message from the client, a sensible default for a public server.
    This one is local, and its caller is the one debugging the run.
    """

    def wrap(tool):
        def run(*args, **kwargs):
            try:
                return tool(*args, **kwargs)
            except ToolError:
                raise
            except Exception as exc:
                raise ToolError(f"{type(exc).__name__}: {exc}") from exc

        @functools.wraps(tool)
        async def wrapped(*args, **kwargs):
            return await asyncio.wrap_future(sim.submit(run, *args, **kwargs))

        return wrapped

    return wrap


@contextlib.contextmanager
def _quiet():
    """Send stray prints to stderr. On a stdio server, stdout *is* the protocol;
    robosuite's import-time banner there would corrupt the first message."""
    with contextlib.redirect_stdout(sys.stderr if sys.stderr else io.StringIO()):
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="robosuite/instructed_pick_place")
    args = parser.parse_args()

    from dream_robot.core.registry import TASKS

    if args.env not in TASKS:
        parser.error(f"unknown env {args.env!r}; have {', '.join(TASKS)}")
    build_server(TASKS[args.env]).run("stdio")


if __name__ == "__main__":
    main()
