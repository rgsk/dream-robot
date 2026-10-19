"""The robot as an MCP server. ROADMAP rung 9's executor side, reached only through tools.

Everything that drives the robot -- Claude Code from the terminal, our own LLM
planner, a scripted test -- goes through these tools and nothing else. That is
the planner/executor boundary from ROADMAP made literal: if a planner can do it,
there is a tool for it, and a tool's return value is the whole return channel.

**The sim is built at startup, with one warm-up reset.** That is how the server
learns which cameras this task has, and the camera list is part of ``look``'s
schema: an enum, so the Inspector shows a dropdown and an LLM cannot ask for a
camera that is not there. A ``reset`` call then starts a real episode.

**Tools return plain dicts and images.** A dict is what both MCP clients and
OpenAI function calling hand straight to the model, so the tool layer needs no
second adapter per planner.

Run it the way a client launches it::

    MUJOCO_GL=egl uv run python -m dream_robot.agent.server --env robosuite/instructed_pick_place

``.mcp.json`` at the repo root registers exactly this command for Claude Code.
"""

# No `from __future__ import annotations`: look's camera type is a Literal built
# at runtime from the env, and a stringified annotation could not see it.

import argparse
import asyncio
import contextlib
import functools
import io
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import imageio.v3 as iio
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from dream_robot.agent.perception import COLOURS, colour_mask, find, overlay
from dream_robot.core.env_api import Env, FailureMode, StepResult

INSTRUCTIONS = """\
A simulated robot arm (Franka Panda) at a table. Call `reset` to start an episode;
it returns the task instruction. `look` shows the scene from a camera. `find_by_colour`
locates objects of a colour and returns their 3D points. `status` reports progress and,
once the episode ends, how it failed.

Points are in metres, in the world frame: origin at the table centre, z up.
"""

# The wide render view, env.render(). Not "front": that is a canonical
# observation camera (schema.CANONICAL_CAMERAS), and a task may record one.
SCENE = "scene"


class RobotSession:
    """The state behind the tools: one env, one episode at a time.

    Construct it on the sim thread -- the constructor builds the sim.
    """

    def __init__(self, make_env: Callable[[], Env]):
        with _quiet():
            self.env = make_env()
            self._obs = self.env.reset()           # warm-up: learn the cameras
        self.cameras: tuple[str, ...] = (SCENE, *self._obs.images)
        self._last: StepResult | None = None
        self._steps = 0
        self._seed: int | None = None

    def reset(self, seed: int | None) -> dict[str, Any]:
        with _quiet():
            self._obs = self.env.reset(seed=seed)
        self._last, self._steps, self._seed = None, 0, seed
        return self.status()

    def frame(self, camera: str) -> np.ndarray:
        if camera == SCENE:
            return np.asarray(self.env.render())
        return np.asarray(self._obs.images[camera])

    def find_by_colour(self, colour: str) -> tuple[dict[str, Any], np.ndarray]:
        frame = self.env.rgbd("top")
        found = find(frame, colour_mask(frame.rgb, colour))
        # One shape for every outcome, so a planner reads the same fields each time.
        candidates = [
            {"id": i, "point": [round(float(c), 4) for c in f.point], "pixels": f.pixels}
            for i, f in enumerate(found, start=1)
        ]
        result = {
            "colour": colour,
            "found": len(found) == 1,
            "ambiguous": len(found) > 1,
            "point": candidates[0]["point"] if len(found) == 1 else None,
            "candidates": candidates,
        }
        return result, overlay(frame, found)

    def paint(self, colors: tuple[str, str]) -> None:
        self._obs = self.env.paint(colors)

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


def build_server(make_env: Callable[[], Env], *, debug: bool = False) -> MCPServer:
    """``debug`` adds tools that edit the world, for building test scenes by hand.
    Never give them to a planner: a planner that can repaint the scene can
    "succeed" by changing the task instead of doing it."""
    sim = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
    session = sim.submit(RobotSession, make_env).result()
    server = MCPServer("dream-robot", instructions=INSTRUCTIONS)
    on_sim = _on_sim_thread(sim)
    Camera = Literal[session.cameras]

    @server.tool()
    @on_sim
    def reset(seed: int | None = None) -> dict[str, Any]:
        """Start a new episode. Same seed, same scene. Returns the instruction and status."""
        return session.reset(seed)

    @server.tool()
    @on_sim
    def look(camera: Camera = SCENE) -> Image:
        """An image of the scene. "scene" is a wide view; the rest are the robot's own cameras."""
        return _png(session.frame(camera))

    if hasattr(session.env, "rgbd"):                 # needs a depth camera

        Colour = Literal[COLOURS]

        @server.tool()
        @on_sim
        def find_by_colour(colour: Colour) -> list:
            """Locate objects of one colour in the top camera.

            Knows colour only -- not shape, size or position. ``found`` and ``point`` are set
            only when exactly one object matches. When several do, ``ambiguous`` is true
            and ``point`` is null: choose from ``candidates`` (each with a 3D point in
            metres, top-face centre), whose ids are drawn on the returned image.
            """
            result, img = session.find_by_colour(colour)
            return [result, _png(img)]

    if debug and hasattr(session.env, "paint"):
        Colour = Literal[session.env.palette]

        @server.tool()
        @on_sim
        def debug_paint(upper_cube: Colour, lower_cube: Colour) -> Image:
            """DEBUG: recolour the two cubes, "upper" and "lower" as seen in the scene image.

            Lasts until the next reset. The instruction is not changed, so it may no
            longer match. Returns the scene after painting.
            """
            # Slot 0 sits nearer the robot, which is the upper cube in every view.
            session.paint((upper_cube, lower_cube))
            return _png(session.frame(SCENE))

    @server.tool()
    @on_sim
    def status() -> dict[str, Any]:
        """The instruction, steps taken, and whether the episode succeeded or how it failed."""
        return session.status()

    return server


def _png(img: np.ndarray) -> Image:
    return Image(data=iio.imwrite("<bytes>", img, extension=".png"))


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
    parser.add_argument("--debug", action="store_true", help="add scene-editing debug tools")
    args = parser.parse_args()

    from dream_robot.core.registry import TASKS

    if args.env not in TASKS:
        parser.error(f"unknown env {args.env!r}; have {', '.join(TASKS)}")
    build_server(TASKS[args.env], debug=args.debug).run("stdio")


if __name__ == "__main__":
    main()
