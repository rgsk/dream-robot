"""The robot as an MCP server. ROADMAP rung 9's executor side, reached only through tools.

Everything that drives the robot -- Claude Code from the terminal, our own LLM
planner, a scripted test -- goes through these tools and nothing else. That is
the planner/executor boundary from ROADMAP made literal: if a planner can do it,
there is a tool for it, and a tool's return value is the whole return channel.

**The sim is built at startup, with one warm-up reset.** That is how the server
learns which cameras this task has, and the camera list is part of ``look``'s
schema: an enum, so the Inspector shows a dropdown and an LLM cannot ask for a
camera that is not there. A ``reset`` call then starts a real episode.

**No tool reports success.** The sim knows whether the named object is in the
named bin; a real robot does not. So the episode does not end when the task is
done, and nothing the driver sees says it is: the driver judges from the images.
Scoring is a referee's job, outside the tools (``build_scored_server``).

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
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal

import imageio.v3 as iio
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from dream_robot.agent import recorder, tabletop
from dream_robot.agent.perception import COLOURS, colour_mask, find, overlay
from dream_robot.core.env_api import Env, StepResult

# The base prompt for whoever drives the robot -- an LLM, Claude Code, a person. Sent as
# the MCP server's instructions, so every client gets the same one. Edit the .md file.
INSTRUCTIONS = (Path(__file__).with_name("driver_prompt.md")).read_text()

# [x, y, z] in metres, world frame.
Point = Annotated[list[float], Field(min_length=3, max_length=3)]

# find_by_colour's top face is a cube's centre this much higher (Lift's cubes:
# 2.0-2.2 cm half-size). It knows colour, not shape, so its grasps assume a cube.
CUBE_TOP_TO_CENTRE = 0.021


class GraspIn(BaseModel):
    """A grasp as find_objects returns it; pass it to pick unchanged. Extra fields are ignored."""

    point: Point = Field(description="where the finger pads close, metres, world frame")
    yaw_deg: float | None = Field(
        None, description="world direction the fingers close along; null keeps the wrist as is"
    )
    above_table: float | None = Field(
        None, description="point's height over the table: how far the object hangs when lifted"
    )

# The wide render view, env.render(). Not "front": that is a canonical
# observation camera (schema.CANONICAL_CAMERAS), and a task may record one.
SCENE = "scene"


class RobotSession:
    """The state behind the tools: one env, one episode at a time.

    Construct it on the sim thread -- the constructor builds the sim.
    """

    def __init__(
        self, make_env: Callable[[], Env], *, viewer: str | None = None,
        score_log: str | None = None, clips_dir: str | None = None,
    ):
        if viewer:
            from dream_robot.agent.viewer import find_display

            find_display(viewer)                   # before the sim build imports glfw
        with _quiet():
            self.env = make_env()
            self._obs = self.env.reset()           # warm-up: learn the cameras
        self.viewer = None
        if viewer and hasattr(self.env, "mujoco"):
            from dream_robot.agent.viewer import can_open_window, make_viewer

            if can_open_window():
                self.viewer = make_viewer(self.env, viewer)
                self.viewer.open()
            else:
                print("viewer: no display found, running headless", file=sys.stderr)
        self.cameras: tuple[str, ...] = (SCENE, *self._obs.images)
        self.arm = _arm_skills(self.env)
        if self.arm is not None:
            self.arm.on_reset()
        self._last: StepResult | None = None
        self._steps = 0
        self._seed: int | None = None
        self._score_log = score_log
        self._started = False                     # a real episode (not the warm-up) is running
        self.recorder = (
            recorder.Recorder(self.env.frame, self.env.control_hz, out_dir=clips_dir)
            if hasattr(self.env, "frame") else None
        )

    def reset(self, seed: int | None) -> dict[str, Any]:
        self.log_score()                           # the episode that is ending
        with _quiet():
            self._obs = self.env.reset(seed=seed)
        if self.arm is not None:
            self.arm.on_reset()
        if self.viewer is not None:
            self.viewer.open()                     # the reset built a new model
        self._last, self._steps, self._seed = None, 0, seed
        self._started = True
        if self.recorder is not None:
            self.recorder.new_episode(f"s{seed}" if seed is not None else "unseeded")
        return self.status()

    def log_score(self) -> None:
        """Append the referee's verdict on the current episode to ``score_log``.

        For a driver that talks to this server over stdio (Claude Code, an LLM): the
        verdict goes to a file, not through any tool, and is read only after the run.
        Written after every skill and at reset, so the file is never behind; a reader
        takes the last line per episode.
        """
        if not (self._score_log and self._started):
            return
        verdict = {"seed": self._seed, "instruction": getattr(self.env, "prompt", None),
                   **self.score()}
        with open(self._score_log, "a") as f:
            f.write(json.dumps(verdict) + "\n")

    def frame(self, camera: str) -> np.ndarray:
        if camera == SCENE:
            return np.asarray(self.env.render())
        return np.asarray(self._obs.images[camera])

    def find_by_colour(self, colour: str) -> tuple[dict[str, Any], np.ndarray]:
        frame = self.env.rgbd("top")
        found = find(frame, colour_mask(frame.rgb, colour))
        # One shape for every outcome, so a planner reads the same fields each time.
        candidates = [
            {"id": i, "point": [round(float(c), 4) for c in f.point], "pixels": f.pixels,
             "grasp": {"point": [round(float(c), 4) for c in f.point - [0, 0, CUBE_TOP_TO_CENTRE]],
                       "yaw_deg": None, "above_table": None}}
            for i, f in enumerate(found, start=1)
        ]
        result = {
            "colour": colour,
            "found": len(found) == 1,
            "ambiguous": len(found) > 1,
            "point": candidates[0]["point"] if len(found) == 1 else None,
            "grasp": candidates[0]["grasp"] if len(found) == 1 else None,
            "candidates": candidates,
        }
        return result, overlay(frame, found)

    def find_objects(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        frame = self.env.rgbd("top")
        things = tabletop.find_objects(frame, self.env.table_xy, self.env.robot_mask("top"))
        return [t.as_dict(i) for i, t in enumerate(things, start=1)], tabletop.overlay(frame, things)

    def step(self, action: np.ndarray) -> tuple[Any, bool]:
        """The one place the episode advances. Skills step through here and get the
        observation and whether time is up -- not the env's success or termination."""
        self._check_running()
        self._last = self.env.step(action)
        self._obs = self._last.observation
        self._steps += 1
        if self.recorder is not None:
            self.recorder.tick()
        if self.viewer is not None:
            self.viewer.sync()
        return self._obs, self.time_up

    @property
    def time_up(self) -> bool:
        """The episode's only end: the clock. Not the task being done."""
        horizon = getattr(self.env, "horizon_steps", None)
        if horizon is not None:
            return self._steps >= horizon
        # No clock exposed (RoboCasa, parked): fall back on the env's own end,
        # which does leak success. Fix when that env is driven again.
        return bool(self._last and (self._last.terminated or self._last.truncated))

    def pick(self, grasp: GraspIn) -> tuple[dict[str, Any], np.ndarray]:
        from dream_robot.sims.robosuite.skills import Grasp

        self._check_running()
        yaw = None if grasp.yaw_deg is None else float(np.deg2rad(grasp.yaw_deg))
        g = Grasp(np.asarray(grasp.point, dtype=float), yaw, grasp.above_table)
        return self._recorded("pick", lambda: self.arm.pick(g, self._obs, self.step))

    def place_at(self, point: list[float], yaw_deg: float | None) -> tuple[dict[str, Any], np.ndarray]:
        self._check_running()
        yaw = None if yaw_deg is None else float(np.deg2rad(yaw_deg))
        try:
            return self._recorded("place_at", lambda: self.arm.place_at(
                np.asarray(point, dtype=float), self._obs, self.step, yaw=yaw))
        except ValueError as exc:
            raise ToolError(str(exc)) from exc

    def home(self) -> tuple[dict[str, Any], np.ndarray]:
        self._check_running()
        return self._recorded("home", lambda: self.arm.home(self._obs, self.step))

    def _recorded(self, skill: str, run) -> tuple[dict[str, Any], np.ndarray]:
        """Run a skill with the cameras recording. Returns its result plus the clip's
        summary, and a grid of scene frames across the clip (the last is the end state)."""
        if self.recorder is None:
            result = run()
            self.log_score()
            return result.as_dict(), np.asarray(self.env.render())
        self.recorder.start(skill)
        try:
            result = run()
        finally:
            clip = self.recorder.stop()
        self.log_score()
        return {**result.as_dict(), **clip.summary()}, recorder.grid(clip, "scene")

    def review_clip(self, clip: str | int | None, camera: str, start: float | None,
                    end: float | None, frames: int, crop: list[float] | None) -> np.ndarray:
        clips = list(self.recorder.clips.values()) if self.recorder else []
        if not clips:
            raise ToolError("no clips yet this episode: pick or place_at records one")
        if clip is None:
            chosen = clips[-1]                         # the latest
        elif isinstance(clip, int) or str(clip).isdigit():
            n = int(clip)
            if not 1 <= n <= len(clips):
                raise ToolError(f"clip number {n}: this episode has {len(clips)} (1-{len(clips)})")
            chosen = clips[n - 1]
        else:
            matches = [c for c in clips if c.id == clip]
            if not matches:
                raise ToolError(f"no clip {clip!r} this episode; have {[c.id for c in clips]}")
            chosen = matches[0]
        return recorder.grid(chosen, camera, start=start or 0.0, end=end,
                             n=frames, crop=None if crop is None else tuple(crop))

    def _check_running(self) -> None:
        if self.time_up:
            raise ToolError("time is up: call reset")

    def paint(self, colors: tuple[str, str]) -> None:
        self._obs = self.env.paint(colors)
        if self.viewer is not None:
            self.viewer.sync(pace=False)

    def status(self) -> dict[str, Any]:
        horizon = getattr(self.env, "horizon_steps", None)
        return {
            "instruction": getattr(self.env, "prompt", None),
            "seed": self._seed,
            "steps": self._steps,
            "steps_left": None if horizon is None else max(0, horizon - self._steps),
            "time_up": self.time_up,
            "clips": [
                {"n": i, "clip": c.id, "skill": c.skill, "seconds": round(c.seconds, 2)}
                for i, c in enumerate(self.recorder.clips.values(), start=1)
            ] if self.recorder else [],
        }

    def _diagnose(self):
        # T5's _diagnose takes the cube's position; T7's reads its own target.
        if hasattr(self.env, "instruction_failure"):
            return self.env._diagnose(self.env.cube_pos)
        return self.env._diagnose()

    def score(self) -> dict[str, Any]:
        """The referee's verdict, from the sim's ground truth. Never a tool."""
        last = self._last
        diagnose = getattr(self.env, "_diagnose", None)
        success = bool(last and last.success)
        verdict = {
            "success": success,
            "failure_mode": "none" if success or diagnose is None else str(self._diagnose()),
            "steps": self._steps,
        }
        if hasattr(self.env, "target_pos"):         # T7: lets a scorer check the naming
            # Where it started (spawn, table-centred = world xy), not where it is now.
            verdict["target_xy"] = [round(float(c), 4) for c in self.env.target.xy]
            verdict["target_kind"] = self.env.target.kind
        return verdict


def build_server(
    make_env: Callable[[], Env], *, debug: bool = False, viewer: str | None = None,
    score_log: str | None = None, clips_dir: str | None = None,
) -> MCPServer:
    """The MCP server. See ``build_scored_server`` for the same plus a referee."""
    return build_scored_server(
        make_env, debug=debug, viewer=viewer, score_log=score_log, clips_dir=clips_dir
    )[0]


def build_scored_server(
    make_env: Callable[[], Env], *, debug: bool = False, viewer: str | None = None,
    score_log: str | None = None, clips_dir: str | None = None,
) -> tuple[MCPServer, Callable[[], dict[str, Any]]]:
    """The server, and ``score()``: the referee's verdict on the current episode.

    ``score`` is for whoever evaluates the driver (a script, a test), in the same
    process -- it is deliberately not a tool, so no driver can call it.

    ``debug`` adds tools that edit the world, for building test scenes by hand.
    Never give them to a planner: a planner that can repaint the scene can
    "succeed" by changing the task instead of doing it.

    ``viewer`` ("sharp" or "docked", see viewer.py) opens a live window on this
    sim and paces it to real time -- for a human watching, never for an eval."""
    sim = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
    with _quiet():
        session = sim.submit(
            lambda: RobotSession(
                make_env, viewer=viewer, score_log=score_log, clips_dir=clips_dir
            )
        ).result()
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

            Knows colour only -- not shape, size or position. ``found``, ``point`` and
            ``grasp`` are set only when exactly one object matches. When several do,
            ``ambiguous`` is true and they are null: choose from ``candidates`` (each with
            a 3D point in metres, top-face centre, and a grasp), whose ids are drawn on the
            returned image. Its grasps assume a ~4 cm cube; find_objects measures instead.
            """
            result, img = session.find_by_colour(colour)
            return [result, _png(img)]

    if all(hasattr(session.env, a) for a in ("rgbd", "table_xy", "robot_mask")):

        @server.tool()
        @on_sim
        def find_objects() -> list:
            """Everything on the table, from the top camera's depth: no colour or name needed.

            Entries are in image order, left to right as the top camera sees them. Each has
            an ``id`` (drawn on the returned image), ``kind`` ("object" or
            "container"), ``centre`` (top centre, metres), ``size`` (footprint length and
            width, height), ``footprint`` ("round" / "rectangular"), ``top`` ("flat" /
            "domed") and ``colour``. Objects carry a ``grasp``: ``point`` where the finger
            pads close, ``yaw_deg`` the world direction they close along (across the narrow
            side), ``width``, and ``too_wide`` if it may not fit the 8 cm gripper.
            Containers carry ``drop_point`` (centre of the opening, rim height) and
            ``opening`` (inside size). Things touching each other come back as one entry.
            """
            things, img = session.find_objects()
            return [{"things": things}, _png(img)]

    if session.arm is not None:

        @server.tool()
        @on_sim
        def pick(grasp: GraspIn) -> list:
            """Turn the wrist, grasp, and lift, as ``grasp`` says.

            Pass an object's ``grasp`` from find_objects (or find_by_colour) as it is.
            Returns ok, or failure: "no_grasp" (closed on nothing), "timeout" (a motion
            phase stalled). ``holding`` is read from the gripper's width. The cameras
            record throughout: returns the ``clip`` id and a grid of 8 scene frames across
            it, the last being the end state. ``review_clip`` shows more.
            """
            result, img = session.pick(grasp)
            return [result, _png(img)]

        @server.tool()
        @on_sim
        def place_at(
            point: Point,
            yaw_deg: Annotated[float | None, Field(
                description="world direction for the held object's long side, turned to "
                "while carrying; null keeps it as picked")] = None,
        ) -> list:
            """Carry what the gripper holds above ``point``, lower it until its bottom is just
            over the point, release, and retreat.

            ``yaw_deg`` turns the held object first, so it fits: a container's ``opening``
            gives its side direction (``axis_yaw_deg``) and ``diagonal``. An object longer
            than the opening's side may still fit along the diagonal (side +- 45 deg for a
            square opening).

            ``point`` is where the object should land: a container's ``drop_point`` from
            find_objects, or a spot on the table. Returns ok, or failure: "dropped" (lost
            it on the way), "no_grasp" (nothing was held), "timeout". The cameras record
            throughout: returns the ``clip`` id and a grid of 8 scene frames across it, the
            last being the end state. ``review_clip`` shows more.
            """
            result, img = session.place_at(point, yaw_deg)
            return [result, _png(img)]

        @server.tool()
        @on_sim
        def home() -> list:
            """Lift the arm straight up, then return to its start pose.

            Use to recover: after a failed pick (the hand may still be pressed on the
            object), or to clear the cameras' view. Keeps the gripper as it is -- if
            holding something, it still is. Recorded like pick and place_at.
            """
            result, img = session.home()
            return [result, _png(img)]

        if session.recorder is not None:
            ClipCamera = Literal[recorder.CAMERAS]

            @server.tool()
            @on_sim
            def review_clip(
                clip: str | None = None,
                camera: ClipCamera = "scene",
                start: Annotated[float | None, Field(ge=0)] = None,
                end: Annotated[float | None, Field(ge=0)] = None,
                frames: Annotated[int, Field(ge=1, le=12)] = 8,
                crop: Annotated[list[float] | None, Field(min_length=4, max_length=4)] = None,
            ) -> Image:
                """More frames from a recorded skill (this episode's clips), to see what
                happened when: e.g. did the arm touch the box, is the bar on the rim.

                ``clip``: omit for the latest; or its number in this episode (1, 2, ...); or
                its id. ``status`` lists this episode's clips.

                ``camera``: "scene" (wide, front), "top" (overhead, angled), "wrist" (on the
                hand). ``start``/``end``: a window in seconds within the clip (default: all
                of it). ``frames``: how many, evenly spread, timestamped. ``crop``: [x0, y0,
                x1, y1] as fractions of the frame (0-1) to enlarge one region.
                """
                return _png(session.review_clip(clip, camera, start, end, frames, crop))

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
        """The instruction, steps taken, steps left before time runs out, and this episode's
        recorded clips. Says nothing about success: judge that from the images."""
        return session.status()

    return server, lambda: sim.submit(session.score).result()


def _arm_skills(env):
    """pick/place for envs that expose a robosuite arm; None for anything else.

    Imported lazily, as registry.py does: a server over a non-robosuite env must
    not need robosuite installed.
    """
    if not (hasattr(env, "mujoco") and hasattr(env, "arm_qpos_idx")):
        return None
    from dream_robot.sims.robosuite.skills import ArmSkills

    return ArmSkills(env)


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
    """Send everything written to stdout -- Python or C -- to stderr meanwhile.

    On a stdio server, stdout *is* the protocol. Python prints are not the only
    risk: MuJoCo's viewer prints "ERROR: could not initialize GLFW" from C, which
    no ``redirect_stdout`` sees. So this swaps file descriptor 1 itself. Once the
    stdio transport runs, the SDK has already moved the wire off fd 1; this
    matters most for the server build, which happens before that.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env", default="robosuite/instructed_pick_place")
    parser.add_argument("--debug", action="store_true", help="add scene-editing debug tools")
    parser.add_argument(
        "--viewer", nargs="?", const="own", choices=("own", "docked", "sharp"),
        help="live window, real-time pacing: own (our window: sharp, 1/3 wide, kept across "
        "resets; the default), docked (right third, a bit soft) or sharp (full resolution, "
        "GNOME maximises it)",
    )
    parser.add_argument(
        "--score-log", help="append the referee's verdict per episode to this JSONL file "
        "(on each reset and at exit); never exposed as a tool",
    )
    parser.add_argument("--clips-dir", help="also write each skill's camera clips as MP4s here")
    args = parser.parse_args()

    from dream_robot.core.registry import task_factory

    try:
        factory = task_factory(args.env)
    except KeyError as exc:
        parser.error(str(exc))
    server, _ = build_scored_server(
        factory, debug=args.debug, viewer=args.viewer, score_log=args.score_log,
        clips_dir=args.clips_dir,
    )
    server.run("stdio")
    # The transport has handed fd 1 back to the wire. C libraries flush their
    # buffered stdout at exit (the viewer's GL warnings), so point it at stderr.
    os.dup2(2, 1)


if __name__ == "__main__":
    main()
