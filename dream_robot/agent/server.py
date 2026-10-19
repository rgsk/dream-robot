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
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal

import imageio.v3 as iio
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel, Field

from dream_robot.agent import recorder, tabletop
from dream_robot.core.env_api import Env, StepResult

# The base prompt for whoever drives the robot -- an LLM, Claude Code, a person. Sent as
# the MCP server's instructions, so every client gets the same one. Edit the .md file.
INSTRUCTIONS = (Path(__file__).with_name("driver_prompt.md")).read_text()

# [x, y, z] in metres, world frame.
Point = Annotated[list[float], Field(min_length=3, max_length=3)]




class MoveToStep(BaseModel):
    point: Point | None = Field(None, description="as move_to: finger pads to here; null stays put")
    yaw_deg: float | None = Field(None, description="as move_to: finger direction; null keeps it")


class MoveByStep(BaseModel):
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    dyaw_deg: float = 0.0


class Step(BaseModel):
    """One primitive: exactly one of the three fields."""

    move_to: MoveToStep | None = None
    move_by: MoveByStep | None = None
    gripper: Literal["open", "close"] | None = None


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
            # Stamped with the reset time: the same seed run again must not overwrite
            # the earlier run's clip files (s2030-1-move_to once collided across runs).
            stamp = time.strftime("%y%m%d-%H%M%S")
            self.recorder.new_episode(f"s{seed}-{stamp}" if seed is not None else stamp)
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

    def find_objects(self) -> tuple[list[dict[str, Any]], np.ndarray]:
        frame = self.env.rgbd("top")
        things = tabletop.find_objects(frame, self.env.table_xy, self.env.robot_mask("top"))
        listed = [t.as_dict(i) for i, t in enumerate(things, start=1)]
        return listed, tabletop.overlay(frame, things)

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

    def move_to(
        self, point: list[float] | None, yaw_deg: float | None
    ) -> tuple[dict[str, Any], np.ndarray]:
        self._check_running()
        if point is None and yaw_deg is None:
            raise ToolError("give a point, a yaw_deg, or both")
        p = None if point is None else np.asarray(point, dtype=float)
        yaw = None if yaw_deg is None else float(np.deg2rad(yaw_deg))
        return self._recorded("move_to", lambda: self.arm.move_to(p, self._obs, self.step, yaw=yaw))

    def move_by(self, delta: list[float], dyaw_deg: float) -> tuple[dict[str, Any], np.ndarray]:
        self._check_running()
        dyaw = float(np.deg2rad(dyaw_deg))
        return self._recorded("move_by", lambda: self.arm.move_by(
            np.asarray(delta, dtype=float), self._obs, self.step, dyaw=dyaw))

    def gripper(self, action: str) -> tuple[dict[str, Any], np.ndarray]:
        self._check_running()
        close = action == "close"
        return self._recorded("gripper", lambda: self.arm.gripper(close, self._obs, self.step))

    def sequence(self, steps: list[Step]) -> tuple[dict[str, Any], np.ndarray]:
        """Primitives in order, as one recorded clip. Stops at the first step that
        fails (blocked, wrist_limit, timeout), closes on nothing, or runs out of time."""
        self._check_running()
        if not steps:
            raise ToolError("give at least one step")
        runs = []
        for i, st in enumerate(steps, start=1):    # check them all before anything moves
            given = [k for k in ("move_to", "move_by", "gripper") if getattr(st, k) is not None]
            if len(given) != 1:
                raise ToolError(f"step {i}: give exactly one of move_to, move_by, gripper")
            kind = given[0]
            if kind == "move_to":
                p, yaw = st.move_to.point, st.move_to.yaw_deg
                if p is None and yaw is None:
                    raise ToolError(f"step {i}: move_to needs a point, a yaw_deg, or both")
                p = None if p is None else np.asarray(p, dtype=float)
                yaw = None if yaw is None else float(np.deg2rad(yaw))
                runs.append(("move_to", lambda p=p, yaw=yaw: self.arm.move_to(
                    p, self._obs, self.step, yaw=yaw)))
            elif kind == "move_by":
                m = st.move_by
                runs.append(("move_by", lambda m=m: self.arm.move_by(
                    np.array([m.dx, m.dy, m.dz]), self._obs, self.step,
                    dyaw=float(np.deg2rad(m.dyaw_deg)))))
            else:
                runs.append((f"gripper {st.gripper}", lambda c=st.gripper == "close":
                             self.arm.gripper(c, self._obs, self.step)))

        def run() -> dict[str, Any]:
            done, stop, total = [], None, 0
            for i, (name, fn) in enumerate(runs, start=1):
                d = self._result(fn())
                total += d["steps"]
                hand = d["hand"]
                done.append({"step": i, "action": name, "phase": d["phase"],
                             "point": hand["point"], "holding": hand["holding"]})
                if not d["ok"] or d["failure"] != "none":
                    stop = d["failure"] if d["failure"] != "none" else d["phase"]
                elif name == "gripper close" and not hand["holding"]:
                    stop = "nothing_held"
                elif d["episode_over"] or self.time_up:
                    stop = "time_up"
                if stop:
                    done[-1]["failure"] = stop
                    break
            return {"ok": stop is None, "completed": len(done) - (stop is not None),
                    "of": len(runs), "stopped": stop or "none", "steps": total,
                    "done": done, "hand": self.arm.hand(self._obs)}

        return self._recorded("sequence", run, summarize=lambda d: d)

    def _recorded(self, skill: str, run, summarize=None) -> tuple[dict[str, Any], np.ndarray]:
        """Run a skill with the cameras recording. Returns its result plus the clip's
        summary, and a grid of scene frames across the clip (the last is the end state)."""
        summarize = summarize or self._result
        if self.recorder is None:
            result = run()
            self.log_score()
            return summarize(result), np.asarray(self.env.render())
        self.recorder.start(skill)
        try:
            result = run()
        finally:
            clip = self.recorder.stop()
        self.log_score()
        return {**summarize(result), **clip.summary()}, recorder.grid(clip, "scene")

    def _result(self, result) -> dict[str, Any]:
        d = result.as_dict()
        if d["phase"] in ("blocked", "wrist_limit"):
            # blocked: stalled short of the target, pressing on something.
            # wrist_limit: the asked turn is past the wrist's stops (see hand.wrist_room_deg).
            d["failure"] = d["phase"]
        if self.arm is not None:
            d["hand"] = self.arm.hand(self._obs)
        return d

    def review_clip(self, clip: str | int | None, camera: str, start: float | None,
                    end: float | None, frames: int, crop: list[float] | None) -> np.ndarray:
        clips = list(self.recorder.clips.values()) if self.recorder else []
        if not clips:
            raise ToolError("no clips yet this episode: every motion records one")
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

    def status(self) -> dict[str, Any]:
        horizon = getattr(self.env, "horizon_steps", None)
        return {
            "instruction": getattr(self.env, "prompt", None),
            "seed": self._seed,
            "steps": self._steps,
            "steps_left": None if horizon is None else max(0, horizon - self._steps),
            "time_up": self.time_up,
            **({"hand": self.arm.hand(self._obs)} if self.arm is not None else {}),
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
            if self.env.instruction.placements:     # all_objects: each goal, met or not
                verdict["goals"] = [
                    {"kind": self.env.objects[i].kind, "side": side,
                     "in": self.env.bin_of(i) == self.env.bin_names.index(side)}
                    for i, side in self.env.instruction.placements
                ]
        return verdict


def build_server(
    make_env: Callable[[], Env], *, viewer: str | None = None,
    score_log: str | None = None, clips_dir: str | None = None,
) -> MCPServer:
    """The MCP server. See ``build_scored_server`` for the same plus a referee."""
    return build_scored_server(
        make_env, viewer=viewer, score_log=score_log, clips_dir=clips_dir
    )[0]


def build_scored_server(
    make_env: Callable[[], Env], *, viewer: str | None = None,
    score_log: str | None = None, clips_dir: str | None = None,
) -> tuple[MCPServer, Callable[[], dict[str, Any]]]:
    """The server, and ``score()``: the referee's verdict on the current episode.

    ``score`` is for whoever evaluates the driver (a script, a test), in the same
    process -- it is deliberately not a tool, so no driver can call it.

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
        def move_to(
            point: Annotated[Point | None, Field(
                description="where the finger pads go, metres, world frame; null stays put"
            )] = None,
            yaw_deg: Annotated[float | None, Field(
                description="world direction from the right finger to the left, -180..180; "
                "null keeps the wrist"
            )] = None,
        ) -> list:
            """Move the hand so the finger pads are at ``point``, with the fingers pointing
            along ``yaw_deg``. Either may be left out: ``yaw_deg`` alone turns on the spot.
            The gripper stays open or closed as it is.

            ``yaw_deg`` is a direction: 30 and 210 are different poses (the same grasp, as
            the fingers are symmetric). The arm turns whichever way keeps the wrist inside
            its stops; if neither way can reach it, nothing moves and failure is
            "wrist_limit" -- ``hand.wrist_room_deg`` says how far it can turn each way.
            Stops by itself if the hand stalls short of the target -- pressing on the table
            or an object -- with failure "blocked"; ``hand`` then says where it stopped.
            """
            result, img = session.move_to(point, yaw_deg)
            return [result, _png(img)]

        @server.tool()
        @on_sim
        def move_by(
            dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, dyaw_deg: float = 0.0,
        ) -> list:
            """Move the hand by (dx, dy, dz) metres from where it is now, and turn the wrist
            by exactly ``dyaw_deg`` degrees about vertical, always that way round (positive =
            counter-clockwise seen from above; 0 keeps it). A turn past the wrist's stops
            goes as far as the stop and reports failure "wrist_limit"; ``hand.wrist_room_deg``
            says how much room is left. Same "blocked" stop as ``move_to``."""
            result, img = session.move_by([dx, dy, dz], dyaw_deg)
            return [result, _png(img)]

        @server.tool()
        @on_sim
        def gripper(action: Literal["open", "close"]) -> list:
            """Open or close the fingers where the hand is, and wait until they stop. After
            "close", ``hand.holding`` says whether something is between them and
            ``hand.opening_m`` how wide."""
            result, img = session.gripper(action)
            return [result, _png(img)]

        @server.tool()
        @on_sim
        def sequence(steps: list[Step]) -> list:
            """Several primitives in one call, run in order: each step is exactly one of
            ``{"move_to": {"point": [x, y, z], "yaw_deg": a}}``, ``{"move_by": {"dz": 0.2}}``
            or ``{"gripper": "close"}``, meaning what those tools mean.

            Stops at the first step that fails ("blocked", "wrist_limit"), a "close" that
            ends holding nothing ("nothing_held"), or time running out; the rest do not run.
            Returns each step done (phase, pad point, holding), where and why it stopped,
            the final ``hand``, and one grid of frames across the whole sequence, recorded
            as one clip (``review_clip`` reads it)."""
            result, img = session.sequence(steps)
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

    @server.tool()
    @on_sim
    def status() -> dict[str, Any]:
        """The instruction, steps taken, steps left before time runs out, and this episode's
        recorded clips. Says nothing about success: judge that from the images."""
        return session.status()

    return server, lambda: sim.submit(session.score).result()


def _arm_skills(env):
    """The arm primitives for envs that expose a robosuite arm; None for anything else.

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
    parser.add_argument(
        "--horizon-seconds", type=float,
        help="episode clock for this server, overriding the task's (e.g. 300 for driving "
        "by hand); evals keep the task's",
    )
    parser.add_argument(
        "--all-objects", action="store_true",
        help="T7: the instruction names a bin for every object, and all must land",
    )
    parser.add_argument("--clips-dir", help="also write each skill's camera clips as MP4s here")
    args = parser.parse_args()

    from dream_robot.core.registry import task_factory

    try:
        factory = task_factory(args.env)
    except KeyError as exc:
        parser.error(str(exc))
    if args.horizon_seconds is not None:
        # Only tasks whose constructor takes it; others fail loudly at startup.
        factory = functools.partial(factory, horizon_seconds=args.horizon_seconds)
    if args.all_objects:
        factory = functools.partial(factory, all_objects=True)
    server, _ = build_scored_server(
        factory, viewer=args.viewer, score_log=args.score_log,
        clips_dir=args.clips_dir,
    )
    server.run("stdio")
    # The transport has handed fd 1 back to the wire. C libraries flush their
    # buffered stdout at exit (the viewer's GL warnings), so point it at stderr.
    os.dup2(2, 1)


if __name__ == "__main__":
    main()
