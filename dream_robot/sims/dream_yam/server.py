"""Dream YAM Sim as an MCP server: the LLM plans in words, Dream-1 moves the arms, time runs on.

The driver never moves a joint. Its one motion tool is ``policy(command, seconds)``: Dream-1 drives
both arms on that command for that long, and the tool returns frames from across the run. The
driver splits a task into commands Dream-1 can do ("put apple into the bowl"), looks, and decides
what to command next. That is ROADMAP's planner/executor split with a learned executor.

**The sim runs on the wall clock** (realtime.py), as a real robot's world does: while Dream-1
answers and while the driver thinks, time passes and the arms hold their last target. Every
image is stamped with the episode time, so the driver can see what happened in between.

**No tool reports success,** as in dream_robot/agent/server.py: the referee's verdict (sim
positions) goes to ``--score-log`` only, after every tool and at reset.

**Paid calls are capped** per server (``--max-calls``): Dream-1 bills each call, so a driver that
loops cannot run up a bill. ``status`` says how many are left.

    MUJOCO_GL=egl envs/dream_yam/.venv/bin/python -m dream_robot.sims.dream_yam.server \
        --task "Put the apple and the orange in the bowl." --score-log scratch/dream1/referee.jsonl
"""

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage
from PIL import ImageDraw
from pydantic import Field

from dream_robot.agent.server import _on_sim_thread, _png, _quiet

INSTRUCTIONS = (Path(__file__).with_name("driver_prompt.md")).read_text()
STEP_S = 0.05
Camera = Literal["all", "top", "left_wrist", "right_wrist"]


def _grid(frames, n, width=960):
    """``n`` of (step, rgb) frames evenly across, stacked top to bottom, stamped with episode time."""
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).round().astype(int)
    rows = []
    for i in idx:
        step, rgb = frames[i]
        im = PILImage.fromarray(rgb)
        im = im.resize((width, round(im.height * width / im.width)))
        ImageDraw.Draw(im).text((6, 4), f"t={step * STEP_S:.1f}s", fill=(255, 255, 0))
        rows.append(np.asarray(im))
    return np.concatenate(rows, axis=0)


def _camera(images, camera):
    from dream_robot.sims.dream_yam.station import CAMERAS, tile

    return tile(images) if camera == "all" else images[CAMERAS[camera]]


def _crop(rgb, camera):
    """One camera's third of a stored tile (left wrist | top | right wrist)."""
    if camera == "all":
        return rgb
    w = rgb.shape[1] // 3
    col = {"left_wrist": 0, "top": 1, "right_wrist": 2}[camera]
    return rgb[:, col * w:(col + 1) * w]


class Session:
    """The state behind the tools; the sim itself runs on realtime.py's loop thread."""

    def __init__(self, task, max_calls, score_log, clips_dir, viewer=False):
        from dream_robot.sims.dream_yam.realtime import Realtime

        with _quiet():
            self.rt = Realtime(viewer=viewer, max_calls=max_calls, clips_dir=clips_dir)
        self.task, self.max_calls, self.score_log = task, max_calls, score_log
        self.episode = None
        self.clips = []                          # (what, start step, end step) this episode
        self.last_end = 0                        # step the last tool's motion ended

    def reset(self, seed):
        self.rt.reset()
        self.episode = time.strftime("%y%m%d-%H%M%S") + (f"-s{seed}" if seed is not None else "")
        self.clips, self.last_end = [], 0
        self.log("reset")
        return self.status()

    def _check(self):
        if self.episode is None:
            raise ToolError("call reset first")

    def policy(self, command, seconds, continue_run):
        self._check()
        idle = (self.rt.steps - self.last_end) * STEP_S
        try:
            run = self.rt.policy(command, seconds, continue_run)
        except ValueError as exc:
            raise ToolError(f"{exc}. Start it fresh (continue_run false)." if "continue" in str(exc)
                            else str(exc)) from exc
        end = self.rt.steps
        self.clips.append((f"policy{' (continued)' if continue_run else ''}: {command}", run.start_step, end))
        self.last_end = end
        result = {
            "command": command, "continued": continue_run,
            "time": f"t={run.start_step * STEP_S:.1f}s to t={end * STEP_S:.1f}s",
            "seconds_since_your_last_motion": round(idle, 1),
            "calls": run.calls, "calls_left": self.max_calls - self.rt.calls,
            "arms_held_waiting_s": round(run.held_steps * STEP_S, 1),
            "clip": len(self.clips),
        }
        if run.rtt_s:
            result["round_trip_s"] = round(statistics.median(run.rtt_s), 2)
            result["pacing"] = max(set(run.modes), key=run.modes.count)
        if run.restarts:
            result["session_restarted"] = run.restarts
        if run.waits:
            result["waited_for"] = sorted(set(run.waits))
        if run.error:
            result["error"] = run.error
        self.log("policy", command=command, continued=continue_run, calls=run.calls,
                 rtt_s=run.rtt_s, modes=run.modes, held_s=round(run.held_steps * STEP_S, 1),
                 idle_before_s=round(idle, 1), restarts=run.restarts, error=run.error)
        return result, _grid(self.rt.frames(run.start_step, end), 6)

    def home(self):
        self._check()
        start, end = self.rt.home()
        self.clips.append(("home", start, end))
        self.last_end = end
        self.log("home")
        return {"time": f"t={end * STEP_S:.1f}s", "clip": len(self.clips)}, _camera(self.rt.images(), "all")

    def look(self, camera):
        return _camera(self.rt.images(), camera)

    def review_clip(self, clip, camera, start, end, frames):
        if not self.clips:
            raise ToolError("no clips yet this episode")
        n = len(self.clips) if clip is None else clip
        if not 1 <= n <= len(self.clips):
            raise ToolError(f"clip {n}: this episode has {len(self.clips)} (1-{len(self.clips)})")
        _, a, b = self.clips[n - 1]
        lo = a + int((start or 0) / STEP_S)
        hi = b if end is None else min(b, a + int(end / STEP_S))
        picked = [(st, _crop(rgb, camera)) for st, rgb in self.rt.frames(lo, max(lo, hi))]
        if not picked:
            raise ToolError("no frames in that window")
        return _grid(picked, frames, width=960 if camera == "all" else 640)

    def status(self):
        return {
            "instruction": self.task,
            "time": f"t={self.rt.steps * STEP_S:.1f}s",
            "dream_calls_used": self.rt.calls,
            "dream_calls_left": self.max_calls - self.rt.calls,
            "clips": [{"n": i, "what": what, "from": f"t={a * STEP_S:.1f}s", "to": f"t={b * STEP_S:.1f}s"}
                      for i, (what, a, b) in enumerate(self.clips, start=1)],
        }

    def log(self, event, **extra):
        """The referee's verdict after each tool: for the score log only, never a tool."""
        if not self.score_log:
            return
        row = {"episode": self.episode, "event": event, "t": round(self.rt.steps * STEP_S, 2),
               "dream_calls": self.rt.calls, "loop_lag_s": round(self.rt.lag_s, 2), **extra,
               "score": self.rt.score()}
        Path(self.score_log).parent.mkdir(parents=True, exist_ok=True)
        with open(self.score_log, "a") as f:
            f.write(json.dumps(row) + "\n")


def build_server(task, max_calls=200, score_log=None, clips_dir=None, viewer=False):
    """The server and its session (the session is for tests and the referee, not a tool)."""
    tools = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tools")   # one tool at a time
    session = Session(task, max_calls, score_log, clips_dir, viewer)
    server = MCPServer("dream-yam", instructions=INSTRUCTIONS)
    one_at_a_time = _on_sim_thread(tools)

    @server.tool()
    @one_at_a_time
    def reset(seed: int | None = None) -> list:
        """Start a new episode: objects back in place, both arms home, the clock at t=0. Returns the
        instruction, status, and all three cameras. (The scene is the same every time; ``seed`` only
        labels it.)"""
        return [session.reset(seed), _png(session.look("all"))]

    @server.tool()
    @one_at_a_time
    def look(camera: Camera = "all") -> Image:
        """The robot's cameras right now. "all": left wrist | top | right wrist, side by side (left and
        right as the robot sees them); or one of them, larger."""
        return _png(session.look(camera))

    @server.tool()
    @one_at_a_time
    def policy(
        command: Annotated[str, Field(description="what the arms should do, in plain words, "
                                      "one short step, e.g. 'put apple into the bowl'")],
        seconds: Annotated[float, Field(ge=2.4, le=90, description="how long Dream-1 drives, "
                                        "seconds; about one paid call per 1.5-2.4 s")] = 15,
        continue_run: Annotated[bool, Field(description="true: go on with the last run, same "
                                            "command, Dream-1 keeps its memory of it; false: "
                                            "a fresh start")] = False,
    ) -> list:
        """Dream-1, a learned robot policy, drives both arms on ``command`` for ``seconds``; then
        the arms hold where they are. It sees the three cameras and the joints, nothing else.

        Time runs on, as for a real robot: between your calls the arms hold their last target
        while the world goes on (a held object may slip, a dropped one may roll).

        A fresh run (the default) starts Dream-1 anew: it does not remember earlier commands.
        ``continue_run`` goes on with the last run -- same command, no ``home`` or ``reset`` in
        between: Dream-1 has seen the time in between, arms holding.

        Returns the episode times it ran, how long it was since your last motion, calls used and
        left, and 6 frames from across the run (stamped with episode time, all three cameras each),
        the last one the end. Nothing here says whether it worked: look."""
        result, grid = session.policy(command, seconds, continue_run)
        return [result, _png(grid)]

    @server.tool()
    @one_at_a_time
    def home() -> list:
        """Both arms back to their home pose (straight, grippers open) over 2 s. Objects stay where
        they are; one held in a gripper is let go there. Free: no Dream-1 call."""
        result, frame = session.home()
        return [result, _png(frame)]

    @server.tool()
    @one_at_a_time
    def review_clip(
        clip: Annotated[int | None, Field(description="its number from status; omit for the latest")] = None,
        camera: Camera = "all",
        start: Annotated[float | None, Field(ge=0)] = None,
        end: Annotated[float | None, Field(ge=0)] = None,
        frames: Annotated[int, Field(ge=1, le=12)] = 8,
    ) -> Image:
        """More frames (5 a second are kept) from an earlier policy run or home move this episode,
        to see what happened when (did it close on the object, did it slip). ``start``/``end`` in
        seconds from the clip's start."""
        return _png(session.review_clip(clip, camera, start, end, frames))

    @server.tool()
    @one_at_a_time
    def status() -> dict[str, Any]:
        """The instruction, the episode time, Dream-1 calls used and left, and this episode's clips.
        Says nothing about success: judge that from the images."""
        return session.status()

    return server, session


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, help="the instruction reset hands the driver")
    parser.add_argument("--max-calls", type=int, default=200, help="paid Dream-1 calls, at most")
    parser.add_argument("--score-log", help="append the referee's verdict after every tool (JSONL)")
    parser.add_argument("--clips-dir", help="write each episode's video (all three cameras) and trace here")
    parser.add_argument("--viewer", action="store_true",
                        help="live MuJoCo window with the three cameras (to watch)")
    args = parser.parse_args()
    server, session = build_server(args.task, args.max_calls, args.score_log, args.clips_dir, args.viewer)
    try:
        server.run("stdio")
    finally:
        session.rt.close()                       # video, traces and the Dream session closed
        # Interpreter teardown with the viewer's GLFW and the renderer's EGL contexts both alive
        # dumps core; everything worth keeping is written, so leave without it.
        os._exit(0)


if __name__ == "__main__":
    main()
