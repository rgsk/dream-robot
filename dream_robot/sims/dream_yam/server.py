"""Dream YAM Sim as an MCP server: the LLM plans in words, Dream-1 moves the arms, time runs on.

The driver never moves a joint. It supervises: ``policy(command)`` starts Dream-1 on a command and
returns at once; Dream-1 keeps driving both arms on it while the driver ``watch``es, ``look``s and
thinks, until the driver ``stop``s it or sends the next command. The driver splits a task into
commands Dream-1 can do ("put apple into the bowl") and decides when each is done. That is
ROADMAP's planner/executor split with a learned executor that never waits for the planner.

**The sim runs on the wall clock** (realtime.py), as a real robot's world does. Every image is
stamped with the episode time. A run nobody checks on for ``--watchdog`` seconds stops itself.

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
    if not frames:
        raise ToolError("no frames in that stretch of time")
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

    def __init__(self, task, max_calls, score_log, clips_dir, viewer=False, watchdog_s=60.0):
        from dream_robot.sims.dream_yam.realtime import Realtime

        with _quiet():
            self.rt = Realtime(viewer=viewer, max_calls=max_calls, clips_dir=clips_dir,
                               watchdog_s=watchdog_s)
        self.task, self.max_calls, self.score_log = task, max_calls, score_log
        self.episode = None
        self.runs = []                           # every run this episode, in order
        self.seen = 0                            # grip events the driver has been told of

    def t(self, step=None):
        return f"t={(self.rt.steps if step is None else step) * STEP_S:.1f}s"

    def reset(self, seed):
        self.rt.touch()
        self.rt.reset()
        self.episode = time.strftime("%y%m%d-%H%M%S") + (f"-s{seed}" if seed is not None else "")
        self.runs = []
        self.seen = 0
        self.log("reset")
        return self.status()

    def _check(self):
        self.rt.touch()
        if not self.rt.alive:
            raise ToolError("the robot's control loop has stopped (a server fault, not yours): "
                            "nothing more can be done in this episode; report the state as it is")
        if self.episode is None:
            raise ToolError("call reset first")

    def _run_summary(self, run):
        d = {"command": run.prompt, "from": self.t(run.start_step),
             "to": self.t(run.stop_step) if run.stop_step is not None else "running",
             "calls": run.calls, "arms_held_waiting_s": round(run.held_steps * STEP_S, 1)}
        if run.stopped_by:
            d["stopped_by"] = run.stopped_by
        if run.rtt_s:
            d["round_trip_s"] = round(statistics.median(run.rtt_s), 2)
            d["pacing"] = max(set(run.modes), key=run.modes.count)
        if run.restarts:
            d["session_restarted"] = run.restarts
        if run.error:
            d["error"] = run.error
        if getattr(run, "rejected", 0):
            d["chunks_rejected"] = run.rejected
        return d

    def policy(self, command, continue_run):
        self._check()
        try:
            run, already = self.rt.policy(command, continued=continue_run)
        except ValueError as exc:
            raise ToolError(f"{exc}. Start it fresh (continue_run false)." if "continue" in str(exc)
                            else str(exc)) from exc
        if not already:
            self.runs.append(run)
            self.seen = len(self.rt.events)      # a new step: earlier grasps / releases are old news
        self.log("policy", command=command, continued=continue_run, already=already)
        return {"running": command, "since": self.t(run.start_step), "now": self.t(),
                **({"note": "already running this command"} if already else {}),
                "calls_left": self.max_calls - self.rt.calls}

    def watch(self, seconds):
        """Let time run ``seconds`` (Dream-1 driving or not); frames from that window."""
        self._check()
        a, calls0 = self.rt.steps, self.rt.calls
        deadline = time.monotonic() + seconds
        run = self.rt.run
        while time.monotonic() < deadline:
            time.sleep(STEP_S)
            self.rt.touch()                      # watching is attending: the watchdog stays quiet
            if run is not None and run.done.is_set():
                break                            # it stopped by itself: say so now
        b = self.rt.steps
        now = self.rt.run
        result = {"from": self.t(a), "to": self.t(b), "calls": self.rt.calls - calls0,
                  "calls_left": self.max_calls - self.rt.calls,
                  "dream_1": f"running: {now.prompt}" if now is not None else "stopped"}
        if run is not None and run.done.is_set():
            result["it_stopped"] = self._run_summary(run)
        self.log("watch", seconds=seconds)
        return result, _grid(self.rt.frames(a, b), 6)

    def holding(self):
        g = self.rt.grip
        return {arm: (f"something, {g.width[arm]} open" if g.holding[arm] else "nothing")
                for arm in ("left", "right")}

    def wait(self, max_seconds):
        """Until a gripper event (grasp or release), Dream-1 stopping by itself, or ``max_seconds``.

        Events the driver has not been told of count, including ones while it was thinking since
        its last tool: grasp to release takes ~3.5 s, about one LLM turn, so a wait that counted
        only events after its own start slept through releases. Those return at once."""
        self._check()
        a, calls0, seen = self.rt.steps, self.rt.calls, self.seen
        run = self.rt.run
        deadline = time.monotonic() + max_seconds
        why = "time"
        while time.monotonic() < deadline:
            time.sleep(STEP_S)
            self.rt.touch()
            if len(self.rt.events) > seen:
                why = "event"
                wait_land = 0.5 - (self.rt.steps - self.rt.events[-1][0]) * STEP_S
                time.sleep(max(0.0, wait_land))  # let a released object land before the picture
                break
            if run is not None and run.done.is_set():
                why = "stopped"
                break
        new = self.rt.events[seen:]
        self.seen = len(self.rt.events)
        if new and new[0][0] < a:
            why = "event (before this wait: while you were thinking)"
        result = {"from": self.t(a), "to": self.t(), "woke_on": why,
                  "events": [{"t": self.t(st), "arm": arm, "what": kind, "opening": o}
                             for st, arm, kind, o in new],
                  "holding": self.holding(), "calls": self.rt.calls - calls0,
                  "calls_left": self.max_calls - self.rt.calls,
                  "dream_1": f"running: {self.rt.run.prompt}" if self.rt.run is not None else "stopped"}
        if why == "stopped":
            result["it_stopped"] = self._run_summary(run)
        self.log("wait", woke_on=why, events=result["events"])
        return result, _camera(self.rt.images(), "top")

    def stop(self):
        self._check()
        run = self.rt.stop()
        self.log("stop")
        result = self._run_summary(run) if run is not None else {"note": "Dream-1 was not running"}
        return result, _camera(self.rt.images(), "all")

    def home(self):
        self._check()
        run = self.rt.run
        start, end = self.rt.home()
        self.log("home")
        result = {"from": self.t(start), "to": self.t(end)}
        if run is not None:
            result["stopped"] = self._run_summary(run)
        return result, _camera(self.rt.images(), "all")

    def look(self, camera):
        self.rt.touch()
        return _camera(self.rt.images(), camera)

    def review(self, start, end, camera, frames):
        self.rt.touch()
        lo = int(start / STEP_S)
        hi = self.rt.steps if end is None else int(end / STEP_S)
        picked = [(st, _crop(rgb, camera)) for st, rgb in self.rt.frames(lo, max(lo, hi))]
        if not picked:
            raise ToolError(f"no frames between t={start}s and t={end}s (now {self.t()})")
        return _grid(picked, frames, width=960 if camera == "all" else 640)

    def status(self):
        self.rt.touch()
        now = self.rt.run
        return {
            "instruction": self.task,
            "now": self.t(),
            "dream_1": f"running: {now.prompt}" if now is not None else "stopped",
            "holding": self.holding(),
            "dream_calls_used": self.rt.calls,
            "dream_calls_left": self.max_calls - self.rt.calls,
            "runs": [self._run_summary(r) for r in self.runs],
        }

    def log(self, event, **extra):
        """The referee's verdict after each tool: for the score log only, never a tool."""
        if not self.score_log:
            return
        run = self.rt.run
        row = {"episode": self.episode, "event": event, "t": round(self.rt.steps * STEP_S, 2),
               "dream_calls": self.rt.calls, "running": run.prompt if run else None,
               "loop_lag_s": round(self.rt.lag_s, 2), "loop_errors": self.rt.errors[-3:], **extra,
               "score": self.rt.score()}
        Path(self.score_log).parent.mkdir(parents=True, exist_ok=True)
        with open(self.score_log, "a") as f:
            f.write(json.dumps(row) + "\n")
        self.rt.save_traces()

    def finish(self):
        """At shutdown: the runs' timings into the score log, then close the sim."""
        self.log("end", runs=[self._run_summary(r) for r in self.runs])
        self.rt.close()


def build_server(task, max_calls=200, score_log=None, clips_dir=None, viewer=False, watchdog_s=60.0):
    """The server and its session (the session is for tests and the referee, not a tool)."""
    tools = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tools")   # one tool at a time
    session = Session(task, max_calls, score_log, clips_dir, viewer, watchdog_s)
    server = MCPServer("dream-yam", instructions=INSTRUCTIONS)
    one_at_a_time = _on_sim_thread(tools)

    @server.tool()
    @one_at_a_time
    def reset(seed: int | None = None) -> list:
        """Start a new episode: Dream-1 stopped, objects back in place, both arms home, the clock at
        t=0. Returns the instruction, status, and all three cameras. (The scene is the same every
        time; ``seed`` only labels it.)"""
        return [session.reset(seed), _png(session.look("all"))]

    @server.tool()
    @one_at_a_time
    def policy(
        command: Annotated[str, Field(description="what the arms should do, in plain words, "
                                      "one short step, e.g. 'put apple into the bowl'")],
        continue_run: Annotated[bool, Field(description="true: resume the stopped run of this "
                                            "same command, Dream-1 keeping its memory of it; "
                                            "false: a fresh start")] = False,
    ) -> dict[str, Any]:
        """Start Dream-1, a learned robot policy, on ``command``. Returns at once: Dream-1 keeps
        driving both arms on it -- while you watch, look and think -- until you ``stop`` it, send
        another command (which replaces it), ``home`` or ``reset``. It sees the three cameras and
        the joints, nothing else, and it does not know when the step is done: stop it.

        Sending the command that is already running changes nothing. A new command starts Dream-1
        fresh (it forgets the last one); ``continue_run`` resumes a stopped run of the same command.
        If no tool is called for 60 s while it runs, it stops by itself."""
        return session.policy(command, continue_run)

    @server.tool()
    @one_at_a_time
    def wait(
        max_seconds: Annotated[float, Field(ge=2, le=20, description="look again after this long "
                                           "at the latest")] = 10,
    ) -> list:
        """Wait a little, then see the table: returns the **top camera** now, as soon as a gripper
        takes hold of something ("grasp") or stops holding it ("release": let go, or it slipped
        out), or Dream-1 stops by itself, or ``max_seconds`` pass ("time"). Grasp and release
        come from the grippers' own sensors -- hints, not proof: a light grip can go unnoticed.
        **The image is the truth: every time, check in it whether the object is at its goal.**
        Dream-1 keeps driving meanwhile."""
        result, frame = session.wait(max_seconds)
        return [result, _png(frame)]

    @server.tool()
    @one_at_a_time
    def watch(
        seconds: Annotated[float, Field(ge=1, le=30, description="how long to watch")] = 10,
    ) -> list:
        """Let ``seconds`` pass and see all of them: 6 frames from that window (stamped with episode
        time, all three cameras each), the last one now. For seeing how something goes; to wait for
        something to happen, ``wait`` is cheaper. Dream-1 keeps driving if it is running; the watch
        ends early if it stops by itself. Nothing here says whether the step worked: look."""
        result, grid = session.watch(seconds)
        return [result, _png(grid)]

    @server.tool()
    @one_at_a_time
    def stop() -> list:
        """Stop Dream-1 now; the arms hold where they are (still gripping what they hold). Returns
        the run's summary and all three cameras."""
        result, frame = session.stop()
        return [result, _png(frame)]

    @server.tool()
    @one_at_a_time
    def look(camera: Camera = "all") -> Image:
        """The robot's cameras right now (Dream-1 may be moving the arms meanwhile). "all": left
        wrist | top | right wrist, side by side (left and right as the robot sees them); or one of
        them, larger."""
        return _png(session.look(camera))

    @server.tool()
    @one_at_a_time
    def home() -> list:
        """Stop Dream-1 and bring both arms back to their home pose (straight, grippers open) over
        2 s. Objects stay where they are; one held in a gripper is let go there. No Dream-1 call."""
        result, frame = session.home()
        return [result, _png(frame)]

    @server.tool()
    @one_at_a_time
    def review(
        start: Annotated[float, Field(ge=0, description="episode time, seconds (t=...)")],
        end: Annotated[float | None, Field(ge=0, description="episode time; omit for now")] = None,
        camera: Camera = "all",
        frames: Annotated[int, Field(ge=1, le=12)] = 8,
    ) -> Image:
        """Frames from any stretch of this episode (5 a second are kept), by episode time, to see
        what happened when: did it close on the object, did it slip, when did it let go."""
        return _png(session.review(start, end, camera, frames))

    @server.tool()
    @one_at_a_time
    def status() -> dict[str, Any]:
        """The instruction, the time now, what Dream-1 is doing, calls used and left, and this
        episode's runs. Says nothing about success: judge that from the images."""
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
    parser.add_argument("--watchdog", type=float, default=60.0,
                        help="stop Dream-1 when no tool is called for this many seconds (0: never)")
    args = parser.parse_args()
    server, session = build_server(args.task, args.max_calls, args.score_log, args.clips_dir,
                                   args.viewer, args.watchdog)
    try:
        server.run("stdio")
    finally:
        session.finish()                       # video, traces and the Dream session closed
        # Interpreter teardown with the viewer's GLFW and the renderer's EGL contexts both alive
        # dumps core; everything worth keeping is written, so leave without it.
        os._exit(0)


if __name__ == "__main__":
    main()
