"""Dream YAM Sim as an MCP server: the LLM plans in words, Dream-1 moves the arms.

The driver never moves a joint. Its one motion tool is ``policy(command, seconds)``: Dream-1 drives
both arms on that command for that long, and the tool returns frames from across the run. The
driver splits a task into commands Dream-1 can do ("put apple into the bowl"), looks, and decides
what to command next. That is ROADMAP's planner/executor split with a learned executor.

**No tool reports success,** as in dream_robot/agent/server.py: the referee's verdict (sim
positions) goes to ``--score-log`` only, after every motion and at reset.

**Paid calls are capped** per server (``--max-calls``): Dream-1 bills each call, so a driver that
loops cannot run up a bill. ``status`` says how many are left.

    MUJOCO_GL=egl envs/dream_yam/.venv/bin/python -m dream_robot.sims.dream_yam.server \
        --task "Put the apple and the orange in the bowl." --score-log scratch/dream1/referee.jsonl
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, Literal

import imageio.v2 as imageio
import numpy as np
from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from PIL import Image as PILImage
from pydantic import Field

from dream_robot.agent.server import _on_sim_thread, _png, _quiet

INSTRUCTIONS = (Path(__file__).with_name("driver_prompt.md")).read_text()
ROW_S = 0.05                 # one row of a chunk
ROWS = 48                    # rows per call: 2.4 s of motion
Camera = Literal["all", "top", "left_wrist", "right_wrist"]


def _grid(frames, n, width=960):
    """``n`` frames evenly across a run, stacked top to bottom, each stamped with its time."""
    from PIL import ImageDraw

    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    rows = []
    for i in idx:
        im = PILImage.fromarray(frames[i])
        im = im.resize((width, round(im.height * width / im.width)))
        ImageDraw.Draw(im).text((6, 4), f"{i * ROW_S:.1f}s", fill=(255, 255, 0))
        rows.append(np.asarray(im))
    return np.concatenate(rows, axis=0)


def _camera(images, camera):
    from dream_robot.sims.dream_yam.station import CAMERAS, tile

    return tile(images) if camera == "all" else images[CAMERAS[camera]]


class Session:
    """The state behind the tools. Built on the sim thread (the renderer's GL context lives there)."""

    def __init__(self, task, max_calls, score_log, clips_dir, viewer=False):
        from dream_robot.sims.dream_yam.station import Station

        with _quiet():
            self.station = Station(viewer=viewer)
        self.task, self.max_calls = task, max_calls
        self.score_log, self.clips_dir = score_log, clips_dir
        self.episode = None
        self.clips = []                          # (name, frames) this episode

    def reset(self, seed):
        self.station.reset()
        self.episode = time.strftime("%y%m%d-%H%M%S") + (f"-s{seed}" if seed is not None else "")
        self.clips = []
        self.log("reset")
        return self.status()

    def _check(self):
        if self.episode is None:
            raise ToolError("call reset first")

    def policy(self, command, seconds, continue_run):
        self._check()
        if continue_run and not self.station.can_continue(command):
            raise ToolError("cannot continue: continue_run needs the same command as the last policy "
                            "run, with no home or reset since. Start it fresh (continue_run false).")
        calls = math.ceil(seconds / (ROWS * ROW_S))
        left = self.max_calls - self.station.calls
        if left <= 0:
            raise ToolError(f"no Dream-1 calls left on this server ({self.max_calls} used)")
        calls = min(calls, left)
        self.station.take_frames()               # drop anything recorded since the last tool
        with _quiet():
            run = self.station.run_policy(command, calls, log=lambda s: None, fresh=not continue_run)
        frames = self.station.take_frames()
        self._keep(f"policy{' (continued)' if continue_run else ''}: {command}", frames)
        self.log("policy", command=command, continued=continue_run, **run)
        return ({"command": command, "continued": continue_run, "sim_seconds": run["sim_s"],
                 "calls": run["calls"],
                 "calls_left": self.max_calls - self.station.calls, "clip": len(self.clips)},
                _grid(frames, 6))

    def home(self):
        self._check()
        self.station.take_frames()
        self.station.home()
        frames = self.station.take_frames()
        self._keep("home", frames)
        self.log("home")
        return {"clip": len(self.clips)}, frames[-1]

    def review_clip(self, clip, camera, start, end, frames):
        if not self.clips:
            raise ToolError("no clips yet this episode")
        n = len(self.clips) if clip is None else clip
        if not 1 <= n <= len(self.clips):
            raise ToolError(f"clip {n}: this episode has {len(self.clips)} (1-{len(self.clips)})")
        name, all_frames = self.clips[n - 1]
        a = int((start or 0) / ROW_S)
        b = len(all_frames) if end is None else int(end / ROW_S)
        window = all_frames[a:max(a + 1, b)]
        if camera != "all":                      # the tile is left wrist | top | right wrist
            w = window[0].shape[1] // 3
            col = {"left_wrist": 0, "top": 1, "right_wrist": 2}[camera]
            window = [f[:, col * w:(col + 1) * w] for f in window]
        grid = _grid(window, min(frames, len(window)), width=960 if camera == "all" else 640)
        return grid

    def _keep(self, name, frames):
        self.clips.append((name, frames))
        if self.clips_dir and frames:
            path = Path(self.clips_dir) / f"{self.episode}-{len(self.clips):02d}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(path, fps=20, macro_block_size=8) as video:
                for f in frames:
                    video.append_data(f)

    def status(self):
        return {
            "instruction": self.task,
            "sim_seconds": round(self.station.steps * ROW_S, 2),
            "dream_calls_used": self.station.calls,
            "dream_calls_left": self.max_calls - self.station.calls,
            "clips": [{"n": i, "what": name, "seconds": round(len(f) * ROW_S, 1)}
                      for i, (name, f) in enumerate(self.clips, start=1)],
        }

    def log(self, event, **extra):
        """The referee's verdict after each motion: for the score log only, never a tool."""
        if not self.score_log:
            return
        extra = {("run_sim_s" if k == "sim_s" else k): v for k, v in extra.items()}
        row = {"episode": self.episode, "event": event, "sim_s": round(self.station.steps * ROW_S, 2),
               "dream_calls": self.station.calls, **extra, "score": self.station.score()}
        Path(self.score_log).parent.mkdir(parents=True, exist_ok=True)
        with open(self.score_log, "a") as f:
            f.write(json.dumps(row) + "\n")

    def save_traces(self):
        """trace.json and chunks.json of the whole server run, next to the clips."""
        if self.clips_dir and self.episode:
            base = Path(self.clips_dir) / self.episode
            Path(f"{base}-trace.json").write_text(json.dumps(self.station.trace))
            Path(f"{base}-chunks.json").write_text(json.dumps(self.station.chunks))


def build_server(task, max_calls=200, score_log=None, clips_dir=None, viewer=False):
    """The server and its session (the session is for tests and the referee, not a tool)."""
    sim = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sim")
    session = sim.submit(lambda: Session(task, max_calls, score_log, clips_dir, viewer)).result()
    server = MCPServer("dream-yam", instructions=INSTRUCTIONS)
    on_sim = _on_sim_thread(sim)

    @server.tool()
    @on_sim
    def reset(seed: int | None = None) -> list:
        """Start a new episode: objects back in place, both arms home. Returns the instruction,
        status, and all three cameras. (The scene is the same every time; ``seed`` only labels it.)"""
        return [session.reset(seed), _png(_camera(session.station.images, "all"))]

    @server.tool()
    @on_sim
    def look(camera: Camera = "all") -> Image:
        """The robot's cameras now. "all": left wrist | top | right wrist, side by side (left and
        right as the robot sees them); or one of them, larger."""
        return _png(_camera(session.station.images, camera))

    @server.tool()
    @on_sim
    def policy(
        command: Annotated[str, Field(description="what the arms should do, in plain words, "
                                      "one short step, e.g. 'put apple into the bowl'")],
        seconds: Annotated[float, Field(ge=2.4, le=90, description="how long Dream-1 drives, "
                                        "simulated seconds; one paid call per 2.4 s")] = 40,
        continue_run: Annotated[bool, Field(description="true: pick up the last run where it paused, "
                                            "same command, Dream-1 keeps its memory of it; false: "
                                            "a fresh start")] = False,
    ) -> list:
        """Dream-1, a learned robot policy, drives both arms on ``command`` for ``seconds``, then the
        arms stop where they are (time stands still until your next motion). It sees the three
        cameras and the joints, nothing else.

        A fresh run (the default) starts Dream-1 anew: it does not remember earlier commands.
        ``continue_run`` goes on with the last run as if it had never paused -- same command, no
        ``home`` or ``reset`` in between -- so a long step can be watched in pieces: run 10-15 s,
        look, continue or change course.

        Returns the calls used and left, and 6 frames from across the run (time stamped, all three
        cameras each), the last one the end. Nothing here says whether it worked: look."""
        result, grid = session.policy(command, seconds, continue_run)
        return [result, _png(grid)]

    @server.tool()
    @on_sim
    def home() -> list:
        """Both arms back to their home pose (straight, grippers open) over 2 s. Objects stay where
        they are; one held in a gripper is let go there. Free: no Dream-1 call."""
        result, frame = session.home()
        return [result, _png(frame)]

    @server.tool()
    @on_sim
    def review_clip(
        clip: Annotated[int | None, Field(description="its number from status; omit for the latest")] = None,
        camera: Camera = "all",
        start: Annotated[float | None, Field(ge=0)] = None,
        end: Annotated[float | None, Field(ge=0)] = None,
        frames: Annotated[int, Field(ge=1, le=12)] = 8,
    ) -> Image:
        """More frames from an earlier policy run or home move this episode, to see what happened
        when (did it close on the object, did it slip). ``start``/``end`` in seconds within the clip."""
        return _png(session.review_clip(clip, camera, start, end, frames))

    @server.tool()
    @on_sim
    def status() -> dict[str, Any]:
        """The instruction, sim time, Dream-1 calls used and left, and this episode's clips. Says
        nothing about success: judge that from the images."""
        return session.status()

    return server, session, sim


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, help="the instruction reset hands the driver")
    parser.add_argument("--max-calls", type=int, default=200, help="paid Dream-1 calls, at most")
    parser.add_argument("--score-log", help="append the referee's verdict after every motion (JSONL)")
    parser.add_argument("--clips-dir", help="write each motion's three-camera clip as MP4 here")
    parser.add_argument("--viewer", action="store_true",
                        help="live MuJoCo window with the three cameras, real-time pacing (to watch)")
    args = parser.parse_args()
    server, session, sim = build_server(args.task, args.max_calls, args.score_log, args.clips_dir,
                                        args.viewer)
    try:
        server.run("stdio")
    finally:
        sim.submit(session.save_traces).result()
        sim.submit(session.station.close).result()
    os.dup2(2, 1)


if __name__ == "__main__":
    main()
