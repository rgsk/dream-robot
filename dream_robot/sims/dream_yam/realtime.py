"""Dream YAM Sim on the wall clock: the world does not stop while Dream-1 or the driver thinks.

station.py holds physics during every Dream-1 call and between tool calls. A real robot cannot:
while a call is out the arms run what they have or hold, and while the planner thinks for 20 s
the arms hold their last target for 20 s -- a held fruit stays squeezed (or slips), a dropped one
rolls. This module runs the station that way:

- **One loop thread owns the sim** (and its GL context): 20 steps a second of wall clock, always.
  Each step it plays the current target: a row of Dream-1's chunk while a run is on, a home path,
  or else the last target, held -- what a position-servoed arm does with no new command.
- **Dream-1 is called the way the docs' own control loop does it** ("Running it in a loop"):
  get_action on a thread of its own, the next chunk asked for early (``client.refill_due``), the
  row due now picked by the time since the observation the call carried, rows already past
  skipped. Below ~0.7 s round trips the arms never stop; above, the client paces sync and the
  arms hold through each call, as on the real station.
- **Between runs the client keeps getting every 0.2 s frame** while the episode is open, as a
  real loop would; a continued run after a long pause is what a real one would be (the session
  may have expired; its window shows the pause).
- Tools never touch the sim directly: they post work to the loop and wait on it.

Frames are kept as 5 Hz JPEGs (the store) for tool grids and review; the full 20 Hz view goes
to an episode MP4 when ``clips_dir`` is set.
"""

import io
import json
import math
import queue
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from dream_robot.sims.dream_yam.grip import GripWatch

STEP_S = 0.05
STORE_W = 960            # width of a stored tile (all three cameras side by side)


@dataclass
class Run:
    """One policy run: a command, until a step (or until stopped); what came back while it ran."""

    prompt: str
    start_step: int
    end_step: int | None             # None: runs until stopped
    continued: bool
    plan: tuple | None = None        # (t_obs step, left rows, right rows, step it took over at)
    calls: int = 0
    rtt_s: list = field(default_factory=list)
    modes: list = field(default_factory=list)
    retry_at: int = 0
    waits: list = field(default_factory=list)
    restarts: int = 0
    held_steps: int = 0              # steps with no row due: the arms held
    error: str | None = None
    stopped_by: str | None = None    # "stop", "home", "reset", "switch", "watchdog", "time", "calls"
    stop_step: int | None = None
    done: threading.Event = field(default_factory=threading.Event)


class Planner:
    """get_action on a thread of its own, one call at a time (the docs' Planner, tagged by run)."""

    def __init__(self, rt):
        self.rt = rt
        self.thread = None
        self.result = None
        self.lock = threading.Lock()

    def busy(self):
        return self.thread is not None and self.thread.is_alive()

    def request(self, run, t_obs):
        def call():
            t0 = time.monotonic()
            try:
                reply = self.rt.station.client.get_action()
                rtt = time.monotonic() - t0
                with self.rt.lock:
                    self.rt.calls += 1
                result = ("ok", run, t_obs, reply, rtt)
            except Exception as error:
                result = ("error", run, t_obs, error, time.monotonic() - t0)
            with self.lock:
                self.result = result

        self.thread = threading.Thread(target=call, daemon=True, name="dream-call")
        self.thread.start()

    def take(self):
        with self.lock:
            result, self.result = self.result, None
        return result

    def wait(self):
        if self.thread is not None:
            self.thread.join()


class Realtime:
    """The station on the wall clock. Construct from any thread; the sim lives on its own."""

    def __init__(self, viewer=False, max_calls=200, clips_dir=None, watchdog_s=60.0):
        """``watchdog_s``: a run nobody has touched (``touch``) for this long stops itself, so a
        driver that stalls cannot keep Dream-1 -- and the bill -- going."""
        self.lock = threading.RLock()
        self.watchdog_s, self.last_touch = watchdog_s, time.monotonic()
        self.max_calls, self.calls = max_calls, 0
        self.clips_dir = clips_dir
        self.jobs = queue.Queue()            # (fn, Future): run on the loop thread at a step's start
        self.stop_flag = False
        self.ready = Future()
        self.lag_s = 0.0                     # how far the loop has fallen behind the wall, worst
        self.errors = []                     # errors the loop survived
        self.thread = threading.Thread(target=self._main, args=(viewer,), daemon=True, name="sim-loop")
        self.thread.start()
        self.ready.result()

    # -- the loop ---------------------------------------------------------------------------------

    def _main(self, viewer):
        try:
            from dream_robot.sims.dream_yam.station import Station
            from yam_env import HOME

            self.HOME = HOME
            self.station = Station(viewer=viewer, pace=False, record=False)
            self.planner = Planner(self)
            self._new_episode()
        except Exception as exc:
            self.ready.set_exception(exc)
            return
        self.ready.set_result(True)
        due = time.monotonic()
        while not self.stop_flag:
            while not self.jobs.empty():
                fn, fut = self.jobs.get()
                try:
                    fut.set_result(fn())
                except Exception as exc:
                    fut.set_exception(exc)
            if self.stop_flag:
                break
            with self.lock:
                try:
                    self._step()
                except Exception as exc:
                    # Never let one bad step kill the world: end the run (arms hold), keep going.
                    self._error(exc)
            due += STEP_S
            late = time.monotonic() - due
            if late > 0.5:                  # fell far behind (a slow render): drop the debt
                self.lag_s = max(self.lag_s, late)
                due = time.monotonic()
            time.sleep(max(0.0, due - time.monotonic()))

    def on_loop(self, fn, timeout=60):
        """Run ``fn`` on the loop thread between two steps; wait for and return its result."""
        fut = Future()
        self.jobs.put((fn, fut))
        return fut.result(timeout=timeout)

    def _new_episode(self):
        s = self.station
        s.reset()
        self.target = (self.HOME.copy(), self.HOME.copy())
        self.path = deque()
        self.run = None
        self.episode_open = False            # the Dream-1 episode can be continued
        self.fed_step = 0
        self.store = []                      # (step, jpeg bytes), 5 Hz
        self.grip = GripWatch()              # grasp / release, from the gripper's own sensor
        self.events = []                     # (step, arm, "grasp" | "release", opening)
        self.video = None
        if self.clips_dir:
            import imageio.v2 as imageio

            self.episode = time.strftime("%y%m%d-%H%M%S")
            Path(self.clips_dir).mkdir(parents=True, exist_ok=True)
            self.video = imageio.get_writer(Path(self.clips_dir) / f"{self.episode}-rt.mp4",
                                            fps=20, macro_block_size=8)

    def _step(self):
        s = self.station
        if self.path:
            self.target = self.path.popleft()
        elif self.run is not None:
            self._run_step(self.run)
        left, right = self.target
        kept = s.play(left, right)
        row = s.trace[-1]
        for arm, kind, opening in self.grip.update(row["cmd_grip"], row["opening"]):
            self.events.append((s.steps, arm, kind, opening))
        if kept:
            self.store.append((s.steps, _jpeg(s.images)))
            if self.episode_open and s.client is not None:
                s._feed()
                self.fed_step = s.steps
        if self.video is not None:
            self.video.append_data(_small(s.images, 1280))

    def _run_step(self, run):
        s = self.station
        result = self.planner.take()
        if result is not None and result[1] is run:
            kind, _, t_obs, payload, rtt = result
            if kind == "ok":
                from dream_control import validate_chunk

                try:
                    left, right = validate_chunk(s.env, payload, 48, verbose=False)
                except ValueError as exc:            # a malformed chunk: skip it, keep the last plan
                    run.rejected = getattr(run, "rejected", 0) + 1
                    self._save_rejected(payload, exc)
                    return
                run.plan = (t_obs, left, right, s.steps - t_obs)
                run.calls += 1
                run.rtt_s.append(round(rtt, 2))
                run.modes.append(getattr(s.client, "mode", "?"))
                s.chunks.append({"step": s.steps, "t_obs": t_obs, "prompt": run.prompt,
                                 "left": np.asarray(left).tolist(), "right": np.asarray(right).tolist()})
            elif type(payload).__name__ == "DreamSessionExpired":
                run.plan, run.restarts = None, run.restarts + 1
            elif getattr(payload, "retryable", False):
                wait = getattr(payload, "retry_after_s", None) or 1.0
                run.retry_at = s.steps + math.ceil(wait / STEP_S)
                run.waits.append(getattr(payload, "reason", type(payload).__name__))
            else:
                run.error = f"{type(payload).__name__}: {payload}"
        executed = remaining = 0
        if run.plan is not None:
            t_obs, left, right, start = run.plan
            k = s.steps - t_obs                 # the row due now; rows already past are skipped
            if k < len(left):
                self.target = (left[k], right[k])
                executed, remaining = k - start, len(left) - k
            else:
                run.held_steps += 1
        else:
            run.held_steps += 1
        if run.end_step is not None and s.steps >= run.end_step:
            run.stopped_by = "time"
        elif self.watchdog_s and time.monotonic() - self.last_touch > self.watchdog_s:
            run.stopped_by = "watchdog"
        elif self.calls >= self.max_calls and remaining == 0 and not self.planner.busy():
            run.stopped_by = "calls"
        if run.stopped_by or run.error:
            self._end(run, run.stopped_by or "error")
            return
        if (not self.planner.busy() and s.steps >= run.retry_at and self.calls < self.max_calls
                and s.client.refill_due(executed, remaining)):
            self.planner.request(run, self.fed_step)

    def _error(self, exc):
        import traceback

        text = "".join(traceback.format_exception(exc))
        print(f"realtime loop error at step {self.station.steps}:\n{text}", file=sys.stderr)
        self.errors.append({"step": self.station.steps, "error": f"{type(exc).__name__}: {exc}"})
        if self.clips_dir:
            with open(Path(self.clips_dir) / "loop-errors.log", "a") as f:
                f.write(f"{getattr(self, 'episode', '?')} step {self.station.steps}\n{text}\n")
        if self.run is not None:
            self.run.error = f"{type(exc).__name__}: {exc}"
            self._end(self.run, "error")

    @property
    def alive(self):
        return self.thread.is_alive()

    def _end(self, run, why):
        """On the loop thread: the run is over; the arms hold the last target."""
        if run is None or run.done.is_set():
            return
        run.stopped_by = run.stopped_by or why
        run.stop_step = self.station.steps
        if self.run is run:
            self.run = None
        run.done.set()

    # -- what the tools ask for (called from tool threads) ----------------------------------------

    def _save_rejected(self, payload, exc):
        print(f"realtime: chunk rejected at step {self.station.steps}: {exc}", file=sys.stderr)
        self.errors.append({"step": self.station.steps, "error": f"chunk rejected: {exc}"})
        if self.clips_dir:
            path = Path(self.clips_dir) / f"{getattr(self, 'episode', 'x')}-rejected-{self.station.steps}.json"
            path.write_text(json.dumps({"error": str(exc), "reply": payload},
                                       default=lambda v: np.asarray(v).tolist()))

    def touch(self):
        """A tool was called: the driver is alive (the watchdog's clock restarts)."""
        self.last_touch = time.monotonic()

    def reset(self):
        self.on_loop(lambda: self._end(self.run, "reset"))
        self.planner.wait()
        self.save_traces()
        def job():
            if self.video is not None:
                self.video.close()
            self._new_episode()
        self.on_loop(job)

    def policy(self, prompt, continued=False, seconds=None):
        """Start Dream-1 on ``prompt``; returns ``(run, already)`` at once, ``already`` when that
        command was running anyway. It drives until ``stop``, ``home``, ``reset``, another command,
        the watchdog, or the call cap. ``seconds``: instead, run that long and return when done.

        Not ``continued``: a fresh Dream-1 episode (a running other command is stopped first).
        ``continued``: resume the stopped run's episode -- same command, no home/reset since."""
        s = self.station
        current = self.run
        if current is not None and current.prompt == prompt:
            return current, True
        if continued and (current is not None or not (self.episode_open and s.prompt == prompt)):
            raise ValueError("cannot continue: a different command, or home/reset since")
        if self.calls >= self.max_calls:
            raise ValueError(f"no Dream-1 calls left ({self.max_calls} used)")
        if current is not None:
            self.on_loop(lambda: self._end(current, "switch"))
        self.planner.wait()                       # reset_episode refuses to run beside a call
        if s.client is None:
            s._open()                             # network, ~14 s the first time: not on the loop
        from dream_control import FRAMES

        while any(len(w) < FRAMES for w in s.windows.values()):
            time.sleep(STEP_S)                    # just after a reset: the cameras' 1.8 s window fills

        def start():
            if not continued:
                s.client.reset_episode()
                s.prompt = prompt
                s._feed(every=0)                  # the last 9 frames of the world as it is now
                self.fed_step = s.steps
                self.episode_open = True
            end = None if seconds is None else s.steps + round(seconds / STEP_S)
            self.run = Run(prompt, s.steps, end, continued)
            return self.run

        self.touch()
        run = self.on_loop(start)
        if seconds is not None:
            run.done.wait(timeout=seconds + 120)
        return run, False

    def stop(self):
        """Stop Dream-1; the arms hold where they are. Returns the run stopped, or None."""
        run = self.run
        self.on_loop(lambda: self._end(run, "stop"))
        return run

    def home(self, seconds=2.0):
        from yam_env import home_path

        def start():
            self._end(self.run, "home")
            self.episode_open = False
            path = home_path(self.station.env.state(), seconds=seconds)
            self.path.extend(path)
            return self.station.steps, self.station.steps + len(path)

        start_step, end_step = self.on_loop(start)
        while self.station.steps < end_step + 10:    # and let a released object settle
            time.sleep(STEP_S)
        return start_step, self.station.steps

    def images(self):
        with self.lock:
            return {k: v.copy() for k, v in self.station.images.items()}

    def frames(self, start_step, end_step):
        """Stored 5 Hz tiles in [start, end], decoded: (step, rgb) pairs."""
        with self.lock:
            picked = [(st, j) for st, j in self.store if start_step <= st <= end_step]
        return [(st, np.asarray(Image.open(io.BytesIO(j)))) for st, j in picked]

    def score(self):
        with self.lock:
            return self.station.score()

    @property
    def steps(self):
        return self.station.steps

    def save_traces(self):
        """trace / chunks / grip events of this episode, rewritten whole: called after every tool,
        so a server killed mid-run (as claude -p's end does) still leaves them."""
        if self.clips_dir and getattr(self, "episode", None):
            with self.lock:
                trace, chunks, events = list(self.station.trace), list(self.station.chunks), list(self.events)
            base = Path(self.clips_dir) / f"{self.episode}-rt"
            Path(f"{base}-trace.json").write_text(json.dumps(trace))
            Path(f"{base}-chunks.json").write_text(json.dumps(chunks))
            Path(f"{base}-events.json").write_text(json.dumps(events))

    def close(self):
        self.planner.wait()
        self.save_traces()
        def job():
            if self.video is not None:
                self.video.close()
            self.station.close()
            self.stop_flag = True
        try:
            self.on_loop(job, timeout=30)
        except Exception as exc:
            print(f"realtime close: {exc}", file=sys.stderr)


def _small(images, width):
    from dream_robot.sims.dream_yam.station import tile

    im = Image.fromarray(tile(images))
    return np.asarray(im.resize((width, round(im.height * width / im.width))))


def _jpeg(images):
    buf = io.BytesIO()
    Image.fromarray(_small(images, STORE_W)).save(buf, format="JPEG", quality=85)
    return buf.getvalue()
