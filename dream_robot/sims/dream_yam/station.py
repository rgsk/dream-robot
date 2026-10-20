"""The Dream YAM Sim station with Dream-1 at the controls: one sim, its camera windows, a policy run.

Runs in envs/dream_yam/.venv only. The sim and the request code are Dream's own (the dreamlabs wheel's
``dream_examples/yam_mujoco``): dream-1-beta reads hand poses, which the server computes only from
the older YAM station keys, and their ``dream_control`` speaks them. This module adds what a run
needs around that: a cap on paid calls, a trace per control step, recorded frames, and a referee.

**Physics holds while a call is out.** Simulated time is the same however slow the network is,
so a run is measured in sim seconds; each call plays ``rows`` targets of 50 ms.

**A run is one Dream-1 episode.** Every ``run_policy`` starts a fresh episode on the model
(``reset_episode``): a new prompt with the old one's memory would mix two tasks.
"""

import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

import dream_examples

# The sim's modules import each other by bare name, so its folder goes on the path.
sys.path.insert(0, str(Path(next(iter(dream_examples.__path__))) / "yam_mujoco"))
import dream_control as dc  # noqa: E402
from yam_env import YamEnv, home_path  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
MODEL = dc.BETA_MODEL
STEP_S = 0.05                                    # one control step, one row of a chunk
LOOSE = ("apple", "orange", "banana", "water_bottle")
# Fixed bodies in yam_env.build_xml: centre (x, y) and the radius that counts as "in" / "on".
BOWL, BOWL_R = np.array([0.0, -0.04]), 0.07
PLATE, PLATE_R = np.array([-0.22, -0.045]), 0.065
# The sim's camera names, and the names the driver sees: left / right as the robot sees them.
CAMERAS = {"top": "top", "left_wrist": "left", "right_wrist": "right"}


def api_key():
    """The key: DREAM_API_KEY in the environment, else DREAM_LABS_API_KEY in the repo's .env."""
    if os.environ.get("DREAM_API_KEY"):
        return os.environ["DREAM_API_KEY"]
    env = REPO / ".env"
    for line in env.read_text().splitlines() if env.exists() else []:
        name, _, value = line.partition("=")
        if name.strip() == "DREAM_LABS_API_KEY":
            return value.strip().strip("\"'")
    return None


def tile(images):
    """The three cameras side by side: left wrist, top, right wrist."""
    return np.concatenate([images["left"], images["top"], images["right"]], axis=1)


class Station:
    """One sim and its camera windows; Dream-1 runs on it through ``run_policy``."""

    def __init__(self, viewer=False, pace=True, record=True):
        """``viewer``: a live MuJoCo window with Dream's three camera tiles. ``pace``: the window
        paces the sim to real time (off when a caller keeps the clock itself, as realtime.py does).
        ``record``: keep every step's full camera tile for ``take_frames`` (off for long runs)."""
        self.pace, self.record = pace, record
        if viewer:
            from dream_robot.agent.viewer import find_display

            find_display("sharp")                # MCP servers start without DISPLAY
        self.env = YamEnv()
        self.client = None                      # the Dream client, opened on the first run
        self.calls = 0                          # paid calls since the server started
        self.viewer = self.panel = None
        if viewer:
            self._open_viewer()
        self.reset()

    def _open_viewer(self):
        import mujoco.viewer

        from dream_robot.agent.viewer import can_open_window
        from camera_panel import CameraPanel
        from yam_env import aim_viewer

        if not can_open_window():
            print("viewer: no display found, running headless", file=sys.stderr)
            return
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data,
                                                   show_left_ui=False, show_right_ui=False)
        aim_viewer(self.viewer)
        self.panel = CameraPanel(self.env, self.viewer)
        self._tick = time.monotonic()

    def _show(self):
        """Sync the window and keep to real time: one step per 50 ms of wall clock."""
        if self.viewer is None or not self.viewer.is_running():
            return
        self.viewer.sync()
        self.panel.update(self.images)
        if not self.pace:
            return
        self._tick = max(self._tick + STEP_S, time.monotonic() - 0.2)
        time.sleep(max(0.0, self._tick - time.monotonic()))

    def reset(self):
        self.env.reset()
        self.steps = 0
        self.windows = {name: deque(maxlen=dc.FRAMES) for name in dc.VIDEO_ROLES}
        self.trace = []                         # one row per control step
        self.chunks = []                        # every chunk as played: enough to replay offline
        self.frames = []                        # tiled camera frames since the last take_frames()
        self.images = self.env.images()
        self.prompt, self.run_end = None, None   # the last policy run: its command, the step it ended

    # -- stepping ---------------------------------------------------------------------------

    def play(self, left, right):
        """One 50 ms step toward these targets; renders, records, and keeps a model frame every 0.2 s."""
        state = self.env.step(left, right)
        kept = self.steps % dc.FRAME_EVERY_STEPS == 0
        self.images = dc.capture(self.env, self.windows, keep=kept)
        if self.record:
            self.frames.append(tile(self.images))
        self.trace.append({
            "t": round(self.steps * STEP_S, 2),
            "cmd_grip": {"left": round(float(left[6]), 3), "right": round(float(right[6]), 3)},
            "opening": {s: round(float(state[s][6]), 3) for s in ("left", "right")},
            "z": {name: round(float(self.env.data.body(name).xpos[2]), 4) for name in LOOSE},
            "apple_tips": self.touching("apple"),
        })
        self.steps += 1
        self._show()
        return kept

    def hold(self, steps):
        for _ in range(steps):
            state = self.env.state()
            self.play(state["left"], state["right"])

    def home(self, seconds=2.0):
        """Both arms back to home (joints zero, grippers open), easing in and out; objects stay."""
        for left, right in home_path(self.env.state(), seconds=seconds):
            self.play(left, right)
        self.hold(10)                            # let a released object settle

    def take_frames(self):
        frames, self.frames = self.frames, []
        return frames

    # -- Dream-1 ----------------------------------------------------------------------------

    def _open(self):
        key = api_key()
        if not key:
            raise RuntimeError("no Dream key: set DREAM_API_KEY, or DREAM_LABS_API_KEY in .env")
        self.client = dc.open_policy(dc.policy_config(MODEL, None, poses=True), key)

    def _feed(self, every=1):
        dc.feed(self.client, self.env, self.windows, self.prompt, MODEL, self.steps, every=every, station=True)

    def can_continue(self, prompt):
        """Whether a run on ``prompt`` can pick up the last one: same prompt, nothing moved since."""
        return self.client is not None and prompt == self.prompt and self.steps == self.run_end

    def run_policy(self, prompt, calls, rows=48, log=print, fresh=True):
        """Dream-1 drives both arms for up to ``calls`` paid calls on ``prompt``.

        ``fresh``: a new episode. The arms hold for 1.8 s first, so each camera's window is nine
        frames of the scene as it is now. Not fresh: the last run goes on -- the same episode, the
        model's memory of it and the camera windows kept -- as if it had never paused (the sim
        holds between runs); only allowed when ``can_continue``.
        Each call's chunk is played and every fourth step's frame is handed to the client.
        Returns the calls made, their round trips, and the sim seconds played.
        """
        if not fresh:
            if not self.can_continue(prompt):
                raise RuntimeError("cannot continue: a different command, or the arms moved since")
        else:
            if self.client is None:
                self._open()
            else:
                self.client.reset_episode()
            self.prompt = prompt
            self.hold(dc.FRAMES * dc.FRAME_EVERY_STEPS)
            self._feed(every=0)
        rtts, t_start = [], self.steps
        while len(rtts) < calls:
            t0 = time.monotonic()
            try:
                reply = self.client.get_action()
            except Exception as error:   # warming, rate limit, session restart: wait, ask again
                if getattr(error, "retryable", False) or type(error).__name__ == "DreamSessionExpired":
                    wait = getattr(error, "retry_after_s", None) or 1.0
                    log(f"  waiting {wait:.0f}s ({getattr(error, 'reason', type(error).__name__)})")
                    time.sleep(wait)
                    continue
                raise
            rtts.append(round(time.monotonic() - t0, 2))
            self.calls += 1
            left, right = dc.validate_chunk(self.env, reply, rows, verbose=False)
            self.chunks.append({"step": self.steps, "prompt": prompt,
                                "left": np.asarray(left).tolist(), "right": np.asarray(right).tolist()})
            for l, r in zip(left[:rows], right[:rows]):
                if self.play(l, r):
                    self._feed()
            log(f"  call {len(rtts)}/{calls}  rtt {rtts[-1]:.2f}s  sim {self.steps * STEP_S:.1f}s")
        self.run_end = self.steps
        return {"calls": len(rtts), "rtt_s": rtts, "sim_s": round((self.steps - t_start) * STEP_S, 2)}

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None
        if self.viewer is not None:
            self.viewer.close()
        self.env.close()

    # -- the referee: sim ground truth, never shown to a driver ------------------------------

    def touching(self, body):
        """Which fingertips (``<side>_tip_<left|right>``) touch a body right now."""
        m, d = self.env.model, self.env.data
        target = m.body(body).id
        tips = set()
        for c in d.contact[: d.ncon]:
            a, b = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for mine, other in ((a, b), (b, a)):
                name = m.body(mine).name
                if other == target and "_tip_" in name:
                    tips.add(name)
        return sorted(tips)

    def score(self):
        """Where each loose object is, and whether it is in the bowl or on the plate."""
        out = {}
        for name in LOOSE:
            pos = self.env.data.body(name).xpos.copy()
            out[name] = {
                "xyz": np.round(pos, 3).tolist(),
                "in_bowl": bool(np.linalg.norm(pos[:2] - BOWL) < BOWL_R),
                "on_plate": bool(np.linalg.norm(pos[:2] - PLATE) < PLATE_R),
            }
        return out
