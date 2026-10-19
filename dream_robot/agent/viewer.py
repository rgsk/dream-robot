"""A live MuJoCo window onto the server's own sim, so a human can watch whoever drives it.

The server is driven by tool calls; without this, the only way to see the robot
is to call ``look``. With it, every step shows up in a native window as it
happens -- the window belongs to the server process, so it shows *that* sim,
the one the caller is acting on, not a copy.

**Real time.** The sim steps ~10x faster than real time, so a pick would flash
by in a fraction of a second. While the window is open, stepping is paced to
the env's control rate.

**Reopened on reset.** robosuite rebuilds MjModel/MjData on every reset, and a
viewer is bound to one model; the old window closes and a new one opens.

**Three modes** (``--viewer sharp|docked|own``), because Wayland trades placement
against sharpness and MuJoCo's viewer cannot be sized:

    sharp    MuJoCo's viewer, native Wayland: full resolution, but it opens bigger
             than a 2x-scaled screen, so GNOME maximises it
    docked   MuJoCo's viewer on XWayland: snapped to the right third on every
             open (x11_dock.py), drawn at half resolution and upscaled -- soft
    own      our window (OwnViewer): native Wayland, sized to a third of the
             screen, full resolution; GNOME places it, you drag it once. Kept
             open across resets. Dimmed to 80%; [ and ] adjust
"""

from __future__ import annotations

import glob
import os
import sys
import time

Mode = str  # "sharp" | "docked" | "own"
MODES = ("sharp", "docked", "own")


def make_viewer(env, mode: Mode):
    return OwnViewer(env) if mode == "own" else LiveViewer(env, mode)


def find_display(mode: Mode) -> None:
    """Fill in display variables from the session, if the environment has none.

    ``docked`` forces X11 (XWayland) for this process: only there can a window be
    placed. The variables are this process's own; the desktop is untouched.

    **Call before anything imports glfw** -- robosuite does, while building the
    sim. The glfw package ships an X11 and a Wayland library and picks one at
    import, from XDG_SESSION_TYPE; set it later and the X11 one is already loaded.

    **Found, not just inherited.** MCP clients launch servers with a stripped
    environment: the Inspector's web mode drops DISPLAY and ignores ``-e`` with
    ``--config``. ``--viewer`` is an explicit request for a window on this
    machine, so when nothing is set, look where the session's display lives:
    the Wayland socket under /run/user/<uid> (no auth cookie needed), then X11.
    GLFW only tries Wayland when XDG_SESSION_TYPE says so, and X11 under
    XWayland wants the session's auth cookie, so both are set too.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    if mode == "docked":
        os.environ.pop("WAYLAND_DISPLAY", None)
        os.environ["XDG_SESSION_TYPE"] = "x11"
        if not os.environ.get("DISPLAY"):
            x11 = sorted(glob.glob("/tmp/.X11-unix/X*"))
            if not x11:
                return
            os.environ["DISPLAY"] = ":" + os.path.basename(x11[0])[1:]
        if not os.environ.get("XAUTHORITY"):
            cookies = glob.glob(f"{runtime}/.mutter-Xwaylandauth.*")
            cookies += glob.glob(os.path.expanduser("~/.Xauthority"))
            if cookies:
                os.environ["XAUTHORITY"] = cookies[0]
        return
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        wayland = [p for p in sorted(glob.glob(f"{runtime}/wayland-*")) if not p.endswith(".lock")]
        if wayland:
            os.environ["XDG_RUNTIME_DIR"] = runtime
            os.environ["WAYLAND_DISPLAY"] = os.path.basename(wayland[0])
            os.environ["XDG_SESSION_TYPE"] = "wayland"
        elif x11 := sorted(glob.glob("/tmp/.X11-unix/X*")):
            os.environ["DISPLAY"] = ":" + os.path.basename(x11[0])[1:]
            cookies = glob.glob(f"{runtime}/.mutter-Xwaylandauth.*")
            cookies += glob.glob(os.path.expanduser("~/.Xauthority"))
            if cookies and not os.environ.get("XAUTHORITY"):
                os.environ["XAUTHORITY"] = cookies[0]
        else:
            return
        found = os.environ.get("WAYLAND_DISPLAY") or os.environ["DISPLAY"]
        print(f"viewer: no display in the environment, found {found}", file=sys.stderr)


def can_open_window() -> bool:
    """Whether GLFW starts. Without a display it cannot, and its failure would kill
    the process from C -- no exception to catch -- inside the viewer. A display
    that is set but refuses us must mean "headless", not a dead server."""
    import glfw

    return bool(glfw.init())


class LiveViewer:
    def __init__(self, env, mode: Mode = "sharp"):
        self._env = env
        self._mode = mode
        self._handle = None
        self._period = 1.0 / float(env.control_hz)
        self._last = 0.0

    def open(self) -> None:
        import mujoco.viewer

        self.close()
        model, data = self._env.mujoco
        self._handle = mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        )
        _look_at_table(self._handle, self._env)
        if self._mode == "docked":
            from dream_robot.agent.x11_dock import dock_right_third_async

            dock_right_third_async()

    def sync(self, *, pace: bool = True) -> None:
        """Push the current state to the window; sleep to hold real time if ``pace``."""
        if self._handle is None or not self._handle.is_running():
            return
        if pace:
            wait = self._period - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
        self._handle.sync()
        self._last = time.monotonic()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _look_at_table(handle, env) -> None:
    """Start from a view of the task, not MuJoCo's default."""
    with handle.lock():
        # robosuite models every part twice: a visual mesh (group 1) and a coloured
        # collision shape (group 0). Its own cameras hide group 0; MuJoCo's viewer
        # shows every group, which draws a green-and-blue arm fighting the real one.
        handle.opt.geomgroup[0] = 0
        _set_view(handle.cam, env)


def _set_view(cam, env) -> None:
    """Aim at the task. An env that knows better says so with ``view()`` -> (lookat,
    distance, azimuth, elevation): a kitchen's origin can be inside a wall, so the
    kitchen aims at its robot. Otherwise: the robosuite table, which sits at the origin."""
    lookat, distance, azimuth, elevation = (
        env.view() if hasattr(env, "view") else ((0.0, 0.0, 0.85), 2.3, 180.0, -25.0)
    )
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation


class OwnViewer:
    """Our own window: GLFW plus MuJoCo's renderer, a third of the screen, sharp.

    Two threads, one lock. The sim thread builds the scene (``mjv_updateScene``,
    CPU only) in ``sync``; the window thread draws it (``mjr_render``) in its own
    GL context and handles the mouse. The sim's offscreen renderer keeps its
    EGL context on the sim thread, as before -- the two never share one.

    The window outlives resets: ``open`` with a new model swaps the scene and
    asks the window thread to rebuild its GPU-side context, instead of closing.
    """

    TITLE = "dream-robot"
    BRIGHTNESS = 0.8        # light scale for this window only; [ and ] adjust it live

    def __init__(self, env):
        import threading

        self._env = env
        self._period = 1.0 / float(env.control_hz)
        self._last = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._model = self._data = self._scene = None
        self._model_changed = False
        self._cam = self._opt = None
        self._snap: list | None = None          # a request, filled by the window thread
        self.brightness = self.BRIGHTNESS

    def open(self) -> None:
        import threading

        import mujoco

        model, data = self._env.mujoco
        with self._lock:
            self._model, self._data = model, data
            self._scene = mujoco.MjvScene(model, maxgeom=10_000)
            if self._cam is None:
                self._cam, self._opt = mujoco.MjvCamera(), mujoco.MjvOption()
                self._opt.geomgroup[0] = 0        # hide collision shapes, as robosuite does
                _set_view(self._cam, self._env)
            elif hasattr(self._env, "view"):       # a new kitchen: the robot moved, re-aim
                _set_view(self._cam, self._env)
            # else: keep the user's view across resets -- the table never moves
            self._model_changed = True
        self._update_scene()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._window_loop, daemon=True)
            self._thread.start()

    def sync(self, *, pace: bool = True) -> None:
        if self._thread is None or not self._thread.is_alive():
            return
        if pace:
            wait = self._period - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
        self._update_scene()
        self._last = time.monotonic()

    def snapshot(self, timeout: float = 2.0):
        """What the window last drew, read back from its framebuffer. For diagnosing."""
        self._snap = []
        deadline = time.monotonic() + timeout
        while not self._snap and time.monotonic() < deadline:
            time.sleep(0.02)
        frame, self._snap = (self._snap or [None])[0], None
        return frame

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def _update_scene(self) -> None:
        import mujoco

        with self._lock:
            mujoco.mjv_updateScene(self._model, self._data, self._opt, None, self._cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self._scene)
            # Dim this window's copy of the lights, never the model's: look and the
            # camera find_by_colour reads must see the scene exactly as recorded.
            for i in range(self._scene.nlight):
                light = self._scene.lights[i]
                light.ambient[:] = light.ambient * self.brightness
                light.diffuse[:] = light.diffuse * self.brightness
                light.specular[:] = light.specular * self.brightness

    def _window_loop(self) -> None:
        import glfw
        import mujoco

        if not glfw.init():
            return
        monitor = glfw.get_primary_monitor()
        _, _, w, h = glfw.get_monitor_workarea(monitor)
        sx, sy = glfw.get_monitor_content_scale(monitor)
        # Workarea is in pixels; window sizes are in screen units (pixels / scale).
        window = glfw.create_window(int(w / sx) // 3, int(h / sy), self.TITLE, None, None)
        if not window:
            return
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        context = None
        mouse = {"button": None, "x": 0.0, "y": 0.0}

        def on_button(win, button, action, mods):
            mouse["button"] = button if action == glfw.PRESS else None
            mouse["x"], mouse["y"] = glfw.get_cursor_pos(win)

        def on_move(win, x, y):
            if mouse["button"] is None:
                return
            dx, dy = x - mouse["x"], y - mouse["y"]
            mouse["x"], mouse["y"] = x, y
            action = (mujoco.mjtMouse.mjMOUSE_MOVE_V if mouse["button"] == glfw.MOUSE_BUTTON_RIGHT
                      else mujoco.mjtMouse.mjMOUSE_ROTATE_V)
            _, height = glfw.get_window_size(win)
            with self._lock:
                mujoco.mjv_moveCamera(self._model, action, dx / height, dy / height,
                                      self._scene, self._cam)
                # The scene's camera is only rebuilt on a sim step; while the robot is
                # idle, a mouse move would never reach the screen. Refresh it now.
                mujoco.mjv_updateCamera(self._model, self._data, self._cam, self._scene)

        def on_scroll(win, dx, dy):
            with self._lock:
                # Reversed from MuJoCo's own viewer: scroll the way photo viewers zoom.
                mujoco.mjv_moveCamera(self._model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0,
                                      0.05 * dy, self._scene, self._cam)
                mujoco.mjv_updateCamera(self._model, self._data, self._cam, self._scene)

        def on_key(win, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            step = {glfw.KEY_LEFT_BRACKET: -0.1, glfw.KEY_RIGHT_BRACKET: 0.1}.get(key)
            if step is not None:
                self.brightness = min(1.5, max(0.1, self.brightness + step))
                self._update_scene()

        glfw.set_key_callback(window, on_key)
        glfw.set_mouse_button_callback(window, on_button)
        glfw.set_cursor_pos_callback(window, on_move)
        glfw.set_scroll_callback(window, on_scroll)
        try:
            while not self._stop.is_set() and not glfw.window_should_close(window):
                glfw.poll_events()
                with self._lock:
                    if self._model_changed or context is None:
                        context = mujoco.MjrContext(self._model,
                                                    mujoco.mjtFontScale.mjFONTSCALE_150)
                        self._model_changed = False
                    fw, fh = glfw.get_framebuffer_size(window)
                    viewport = mujoco.MjrRect(0, 0, fw, fh)
                    mujoco.mjr_render(viewport, self._scene, context)
                    if self._snap is not None and not self._snap:
                        import numpy as np

                        rgb = np.zeros((fh, fw, 3), dtype=np.uint8)
                        mujoco.mjr_readPixels(rgb, None, viewport, context)
                        self._snap.append(rgb[::-1])
                glfw.swap_buffers(window)
        finally:
            if context is not None:
                context.free()
            glfw.destroy_window(window)
