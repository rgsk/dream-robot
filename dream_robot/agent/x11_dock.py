"""Move this process's window to the right third of the screen, through X11.

For ``--viewer docked``. Wayland lets no app place its own window, so docked
mode runs the viewer on XWayland, where the window manager still takes
requests. The request is the desktop's standard one (EWMH
``_NET_MOVERESIZE_WINDOW``, sent to the root window): a plain
``XMoveResizeWindow`` was ignored by GNOME, which places new windows itself.

libX11 through ctypes: it is on every Linux desktop, and this is ~10 calls.
The window is found by ``_NET_WM_PID``, so a server only ever moves its own
window, never another server's.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import time

_ClientMessage = 33
_SubstructureNotify, _SubstructureRedirect = 1 << 19, 1 << 20


class _ClientMessageEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int), ("serial", ctypes.c_ulong), ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p), ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong), ("format", ctypes.c_int),
        ("data", ctypes.c_long * 5), ("pad", ctypes.c_long * 16),   # XEvent is 24 longs
    ]


def dock_right_third_async(pid: int | None = None) -> None:
    """Dock in the background: the window may not exist yet, and the sim should not wait."""
    threading.Thread(target=dock_right_third, args=(pid or os.getpid(),), daemon=True).start()


def dock_right_third(pid: int, timeout: float = 5.0) -> bool:
    x = _libx11()
    if x is None:
        return False
    dpy = x.XOpenDisplay(None)
    if not dpy:
        return False
    try:
        root = x.XDefaultRootWindow(dpy)
        deadline = time.monotonic() + timeout
        win = 0
        while not win and time.monotonic() < deadline:
            win = _find(x, dpy, root, pid)
            time.sleep(0.1)
        if not win:
            return False
        time.sleep(0.5)                       # let GNOME finish its own placement first
        W, H = x.XDisplayWidth(dpy, 0), x.XDisplayHeight(dpy, 0)
        atom = lambda name: x.XInternAtom(dpy, name, 0)  # noqa: E731
        for _ in range(3):                    # GNOME can overrule the first request
            # Un-maximise (action 0 = remove), source 2 = pager: a user-level request.
            _send(x, dpy, root, win, atom(b"_NET_WM_STATE"),
                  0, atom(b"_NET_WM_STATE_MAXIMIZED_VERT"),
                  atom(b"_NET_WM_STATE_MAXIMIZED_HORZ"), 2)
            # flags: x, y, width, height all given (bits 8-11); source pager (2 << 12).
            _send(x, dpy, root, win, atom(b"_NET_MOVERESIZE_WINDOW"),
                  (0xF << 8) | (2 << 12), W - W // 3, 0, W // 3, H)
            time.sleep(0.3)
        return True
    finally:
        x.XCloseDisplay(dpy)


def _send(x, dpy, root, win, message_type, *data) -> None:
    ev = _ClientMessageEvent(type=_ClientMessage, send_event=1, display=dpy, window=win,
                             message_type=message_type, format=32)
    for i, v in enumerate(data):
        ev.data[i] = v
    x.XSendEvent(dpy, root, 0, _SubstructureRedirect | _SubstructureNotify, ctypes.byref(ev))
    x.XFlush(dpy)


def _find(x, dpy, win, pid: int, depth: int = 0) -> int:
    """Depth-first search for a window whose _NET_WM_PID is ``pid``."""
    if _pid_of(x, dpy, win) == pid:
        return win
    if depth >= 4:
        return 0
    root, parent = ctypes.c_ulong(), ctypes.c_ulong()
    kids, n = ctypes.POINTER(ctypes.c_ulong)(), ctypes.c_uint()
    if not x.XQueryTree(dpy, win, ctypes.byref(root), ctypes.byref(parent),
                        ctypes.byref(kids), ctypes.byref(n)):
        return 0
    try:
        for i in range(n.value):
            if found := _find(x, dpy, kids[i], pid, depth + 1):
                return found
        return 0
    finally:
        if kids:
            x.XFree(kids)


def _pid_of(x, dpy, win) -> int | None:
    actual_type, actual_format = ctypes.c_ulong(), ctypes.c_int()
    nitems, after = ctypes.c_ulong(), ctypes.c_ulong()
    prop = ctypes.POINTER(ctypes.c_ubyte)()
    ok = x.XGetWindowProperty(dpy, win, x.XInternAtom(dpy, b"_NET_WM_PID", 1), 0, 1, 0, 0,
                              ctypes.byref(actual_type), ctypes.byref(actual_format),
                              ctypes.byref(nitems), ctypes.byref(after), ctypes.byref(prop))
    try:
        if ok != 0 or nitems.value != 1 or actual_format.value != 32:
            return None
        return ctypes.cast(prop, ctypes.POINTER(ctypes.c_long))[0]
    finally:
        if prop:
            x.XFree(prop)


def _libx11():
    path = ctypes.util.find_library("X11")
    if path is None:
        return None
    x = ctypes.CDLL(path)
    vp, ul = ctypes.c_void_p, ctypes.c_ulong
    x.XOpenDisplay.restype = vp
    x.XCloseDisplay.argtypes = [vp]
    x.XDefaultRootWindow.restype, x.XDefaultRootWindow.argtypes = ul, [vp]
    x.XDisplayWidth.argtypes = x.XDisplayHeight.argtypes = [vp, ctypes.c_int]
    x.XInternAtom.restype, x.XInternAtom.argtypes = ul, [vp, ctypes.c_char_p, ctypes.c_int]
    x.XQueryTree.argtypes = [vp, ul, ctypes.POINTER(ul), ctypes.POINTER(ul),
                             ctypes.POINTER(ctypes.POINTER(ul)), ctypes.POINTER(ctypes.c_uint)]
    x.XGetWindowProperty.argtypes = [vp, ul, ul, ctypes.c_long, ctypes.c_long, ctypes.c_int, ul,
                                     ctypes.POINTER(ul), ctypes.POINTER(ctypes.c_int),
                                     ctypes.POINTER(ul), ctypes.POINTER(ul),
                                     ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))]
    x.XSendEvent.argtypes = [vp, ul, ctypes.c_int, ctypes.c_long, vp]
    x.XFlush.argtypes = [vp]
    x.XFree.argtypes = [vp]
    return x
