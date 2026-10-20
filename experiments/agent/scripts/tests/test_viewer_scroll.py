"""Does the run viewer follow a live run, and let the user scroll away and back? Real Chrome, real page.

A thread replays a real run's stream log into a temp folder one line every 0.3 s (a live run);
the viewer serves that folder; Playwright drives Chrome and measures, once a second:
  gap    = page height - (scroll position + window height): 0 means "at the bottom"
  follow = the checkbox
  y      = scroll position

Needs Playwright (pip install playwright==1.48.0) and an installed Chrome; replays a run from
scratch/agent/headless. Not part of pytest: run it directly with that Python.
"""

import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path("/home/rahul/Documents/codes/ml/dream-robot")
SOURCE = REPO / "scratch/agent/headless/0-dream1-wait5.jsonl"
PORT = 8799
STEP_S = 0.3

tmp = Path(tempfile.mkdtemp())
lines = SOURCE.read_text().splitlines(keepends=True)
live = tmp / "live-run.jsonl"
live.write_text("")


def replay():
    with open(live, "a") as f:
        for line in lines + lines[2:]:                # the run twice over: entries keep coming
            f.write(line)
            f.flush()
            time.sleep(STEP_S)


viewer = subprocess.Popen([sys.executable, str(REPO / "experiments/agent/scripts/run_viewer.py"),
                           "--dir", str(tmp), "--port", str(PORT)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
failures = []


def state(page):
    return page.evaluate("""() => ({
        gap: Math.round(document.documentElement.scrollHeight - (window.scrollY + window.innerHeight)),
        y: Math.round(window.scrollY), follow: document.getElementById('follow').checked,
        cards: document.querySelectorAll('#tl > *').length })""")


def show(phase, s):
    print(f"{phase:10} gap {s['gap']:6}  y {s['y']:6}  follow {str(s['follow']):5}  entries {s['cards']}")


try:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 700})
        threading.Thread(target=replay, daemon=True).start()
        page.goto(f"http://localhost:{PORT}/")

        # Phase 1: just follow. After a moment to settle, the view should sit at the bottom.
        for i in range(8):
            time.sleep(1)
            s = state(page)
            show("follow", s)
            if i >= 2 and s["gap"] > 60:
                failures.append(f"follow: {s['gap']} px short of the bottom")

        # Phase 2: the user scrolls up. Follow should turn off and the view stay put.
        page.mouse.move(500, 400)
        page.mouse.wheel(0, -1500)
        time.sleep(0.5)
        before = state(page)
        show("scrolled up", before)
        if before["follow"]:
            failures.append("scrolling up did not turn follow off")
        for _ in range(4):
            time.sleep(1)
            s = state(page)
            show("away", s)
            if abs(s["y"] - before["y"]) > 5:
                failures.append(f"view moved while the user was away: y {before['y']} -> {s['y']}")

        # Phase 2b/2c: away by the scrollbar (a scroll with no wheel event) and by the keyboard.
        for how, act in (("scrollbar", lambda: page.evaluate("window.scrollBy(0, -1500)")),
                         ("PageUp", lambda: (page.mouse.click(500, 60), [page.keyboard.press("PageUp") for _ in range(3)]))):
            page.mouse.wheel(0, 100000)              # start from the bottom, following
            time.sleep(1.2)
            act()
            time.sleep(0.4)
            before = state(page)
            show(how, before)
            if before["follow"]:
                failures.append(f"{how}: scrolling up did not turn follow off")
            for _ in range(3):
                time.sleep(1)
                s = state(page)
                show("away", s)
                if abs(s["y"] - before["y"]) > 5:
                    failures.append(f"{how}: view moved while the user was away: y {before['y']} -> {s['y']}")

        # Phase 3: the user scrolls back to the bottom. Follow should come back on and keep up.
        page.mouse.wheel(0, 100000)
        time.sleep(0.5)
        s = state(page)
        show("back down", s)
        if not s["follow"]:
            failures.append("scrolling back to the bottom did not turn follow on")
        for i in range(5):
            time.sleep(1)
            s = state(page)
            show("follow", s)
            if i >= 1 and s["gap"] > 60:
                failures.append(f"follow after return: {s['gap']} px short of the bottom")
        browser.close()
finally:
    viewer.terminate()
    shutil.rmtree(tmp, ignore_errors=True)

print("\nPASS" if not failures else "\nFAIL:\n  " + "\n  ".join(dict.fromkeys(failures)))
