"""Opening the viewer on a long run that is already far along: does it land at the newest entry?
Needs Playwright (pip install playwright==1.48.0) and an installed Chrome; replays a run from
scratch/agent/headless. Not part of pytest: run it directly with that Python.
"""
import shutil, subprocess, sys, tempfile, time
from pathlib import Path
from playwright.sync_api import sync_playwright
REPO = Path("/home/rahul/Documents/codes/ml/dream-robot")
tmp = Path(tempfile.mkdtemp())
shutil.copy(REPO / "scratch/agent/headless/0-dream1-wrist.jsonl", tmp / "long-run.jsonl")
v = subprocess.Popen([sys.executable, str(REPO / "experiments/agent/scripts/run_viewer.py"), "--dir", str(tmp),
                      "--port", "8798"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)
try:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1000, "height": 700})
        page.goto("http://localhost:8798/")
        for i in range(4):
            time.sleep(1)
            s = page.evaluate("""() => ({gap: Math.round(document.documentElement.scrollHeight - scrollY - innerHeight),
                height: document.documentElement.scrollHeight, imgs: document.images.length,
                loaded: [...document.images].filter(i => i.complete).length})""")
            print(s)
        print("PASS" if s["gap"] <= 60 else "FAIL: not at the newest entry")
        b.close()
finally:
    v.terminate(); shutil.rmtree(tmp, ignore_errors=True)
