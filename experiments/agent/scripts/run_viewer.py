"""A live web view of a headless driver run: what the LLM said, called, and saw -- images included.

headless_drive.sh writes claude -p's stream to scratch/agent/headless/<run>.jsonl; every tool
result there carries the images exactly as the model received them. This serves one page that
renders that file as a timeline and follows it while the run is going (polls once a second).

    python3 experiments/agent/scripts/run_viewer.py            # http://localhost:8765
    python3 experiments/agent/scripts/run_viewer.py --port 9000 --dir scratch/agent/headless

Standard library only. Pick a run from the menu (newest first); it shows the newest by default.
"""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Run viewer</title>
<style>
:root { --bg:#f6f6f4; --card:#fff; --ink:#1d1d1b; --muted:#6b6b66; --line:#e3e2dc;
        --call:#2f5d8a; --ok:#2e7d4f; --err:#b3261e; --think:#7a6a9a; color-scheme: light; }
@media (prefers-color-scheme: dark) { :root { --bg:#161615; --card:#21211f; --ink:#ecebe6;
        --muted:#9a998f; --line:#34332f; --call:#8db8e3; --ok:#7fc79b; --err:#f2a39c;
        --think:#b9a8dc; color-scheme: dark; } }
* { box-sizing: border-box; }
html { overflow-anchor: none; }   /* no browser nudges: every scroll is ours or the user's */
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
header h1 { font-size:15px; margin:0; }
select { font:inherit; padding:4px 6px; max-width:100%; }
#live { font-size:12px; color:var(--muted); }
#live.on::before { content:"● "; color:var(--ok); }
main { max-width:980px; margin:0 auto; padding:16px; display:flex; flex-direction:column; gap:10px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.n { color:var(--muted); font-size:12px; margin-right:6px; font-variant-numeric:tabular-nums; }
.say { white-space:pre-wrap; }
.call { border-left:4px solid var(--call); }
.call .name { font-weight:600; color:var(--call); font-family:ui-monospace, monospace; }
.args { font-family:ui-monospace, monospace; font-size:12.5px; color:var(--muted); margin-left:6px; }
.result { margin-top:8px; padding-top:8px; border-top:1px dashed var(--line); }
.result.err pre { color:var(--err); }
pre { margin:0; white-space:pre-wrap; word-break:break-word; font:12.5px/1.4 ui-monospace, monospace; }
.imgs { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
.imgs img { max-width:100%; border-radius:4px; border:1px solid var(--line); cursor:zoom-in; }
details.think summary { color:var(--think); cursor:pointer; font-size:12.5px; }
details.think div { white-space:pre-wrap; color:var(--muted); font-size:13px; margin-top:6px; }
.done { border-left:4px solid var(--ok); }
.empty { color:var(--muted); text-align:center; padding:40px; }
#zoom { position:fixed; inset:0; background:rgba(0,0,0,.85); display:flex; align-items:center;
        justify-content:center; z-index:9; cursor:zoom-out; }
#zoom img { max-width:96vw; max-height:96vh; }
#zoom[hidden] { display:none; }
</style></head><body>
<header><h1>Run viewer</h1><select id="run"></select><span id="live"></span>
<label style="font-size:12px;color:var(--muted)"><input type="checkbox" id="follow" checked> follow</label></header>
<main id="tl"><div class="empty">loading…</div></main>
<div id="zoom" hidden><img></div>
<script>
const tl = document.getElementById('tl'), sel = document.getElementById('run');
const live = document.getElementById('live'), follow = document.getElementById('follow');
const zoom = document.getElementById('zoom');
if (new URLSearchParams(location.search).get('follow') === '0') follow.checked = false;
let run = null, count = 0, calls = {}, ncall = 0;
zoom.onclick = () => { zoom.hidden = true; };
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls;
  if (text != null) e.textContent = text; return e; }
function images(list) { const box = el('div', 'imgs');
  for (const src of list) { const im = el('img'); im.onload = stick; im.src = src;
    im.onclick = () => { zoom.querySelector('img').src = src; zoom.hidden = false; }; box.append(im); }
  return box; }
// Follow the newest entry: scroll to the end now, and again as each image loads (it grows the
// page). Any scroll the user makes -- wheel, trackpad, scrollbar, keys -- decides by where it
// lands: away from the end stops following, at the end resumes it. Our own scrolls are skipped.
const page = document.scrollingElement;
let ours = null;                                       // the scrollTop our last stick() aimed at
function atEnd() { return page.scrollTop + window.innerHeight >= page.scrollHeight - 40; }
function stick() {
  if (!follow.checked) return;
  const target = page.scrollHeight - window.innerHeight;
  if (Math.abs(page.scrollTop - target) < 2) return;  // already there: no scroll event will come
  ours = target; page.scrollTop = target; }
window.addEventListener('scroll', () => {
  if (ours !== null && Math.abs(page.scrollTop - ours) < 2) { ours = null; return; }
  ours = null; follow.checked = atEnd(); }, {passive: true});
follow.onchange = stick;
function pretty(text) { try { return JSON.stringify(JSON.parse(text), null, 1); } catch { return text; } }
function add(e) {
  if (e.kind === 'text') { const c = el('div', 'card'); c.append(el('div', 'say', e.text)); tl.append(c); }
  else if (e.kind === 'thinking') { const d = el('details', 'card think');
    d.append(el('summary', null, 'thinking')); d.append(el('div', null, e.text)); tl.append(d); }
  else if (e.kind === 'call') { const c = el('div', 'card call'); ncall += 1;
    c.append(el('span', 'n', '#' + ncall)); c.append(el('span', 'name', e.name));
    c.append(el('span', 'args', e.args)); calls[e.id] = c; tl.append(c); }
  else if (e.kind === 'result') { const host = calls[e.id] || (() => { const c = el('div', 'card'); tl.append(c); return c; })();
    const r = el('div', 'result' + (e.error ? ' err' : ''));
    if (e.text) r.append(el('pre', null, pretty(e.text)));
    if (e.images.length) r.append(images(e.images));
    host.append(r); }
  else if (e.kind === 'done') { const c = el('div', 'card done'); c.append(el('pre', null, e.text)); tl.append(c); }
}
async function runs() {
  const list = await (await fetch('/api/runs')).json();
  const keep = sel.value;
  sel.innerHTML = '';
  for (const r of list) { const o = el('option', null, r.name + '  (' + r.when + ')'); o.value = r.name; sel.append(o); }
  if (keep && list.some(r => r.name === keep)) sel.value = keep;
  return list;
}
function open(name) { run = name; count = 0; calls = {}; ncall = 0; tl.innerHTML = ''; poll(); }
sel.onchange = () => open(sel.value);
async function poll() {
  if (!run) return;
  const mine = run;
  const r = await (await fetch('/api/entries?run=' + encodeURIComponent(run) + '&since=' + count)).json();
  if (mine !== run) return;
  if (!count && !r.entries.length) tl.innerHTML = '<div class="empty">nothing yet</div>';
  else if (!count) tl.innerHTML = '';
  for (const e of r.entries) add(e);
  count += r.entries.length;
  live.textContent = r.finished ? 'finished' : 'running'; live.className = r.finished ? '' : 'on';
  if (r.entries.length) stick();
}
(async () => { const list = await runs(); if (list.length) open(list[0].name);
  setInterval(poll, 1000); setInterval(runs, 5000); })();
</script></body></html>
"""


class Runs:
    """Parsed entries per run file, read incrementally as the file grows."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.cache = {}                       # name -> (offset, entries, finished)

    def list(self):
        import time

        files = sorted(self.folder.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [{"name": p.stem, "when": time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))}
                for p in files]

    def entries(self, name):
        path = self.folder / f"{Path(name).name}.jsonl"
        offset, entries, finished = self.cache.get(name, (0, [], False))
        if path.stat().st_size < offset:      # rewritten: start over
            offset, entries, finished = 0, [], False
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
        complete = data.rfind(b"\n") + 1      # only whole lines; a line being written waits
        for line in data[:complete].splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            entries.extend(_entries(d))
            finished = finished or d.get("type") == "result"
        self.cache[name] = (offset + complete, entries, finished)
        return entries, finished


def _entries(d):
    """The timeline entries in one stream-json line."""
    kind = d.get("type")
    out = []
    if kind == "assistant":
        for b in d["message"].get("content", []):
            if b.get("type") == "text" and b["text"].strip():
                out.append({"kind": "text", "text": b["text"]})
            elif b.get("type") == "thinking" and b.get("thinking", "").strip():
                out.append({"kind": "thinking", "text": b["thinking"]})
            elif b.get("type") == "tool_use":
                name = b["name"].split("__")[-1]
                out.append({"kind": "call", "id": b["id"], "name": name,
                            "args": json.dumps(b.get("input", {}), ensure_ascii=False)})
    elif kind == "user":
        content = d["message"].get("content", [])
        for b in content if isinstance(content, list) else []:
            if b.get("type") != "tool_result":
                continue
            texts, images = [], []
            parts = b.get("content")
            for p in parts if isinstance(parts, list) else [{"type": "text", "text": str(parts or "")}]:
                if p.get("type") == "text":
                    texts.append(p["text"])
                elif p.get("type") == "image":
                    src = p["source"]
                    images.append(f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}")
            out.append({"kind": "result", "id": b.get("tool_use_id"), "text": "\n".join(texts),
                        "images": images, "error": bool(b.get("is_error"))})
    elif kind == "result":
        usage = d.get("usage", {})
        out.append({"kind": "done", "text": (
            f"done: {d.get('subtype')} · {d.get('num_turns')} turns · "
            f"{(d.get('duration_ms') or 0) / 1000:.0f} s · read {usage.get('cache_read_input_tokens', 0):,} "
            f"cached + {usage.get('input_tokens', 0) + usage.get('cache_creation_input_tokens', 0):,} tokens · "
            f"wrote {usage.get('output_tokens', 0):,}")})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default="scratch/agent/headless")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    runs = Runs(args.dir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            try:
                if url.path == "/":
                    self.send(PAGE, "text/html; charset=utf-8")
                elif url.path == "/api/runs":
                    self.send(json.dumps(runs.list()), "application/json")
                elif url.path == "/api/entries":
                    entries, finished = runs.entries(q["run"][0])
                    since = int(q.get("since", ["0"])[0])
                    self.send(json.dumps({"entries": entries[since:], "finished": finished}), "application/json")
                else:
                    self.send_error(404)
            except (KeyError, FileNotFoundError) as exc:
                self.send_error(404, str(exc))

    print(f"run viewer: http://localhost:{args.port}  (runs in {os.path.abspath(args.dir)})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
