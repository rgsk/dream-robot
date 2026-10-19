"""Token and time cost of an LLM driving the robot, read from a Claude Code session transcript.

For every robot tool call (an MCP tool whose name contains --server, default t7):
  context  -- tokens the model read to make this call (input + cache write + cache read)
  added    -- context growth since the previous call: the tool result + the model's own text
  image    -- the returned image's size, and its estimated tokens (w * h / 750)
  tool_s   -- wall time from the call to its result (sim + render + recording)
  model_s  -- wall time from the previous result to this call (the model thinking)

Usage:
  python experiments/agent/scripts/token_cost.py <session.jsonl> [--from-reset SEED]
Transcripts: ~/.claude/projects/<project>/<session-id>.jsonl
"""

import argparse
import base64
import json
import struct
from datetime import datetime


def png_size(b64: str) -> tuple[int, int]:
    head = base64.b64decode(b64[:64])
    return struct.unpack(">II", head[16:24])


def ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--server", default="t7")
    ap.add_argument("--from-reset", type=int, default=None, help="start at reset(seed=...)")
    args = ap.parse_args()

    msgs = {}  # assistant message id -> usage, first timestamp
    calls = []  # tool_use in order: id, name, input, message id, timestamp
    results = {}  # tool_use_id -> timestamp, text chars, image sizes
    for line in open(args.transcript):
        d = json.loads(line)
        if d.get("type") == "assistant":
            m = d["message"]
            msgs.setdefault(m["id"], {"usage": m.get("usage", {}), "t": ts(d["timestamp"])})
            for c in m["content"]:
                if c.get("type") == "tool_use" and args.server in c["name"]:
                    calls.append({"id": c["id"], "name": c["name"].split("__")[-1],
                                  "input": c["input"], "msg": m["id"], "t": ts(d["timestamp"])})
        elif d.get("type") == "user" and isinstance(d["message"]["content"], list):
            for c in d["message"]["content"]:
                if c.get("type") != "tool_result":
                    continue
                parts = c["content"] if isinstance(c["content"], list) else []
                imgs = [png_size(p["source"]["data"]) for p in parts if p.get("type") == "image"]
                chars = sum(len(p.get("text", "")) for p in parts if p.get("type") == "text")
                results[c["tool_use_id"]] = {"t": ts(d["timestamp"]), "imgs": imgs, "chars": chars}

    if args.from_reset is not None:
        start = next(i for i, c in enumerate(calls)
                     if c["name"] == "reset" and c["input"].get("seed") == args.from_reset)
        end = next((i for i, c in enumerate(calls[start + 1:], start + 1) if c["name"] == "reset"),
                   len(calls))
        calls = calls[start:end]

    def context(u):
        return (u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                + u.get("cache_read_input_tokens", 0))

    print(f"{'#':>2} {'tool':13} {'context':>8} {'added':>6} {'image':>9} {'img_tok':>7} "
          f"{'json_ch':>7} {'tool_s':>6} {'model_s':>7}")
    tot = {"img_tok": 0, "json_ch": 0, "tool_s": 0.0, "model_s": 0.0, "added": 0}
    prev_ctx, prev_res_t = None, None
    for i, c in enumerate(calls, 1):
        u = msgs[c["msg"]]["usage"]
        ctx = context(u)
        added = ctx - prev_ctx if prev_ctx is not None else 0
        r = results.get(c["id"], {"t": c["t"], "imgs": [], "chars": 0})
        img = "x".join(map(str, r["imgs"][0])) if r["imgs"] else "-"
        img_tok = sum(w * h // 750 for w, h in r["imgs"])
        tool_s = r["t"] - c["t"]
        model_s = c["t"] - prev_res_t if prev_res_t is not None else 0.0
        print(f"{i:>2} {c['name']:13} {ctx:>8} {added:>6} {img:>9} {img_tok:>7} "
              f"{r['chars']:>7} {tool_s:>6.1f} {model_s:>7.1f}")
        for k, v in (("img_tok", img_tok), ("json_ch", r["chars"]), ("tool_s", tool_s),
                     ("model_s", model_s), ("added", added)):
            tot[k] += v
        prev_ctx, prev_res_t = ctx, r["t"]

    n = len(calls)
    first = context(msgs[calls[0]["msg"]]["usage"])
    read = sum(context(msgs[c["msg"]]["usage"]) for c in calls)
    out = sum(msgs[m]["usage"].get("output_tokens", 0) for m in {c["msg"] for c in calls})
    print(f"\ncalls {n}; context at first call {first}, at last {prev_ctx}")
    print(f"tokens read over the episode (sum of contexts) {read}; output {out}")
    print(f"added {tot['added']}: images ~{tot['img_tok']}, json ~{tot['json_ch'] // 4} "
          f"({tot['json_ch']} chars / 4), rest = model's own text/thinking")
    print(f"wall: tool {tot['tool_s']:.0f} s, model {tot['model_s']:.0f} s")


if __name__ == "__main__":
    main()
