"""Readable log of a `claude -p --output-format stream-json` run: what it says, calls, gets back."""

import json
import sys

for line in sys.stdin:
    d = json.loads(line)
    t = d.get("type")
    if t == "system" and d.get("subtype") == "init":
        print(f"session {d['session_id']}  tools {len(d.get('tools', []))}", flush=True)
    elif t == "assistant":
        for c in d["message"]["content"]:
            if c["type"] == "text" and c["text"].strip():
                print("CLAUDE:", c["text"].strip(), flush=True)
            elif c["type"] == "tool_use":
                print("CALL  ", c["name"].split("__")[-1], json.dumps(c["input"]), flush=True)
    elif t == "user":
        for c in d["message"]["content"]:
            if c.get("type") == "tool_result":
                parts = c["content"] if isinstance(c["content"], list) else [
                    {"type": "text", "text": str(c["content"])}]
                txt = " ".join(p.get("text", "") for p in parts if p.get("type") == "text")
                print("   ->", txt.replace("\n", " ")[:160], flush=True)
    elif t == "result":
        print(f"DONE  turns {d.get('num_turns')}  {d.get('duration_ms', 0) / 1000:.0f} s  "
              f"usage {json.dumps(d.get('usage'))}", flush=True)
