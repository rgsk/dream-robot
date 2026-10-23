"""Grip events of a trial, from its trace.json: closes, opens, fingertip contact, and object lifts.

    python3 experiments/dream1/scripts/grip_events.py scratch/dream1/apple_bowl_50 [--arm left] [--obj apple]

Prints one line per change: when the commanded grip crosses 0.5, when the fingertips touching the
object change, and when the object's height leaves / returns to its resting height (+-1 cm).
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--arm", default="left")
    p.add_argument("--obj", default="apple")
    a = p.parse_args()
    trace = json.loads((a.run / "trace.json").read_text())
    rest = trace[0]["z"][a.obj]
    closed = tips = lifted = None
    peak = rest
    for row in trace:
        t, z = row["t"], row["z"][a.obj]
        peak = max(peak, z)
        now_closed = row["cmd_grip"][a.arm] < 0.5
        if now_closed != closed:
            print(f"{t:6.2f}s  cmd {'CLOSE' if now_closed else 'open '} ({row['cmd_grip'][a.arm]:.2f})"
                  f"  opening {row['opening'][a.arm]:.2f}  {a.obj} z {z:.3f}")
            closed = now_closed
        now_tips = [x for x in row["apple_tips"] if x.startswith(a.arm)] if a.obj == "apple" else None
        if now_tips is not None and now_tips != tips:
            print(f"{t:6.2f}s  tips on {a.obj}: {now_tips or 'none'}  opening {row['opening'][a.arm]:.2f}")
            tips = now_tips
        now_lifted = z > rest + 0.01
        if now_lifted != lifted:
            print(f"{t:6.2f}s  {a.obj} {'LIFTED' if now_lifted else 'at rest'}  z {z:.3f}")
            lifted = now_lifted
    print(f"rest z {rest:.3f}, peak z {peak:.3f} (+{(peak - rest) * 100:.1f} cm), {len(trace)} steps")


if __name__ == "__main__":
    main()
