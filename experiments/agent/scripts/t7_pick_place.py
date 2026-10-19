"""T7 through the MCP tools, without an LLM: find_objects -> pick(grasp) -> place_at.

Scores the executor, not the naming: the target is matched to a found object by
ground truth (nearest to the named object's true position, read from a second
"referee" env reset with the same seed -- the server's sim is never touched) -- the step an LLM does
from the numbered image. The bin is the found container on the instruction's side
(left = -y). Everything the arm does comes from the tools' own outputs.
"""

import argparse
import asyncio
import json
from collections import defaultdict

import numpy as np

from dream_robot.agent.server import build_server
from dream_robot.core.registry import task_factory

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=20)
p.add_argument("--first", type=int, default=1000)
args = p.parse_args()

server = build_server(task_factory("robosuite/assorted_pick_place"))
env = task_factory("robosuite/assorted_pick_place")()      # the referee

async def call(name, arguments=None):
    out = await server.call_tool(name, arguments or {})
    if out.is_error:
        raise RuntimeError(f"{name}: {out.content[0].text}")
    return json.loads(out.content[0].text)


async def episode(seed):
    await call("reset", {"seed": seed})
    env.reset(seed=seed)
    target = env.target
    truth = env.target_pos
    things = (await call("find_objects"))["things"]
    objs = [t for t in things if t["kind"] == "object"]
    bins = [t for t in things if t["kind"] == "container"]
    chosen = min(objs, key=lambda t: np.linalg.norm(np.array(t["centre"][:2]) - truth[:2]))
    side = -1 if env.instruction.side == "left" else 1
    bin_ = max(bins, key=lambda t: side * t["centre"][1])
    picked = await call("pick", {"grasp": chosen["grasp"]})
    placed = await call("place_at", {"point": bin_["drop_point"]}) if picked["ok"] else None
    final = await call("status")
    return target, chosen, picked, placed, final


async def main():
    by_kind = defaultdict(lambda: [0, 0])
    wins = 0
    for seed in range(args.first, args.first + args.seeds):
        target, chosen, picked, placed, final = await episode(seed)
        ok = final["success"]
        wins += ok
        by_kind[target.kind][0] += ok
        by_kind[target.kind][1] += 1
        why = ("ok" if ok else
               f"pick {picked['failure']}@{picked['phase']}" if not picked["ok"] else
               f"place {placed['failure']}@{placed['phase']}" if not placed["ok"] else
               f"placed, task {final['failure_mode']}")
        g = chosen["grasp"]
        print(f"seed {seed}: {final['instruction']:40s} grasp yaw {g['yaw_deg']:6.1f} "
              f"w {g['width']*100:4.1f}cm{' WIDE' if g['too_wide'] else ''}  -> {why}", flush=True)
    print(f"\n{wins}/{args.seeds} task successes")
    for k, (w, n) in sorted(by_kind.items()):
        print(f"  {k:9s} {w}/{n}")


asyncio.run(main())
