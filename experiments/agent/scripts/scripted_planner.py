"""A stand-in planner for T5: parse "put the {colour} cube in the {side} box", then
find_by_colour -> pick(grasp) -> find_objects for the bin -> place_at, through the
MCP tools (in-process). The side picks the container from the image: "left" is the
leftmost the top camera sees. Success comes from the referee (score()), never a tool.

No LLM: this checks the tools, not the planning. Prints one line per episode.
"""

import argparse
import asyncio
import base64
import json
import re

from mcp import Client

from dream_robot.agent.server import build_scored_server
from dream_robot.core.registry import task_factory

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=10)
p.add_argument("--first", type=int, default=1000)
p.add_argument("--save", default="scratch/agent/plan")
args = p.parse_args()

server, score = build_scored_server(task_factory("robosuite/instructed_pick_place"))


def data(result):
    return json.loads(result.content[0].text)


async def episode(c, seed):
    status = (await c.call_tool("reset", {"seed": seed})).structured_content
    colour, side = re.match(r"put the (\w+) cube in the (\w+) box", status["instruction"]).groups()
    found = data(await c.call_tool("find_by_colour", {"colour": colour}))
    if not found["found"]:
        return f"find: {found}"
    pick = await c.call_tool("pick", {"grasp": found["grasp"]})
    if pick.is_error:
        return f"pick error: {pick.content[0].text}"
    picked = data(pick)
    if not picked["ok"]:
        return f"pick {picked}"
    things = data(await c.call_tool("find_objects", {}))["things"]
    bins = [t for t in things if t["kind"] == "container"]
    bin_ = bins[0] if side == "left" else bins[-1]          # left to right in the image
    place = await c.call_tool("place_at", {"point": bin_["drop_point"]})
    placed = data(place)
    open(f"{args.save}_{seed}.png", "wb").write(base64.b64decode(place.content[1].data))
    final = score()
    return (f"{colour:6s}->{side:5s} pick {picked['steps']} steps, "
            f"place {placed['failure'] if not placed['ok'] else 'ok'} ({placed['steps']}); "
            f"task success {final['success']}, {final['steps']} steps")


async def main():
    async with Client(server) as c:
        wins = 0
        for seed in range(args.first, args.first + args.seeds):
            line = await episode(c, seed)
            wins += "task success True" in line
            print(f"seed {seed}: {line}", flush=True)
        print(f"\n{wins}/{args.seeds} task successes")


asyncio.run(main())
