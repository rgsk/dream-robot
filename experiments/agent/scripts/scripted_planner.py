"""A stand-in planner for T5: parse "put the {colour} cube in the {side} box", then
find_objects (the named colour, and the bin) -> the driver prompt's pick and place recipes
over the primitives (move_to, gripper, move_by), through the MCP tools (in-process).
The side picks the container from the image: "left" is the
leftmost the top camera sees. Success comes from the referee (score()), never a tool.

No LLM: this checks the tools, not the planning. Prints one line per episode.
"""

import argparse
import asyncio
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
    things = data(await c.call_tool("find_objects", {}))["things"]
    bins = [t for t in things if t["kind"] == "container"]
    bin_ = bins[0] if side == "left" else bins[-1]          # left to right in the image
    cubes = [t for t in things if t["kind"] == "object" and t["colour"] == colour]
    if len(cubes) != 1:
        return f"find: {len(cubes)} {colour} objects"
    found = cubes[0]
    # The driver prompt's pick and place recipes, over the primitives.
    x, y, z = found["grasp"]["point"]
    bx, by, bz = bin_["drop_point"]
    hang = found["grasp"]["above_table"]
    steps = [("move_to", {"point": [x, y, z + 0.10], "yaw_deg": found["grasp"]["yaw_deg"]}),
             ("move_to", {"point": [x, y, z]}),
             ("gripper", {"action": "close"}), ("move_by", {"dz": 0.2}),
             ("move_to", {"point": [bx, by, bz + hang + 0.15]}),
             ("move_to", {"point": [bx, by, bz + hang + 0.02]}),
             ("gripper", {"action": "open"}), ("move_by", {"dz": 0.12})]
    for name, args in steps:
        r = await c.call_tool(name, args)
        if r.is_error:
            return f"{name} error: {r.content[0].text}"
        d = data(r)
        if name == "gripper" and args["action"] == "close" and not d["hand"]["holding"]:
            return "closed on nothing"
    final = score()
    return f"{colour:6s}->{side:5s} task success {final['success']}, {final['steps']} steps"


async def main():
    async with Client(server) as c:
        wins = 0
        for seed in range(args.first, args.first + args.seeds):
            line = await episode(c, seed)
            wins += "task success True" in line
            print(f"seed {seed}: {line}", flush=True)
        print(f"\n{wins}/{args.seeds} task successes")


asyncio.run(main())
