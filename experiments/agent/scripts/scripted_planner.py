"""A stand-in planner: parse "put the {colour} cube in the {side} box", then
find_by_colour -> pick -> place, all through the MCP server over stdio.

No LLM: this checks the tools, not the planning. Prints one line per episode.
"""

import argparse
import asyncio
import base64
import json
import os
import re

from mcp import Client, StdioServerParameters

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=10)
p.add_argument("--first", type=int, default=1000)
p.add_argument("--save", default="scratch/agent/plan")
args = p.parse_args()

cfg = json.load(open(".mcp.json"))["mcpServers"]["dream-robot"]
server = StdioServerParameters(
    command=cfg["command"], args=cfg["args"], env={**os.environ, **cfg["env"]}
)


def data(result):
    return json.loads(result.content[0].text)


async def episode(c, seed):
    status = (await c.call_tool("reset", {"seed": seed})).structured_content
    colour, side = re.match(r"put the (\w+) cube in the (\w+) box", status["instruction"]).groups()
    found = data(await c.call_tool("find_by_colour", {"colour": colour}))
    if not found["found"]:
        return f"find: {found}"
    pick = await c.call_tool("pick", {"point": found["point"]})
    if pick.is_error:
        return f"pick error: {pick.content[0].text}"
    picked = data(pick)
    if not picked["ok"]:
        return f"pick {picked}"
    place = await c.call_tool("place", {"bin": side})
    placed = data(place)
    open(f"{args.save}_{seed}.png", "wb").write(base64.b64decode(place.content[1].data))
    final = (await c.call_tool("status", {})).structured_content
    return (f"{colour:6s}->{side:5s} pick {picked['steps']} steps, "
            f"place {placed['failure'] if not placed['ok'] else 'ok'} ({placed['steps']}); "
            f"task success {final['success']}, {final['steps']} steps")


async def main():
    async with Client(server, read_timeout_seconds=300) as c:
        wins = 0
        for seed in range(args.first, args.first + args.seeds):
            line = await episode(c, seed)
            wins += "task success True" in line
            print(f"seed {seed}: {line}", flush=True)
        print(f"\n{wins}/{args.seeds} task successes")


asyncio.run(main())
