#!/usr/bin/env bash
# One T7 episode driven by a fresh headless Claude Code session (billed to the subscription):
# only the t7 MCP server, no built-in tools, a one-line system prompt.
#   experiments/agent/scripts/headless_drive.sh SEED [TAG]
# Writes scratch/agent/headless/<seed>[-TAG].jsonl (raw stream) and .log (readable; tail -f it).
# MCP_CONFIG=<file> picks another server config (default .mcp.json), e.g. a tool hidden;
# SERVER=<name> one of its servers (default the first). Only that server is started.
# Cost per call: python experiments/agent/scripts/token_cost.py <session transcript> --from-reset SEED
set -euo pipefail
seed=${1:?seed}
name=$seed${2:+-$2}
src=${MCP_CONFIG:-.mcp.json}
server=${SERVER:-$(python3 -c 'import json, sys; print(next(iter(json.load(open(sys.argv[1]))["mcpServers"])))' "$src")}
cfg=scratch/agent/headless/mcp-$server.json
mkdir -p scratch/agent/headless
python3 -c 'import json, sys; d = json.load(open(sys.argv[1]))["mcpServers"]; json.dump({"mcpServers": {sys.argv[2]: d[sys.argv[2]]}}, open(sys.argv[3], "w"), indent=2)' "$src" "$server" "$cfg"
out=scratch/agent/headless
mkdir -p "$out"

claude -p "Call reset with seed $seed. Do what the instruction says. Finish with your verdict \
for each object, judged from the images." \
  --strict-mcp-config --mcp-config "$cfg" \
  --tools "" \
  --allowedTools "mcp__$server" \
  --system-prompt "You drive a robot arm through the $server tools." \
  --output-format stream-json --verbose \
  | tee "$out/$name.jsonl" \
  | python3 -u experiments/agent/scripts/stream_log.py > "$out/$name.log"
