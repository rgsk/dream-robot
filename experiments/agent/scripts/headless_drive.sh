#!/usr/bin/env bash
# One T7 episode driven by a fresh headless Claude Code session (billed to the subscription):
# only the t7 MCP server, no built-in tools, a one-line system prompt.
#   experiments/agent/scripts/headless_drive.sh SEED [TAG]
# Writes scratch/agent/headless/<seed>[-TAG].jsonl (raw stream) and .log (readable; tail -f it).
# MCP_CONFIG=<file> picks another server config (default .mcp.json), e.g. a tool hidden.
# Cost per call: python experiments/agent/scripts/token_cost.py <session transcript> --from-reset SEED
set -euo pipefail
seed=${1:?seed}
name=$seed${2:+-$2}
out=scratch/agent/headless
mkdir -p "$out"

claude -p "Call reset with seed $seed. Do what the instruction says. Finish with your verdict \
for each object, judged from the images." \
  --strict-mcp-config --mcp-config "${MCP_CONFIG:-.mcp.json}" \
  --tools "" \
  --allowedTools "mcp__t7-pick-place-objects" \
  --system-prompt "You drive a robot arm through the t7-pick-place-objects tools." \
  --output-format stream-json --verbose \
  | tee "$out/$name.jsonl" \
  | python3 -u experiments/agent/scripts/stream_log.py > "$out/$name.log"
