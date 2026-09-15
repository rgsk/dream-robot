#!/usr/bin/env bash
# T2 step after pipeline.sh: gripper paths for BC, ACT (z = 0), ACT (z sampled), and the side-by-side plot.
#
#   bash experiments/t2/scripts/diagnose.sh 2>&1 | command grep --line-buffered -v "Svt\|WARNING\|INFO" > experiments/t2/diagnose.log
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MUJOCO_GL=glfw PYTHONUNBUFFERED=1

echo "== waiting for pipeline $(date +%T)"
while pgrep -f "t2/scripts/pipeline.sh" >/dev/null; do sleep 30; done
test -f experiments/t2_act_half100/checkpoint.pt

S=experiments/t2/scripts/rollout_paths.py
echo "== BC paths $(date +%T)"
uv run python $S run --policy bc --checkpoint experiments/t2_bc_half100/checkpoint.pt \
    --out experiments/t2/paths/bc.json
echo "== ACT paths, z = 0 $(date +%T)"
uv run python $S run --policy act --checkpoint experiments/t2_act_half100/checkpoint.pt \
    --out experiments/t2/paths/act.json
echo "== ACT paths, z sampled $(date +%T)"
uv run python $S run --policy act --checkpoint experiments/t2_act_half100/checkpoint.pt \
    --sample-z --out experiments/t2/paths/act_sampled_z.json

uv run python $S plot experiments/t2/paths/{expert,bc,act,act_sampled_z}.json \
    --out scratch/t2/paths_all.png
echo "== done $(date +%T)"
