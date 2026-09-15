#!/usr/bin/env bash
# T2 cell, 100 half-shake two-bin demos: expert ceiling, then BC, then ACT.
# Sequential on purpose -- training is CPU-bound here, parallel runs are no faster.
# Same training configs as T1's 100-demo cells: BC 60 epochs, ACT 50 epochs.
#
#   bash experiments/t2/scripts/pipeline.sh 2>&1 | grep --line-buffered -v Svt > experiments/t2/pipeline.log
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MUJOCO_GL=glfw PYTHONUNBUFFERED=1

DATA=data/robosuite/pick_place_two_bins_noise0025_n100
REPO=dream_robot/pick_place_two_bins
ENV=robosuite/pick_place_two_bins

echo "== waiting for recording $(date +%T)"
while pgrep -f "pick_place_two_bins.record" >/dev/null; do sleep 20; done
test -f "$DATA/bin_choices.json"

echo "== expert eval $(date +%T)"
uv run python -m dream_robot.core.run --env $ENV --policy expert --experiments experiments/t2

echo "== BC train $(date +%T)"
uv run python -m dream_robot.policies.bc.train --root $DATA --repo-id $REPO \
    --out experiments/t2_bc_half100 --epochs 60
echo "== BC eval $(date +%T)"
uv run python -m dream_robot.core.run --env $ENV --policy bc \
    --checkpoint experiments/t2_bc_half100/checkpoint.pt --experiments experiments/t2_bc_half100/eval

echo "== ACT train $(date +%T)"
uv run python -m dream_robot.policies.act.train --root $DATA --repo-id $REPO \
    --out experiments/t2_act_half100 --epochs 50
echo "== ACT eval $(date +%T)"
uv run python -m dream_robot.core.run --env $ENV --policy act \
    --checkpoint experiments/t2_act_half100/checkpoint.pt --experiments experiments/t2_act_half100/eval

echo "== done $(date +%T)"
