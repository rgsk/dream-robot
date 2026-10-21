#!/usr/bin/env bash
# UWM eval on the pod: action server up, T5 eval through it, server down.
#   CKPT=/workspace/runs/b/model.pt NAME=b EPISODES=30 [DROP=1 | JOINT=1] [SEED=1000] [PAIRS="seen held_out"] \
#       bash eval_pod.sh
# Results: /workspace/results/uwm_<NAME>[_drop|_joint]/<pairs>/ (results.json, episodes.json, videos/).
set -euo pipefail
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PATH="$HOME/.local/bin:$PATH"
CKPT=${CKPT:?set CKPT} NAME=${NAME:?set NAME} EPISODES=${EPISODES:-10} SEED=${SEED:-1000}
DROP=${DROP:-0} JOINT=${JOINT:-0} PORT=${PORT:-6001} PAIRS=${PAIRS:-"seen held_out"}
SUFFIX=""; FLAG=""  # not $(... && ...): under set -e that exits
if [ "$DROP" = 1 ]; then SUFFIX=_drop; FLAG=--drop; fi
if [ "$JOINT" = 1 ]; then SUFFIX=_joint; FLAG=--joint; fi
OUT=/workspace/results/uwm_${NAME}${SUFFIX}
mkdir -p "$OUT"
cd /workspace/dream-robot

PYTHONPATH=envs/uwm/src:experiments/uwm/scripts envs/uwm/.venv/bin/python \
    experiments/uwm/scripts/serve.py --ckpt "$CKPT" --port "$PORT" $FLAG \
    > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until grep -q "^serving" "$OUT/server.log"; do
    kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 3
done
echo "server up $(date +%T)"

export UV_PROJECT_ENVIRONMENT=/root/venv-dream-robot UV_CACHE_DIR=/root/uv-cache
for pairs in $PAIRS; do
    uv run python experiments/t5/scripts/eval_t5.py --policy uwm --checkpoint "localhost:$PORT" \
        --camera-size 224 --pairs "$pairs" --episodes "$EPISODES" --seed-start "$SEED" \
        --out "$OUT/$pairs$([ "$SEED" = 1000 ] || echo "_s$SEED")" 2>&1 | grep --line-buffered -v "robosuite"
done
echo "eval done $(date +%T)"
