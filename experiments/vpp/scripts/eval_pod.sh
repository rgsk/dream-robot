#!/usr/bin/env bash
# VPP2 eval on the pod: action server up (their env), T5 eval through it (ours), server down.
#   CKPT=runs/t5/b/checkpoints/weights/step_002000.pt RUN=runs/t5/b NAME=b2000 EPISODES=10 \
#       [SEED=1000] [EXECUTE=16] [STEPS=10] [PAIRS="seen held_out"] bash eval_pod.sh
# Paths relative to envs/vpp2/src. Results: /workspace/results/vpp2_<NAME>/<pairs>/.
set -euo pipefail
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PATH="$HOME/.local/bin:$PATH"
CKPT=${CKPT:?set CKPT} RUN=${RUN:?set RUN} NAME=${NAME:?set NAME} EPISODES=${EPISODES:-10}
SEED=${SEED:-1000} EXECUTE=${EXECUTE:-16} STEPS=${STEPS:-10} PORT=${PORT:-6002} PAIRS=${PAIRS:-"seen held_out"}
OUT=/workspace/results/vpp2_${NAME}
mkdir -p "$OUT"

cd /workspace/dream-robot/envs/vpp2/src
PYTHONPATH=src ../.venv/bin/python /workspace/dream-robot/experiments/vpp/scripts/serve.py \
    --ckpt "$CKPT" --stats "$RUN/dataset_stats.json" --video --steps "$STEPS" --port "$PORT" > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until grep -q "^serving" "$OUT/server.log"; do
    kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 5
done
echo "server up $(date +%T)"

cd /workspace/dream-robot
export UV_PROJECT_ENVIRONMENT=/root/venv-dream-robot UV_CACHE_DIR=/root/uv-cache
for pairs in $PAIRS; do
    uv run python experiments/t5/scripts/eval_t5.py --policy vpp2 --checkpoint "localhost:$PORT" \
        --camera-size 224 --execute "$EXECUTE" --pairs "$pairs" --episodes "$EPISODES" \
        --seed-start "$SEED" --out "$OUT/$pairs" 2>&1 | grep --line-buffered -v "robosuite"
done
echo "eval done $(date +%T)"
