#!/usr/bin/env bash
# Stage B eval on the pod, after training: checkpoint -> .pt, action server up, T5 eval through it.
#   CKPT=/root/out/cosmos_policy/cosmos_policy_t5/b1/checkpoints/iter_000002000 EPISODES=10 \
#       bash stage_b_eval.sh
# Results: /workspace/results/<ckpt name>/{seen,held_out}/ (results.json, episodes.json, videos/).
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HOME=/root/hf UV_CACHE_DIR=/root/uv-cache
export PATH="$HOME/.local/bin:$PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CKPT=${CKPT:?set CKPT} EPISODES=${EPISODES:-10} PORT=${PORT:-6000}
DATA=/root/data/cosmos_policy/t5_224
SCRIPTS=/workspace/dream-robot/experiments/cosmos_policy/scripts
OUT=/workspace/results/$(basename "$(dirname "$(dirname "$CKPT")")")_$(basename "$CKPT")
mkdir -p "$OUT"

cd /root/cosmos-policy
COSMOS="uv run --extra cu128 --group libero --python 3.10 python"
CUDA_VISIBLE_DEVICES="" $COSMOS "$SCRIPTS/to_pt.py" "$CKPT"

$COSMOS "$SCRIPTS/serve.py" --ckpt "$CKPT.pt" --stats "$DATA/dataset_statistics.json" \
    --embeddings "$DATA/t5_embeddings.pkl" --port "$PORT" > "$OUT/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until grep -q "^serving" "$OUT/server.log"; do
    kill -0 $SERVER 2>/dev/null || { echo "server died:"; tail -20 "$OUT/server.log"; exit 1; }
    sleep 5
done
echo "server up $(date +%T)"

cd /workspace/dream-robot
export UV_PROJECT_ENVIRONMENT=/root/venv-dream-robot
for pairs in seen held_out; do
    uv run python experiments/t5/scripts/eval_t5.py --policy cosmos_policy \
        --checkpoint "localhost:$PORT" --camera-size 224 --execute 16 \
        --pairs "$pairs" --episodes "$EPISODES" --out "$OUT/$pairs" 2>&1 | grep -v "robosuite"
done
echo "eval done $(date +%T)"
