#!/usr/bin/env bash
# UWM on a RunPod pod: their pinned env (Python 3.10, torch 2.2.2 cu121) + the LIBERO-90 checkpoint.
# Host driver must do CUDA >= 12.1 (their torch); the runpod/pytorch cu128 image needs >= 12.8.
#   local: rsync the repo + envs/uwm/src to /workspace/dream-robot, data/cosmos_policy/t5_224 too
#   pod:   bash /workspace/dream-robot/experiments/uwm/scripts/pod_setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive UV_CACHE_DIR=/root/uv-cache
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/dream-robot/envs/uwm
echo "== env $(date +%T)"
[ -d .venv ] || uv venv -q --python 3.10 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -q -r src/requirements.txt gdown
VIRTUAL_ENV=$PWD/.venv uv pip install -q -e src --no-deps
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(0))"

echo "== checkpoint $(date +%T)"
mkdir -p ckpt
[ -f ckpt/libero_90.pt ] || .venv/bin/gdown -q 1VfdOL8SgCl2hFwQjnr-4KwqVzB-S6La7 -O ckpt/libero_90.pt
ls -la ckpt
echo "== setup done $(date +%T)"
