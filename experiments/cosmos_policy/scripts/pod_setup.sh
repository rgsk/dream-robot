#!/usr/bin/env bash
# Cosmos Policy on a RunPod pod, no Docker: their image is CUDA 12.8 + a few apt packages + uv,
# which the "RunPod PyTorch" template (CUDA >= 12.8) already is, minus the packages.
# Everything on the pod's own disk (/root): no network volume needed.
# The HOST driver must do CUDA >= 12.8 (their torch is cu128 / cu130): create the pod with
# --min-cuda-version 12.8. A 550 driver (CUDA 12.4) fails with 'CUDA unknown error'.
#   local:  rsync the repo (runpod.md step 2) + data/cosmos_policy/t5_224 to /root/data/...
#   pod:    bash /workspace/dream-robot/experiments/cosmos_policy/scripts/pod_setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
export HF_HOME=/root/hf UV_CACHE_DIR=/root/uv-cache

apt-get update -qq
apt-get install -y -qq --no-install-recommends build-essential cmake ffmpeg git git-lfs \
    libgl1 libegl1-mesa-dev libgl1-mesa-dri libglib2.0-0 > /dev/null
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /root
[ -d cosmos-policy ] || git clone -q --depth 1 https://github.com/NVlabs/cosmos-policy
cd cosmos-policy
echo "== uv sync $(date +%T)"
uv sync --extra cu128 --group libero --python 3.10 2>&1 | tail -3
# LIBERO asks where to keep its datasets on first import; answer once with the default.
echo n | uv run --extra cu128 --group libero --python 3.10 python -c "import libero.libero" > /dev/null
uv run --extra cu128 --group libero --python 3.10 python -c \
    "import torch; assert torch.cuda.is_available(), 'no CUDA: host driver too old?'; print('cuda ok')"
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "== setup done $(date +%T)"
