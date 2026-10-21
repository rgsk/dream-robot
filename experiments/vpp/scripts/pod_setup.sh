#!/usr/bin/env bash
# VPP2 on a RunPod pod (80 GB GPU): their code, torch cu128 (their requirements allow torch>=2.5;
# tested 2.11+cu130 would need a CUDA 13 driver), the LIBERO weights (~51 GB) and our T5 data.
# Plain PyTorch attention throughout: no flash-attn build.
#   local: rsync the repo + envs/vpp2/src to /workspace/dream-robot; HF_TOKEN set on the pod
#   pod:   bash /workspace/dream-robot/experiments/vpp/scripts/pod_setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive UV_CACHE_DIR=/root/uv-cache HF_HUB_ENABLE_HF_TRANSFER=1
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/dream-robot/envs/vpp2
echo "== env $(date +%T)"
[ -d .venv ] || uv venv -q --python 3.10 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -q torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
VIRTUAL_ENV=$PWD/.venv uv pip install -q -r src/requirements.txt huggingface_hub hf_transfer
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.cuda.get_device_name(0))"

echo "== weights $(date +%T)"  # their README: Video-10k + Action-30k + shared Wan encoders
cd src
hfpy() { ../.venv/bin/python -c "$1"; }
hfpy "
from huggingface_hub import snapshot_download
snapshot_download('Haodong082399/VPP2', local_dir='weights', max_workers=16,
                  allow_patterns=['config.json', 'checkpoints/libero/*', 'checkpoints/Wan2.1-I2V-14B-480P/**'])"
mkdir -p weights/libero
ln -sfn ../checkpoints/libero/video_step010000.pt weights/libero/video_step010000.pt
ln -sfn ../checkpoints/libero/action_step030000.pt weights/libero/action_step030000.pt
ln -sfn checkpoints/Wan2.1-I2V-14B-480P weights/Wan2.1-I2V-14B-480P
du -sh weights/checkpoints/*

echo "== data $(date +%T)"  # our T5 demos (private dataset repo)
hfpy "
from huggingface_hub import snapshot_download
snapshot_download('rgsk/dream-robot-data', repo_type='dataset', allow_patterns=['t5_224/*'],
                  local_dir='/workspace/dream-robot/data/cosmos_policy')"
ls /workspace/dream-robot/data/cosmos_policy/t5_224
echo "== setup done $(date +%T)"
