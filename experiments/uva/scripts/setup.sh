#!/usr/bin/env bash
# UVA local env (their conda pins as pip: Python 3.9, torch 2.1.0 cu121) + LIBERO-10 checkpoint + VAE / MAR-base.
#   bash experiments/uva/scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../../../envs/uva"

echo "== downloads start in background $(date +%T)"
mkdir -p ckpt
(
  [ -d .venv-dl ] || { uv venv -q --python 3.10 .venv-dl && VIRTUAL_ENV=$PWD/.venv-dl uv pip install -q gdown requests; }
  [ -f ckpt/libero10.ckpt ] || .venv-dl/bin/gdown -q 11c2VrmaRp48yw__5A5xpcu8EPzkexHSi -O ckpt/libero10.ckpt
  (cd src && ../.venv-dl/bin/python unified_video_action/utils/download.py)
  echo "== downloads done $(date +%T)"; ls -la ckpt src/pretrained_models/*/ src/pretrained_models/mar/*/
) &
dl=$!

echo "== env $(date +%T)"
[ -d .venv ] || uv venv -q --python 3.9 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -q \
  --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple \
  torch==2.1.0 torchvision==0.16.0 numpy==1.26.4 scipy==1.13.1 numba==0.60.0 pandas==2.1.4 \
  matplotlib==3.7.5 av==12.3.0 pyyaml==6.0.1 tqdm==4.65.0 zarr==2.16.1 numcodecs==0.11.0 \
  hydra-core==1.2.0 dill==0.3.7 einops==0.6.1 diffusers==0.18.2 timm==0.9.7 accelerate==1.0.1 \
  click gdown==5.2.0 imagecodecs==2024.12.30 kornia==0.8.0 opencv-python==4.11.0.86 \
  pygame==2.6.1 pymunk==6.11.1 scikit-image==0.24.0 shapely==2.0.7 \
  transformers==4.28.0 huggingface_hub==0.25.2 wandb==0.18.3 tensorboard==2.19.0 h5py filelock threadpoolctl ipykernel
VIRTUAL_ENV=$PWD/.venv uv pip install -q -e src --no-deps
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('cuda ok', torch.__version__, torch.cuda.get_device_name(0))"
echo "== env done $(date +%T)"

wait $dl
echo "== setup done $(date +%T)"

# pytorch3d stand-in (rotation conversions, unused with joint actions) on the env's path
echo "$PWD/../../experiments/uva/scripts/stubs" > .venv/lib/python3.9/site-packages/uva_stubs.pth
