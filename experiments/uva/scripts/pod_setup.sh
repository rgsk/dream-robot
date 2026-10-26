#!/usr/bin/env bash
# UVA on a RunPod pod: their pinned env + LIBERO-10 checkpoint + VAE (setup.sh), and our data as
# the loader's own cache (26 MB zarr from the private dataset repo; with it present the loader
# never reads the HDF5 files, so only an empty folder stands in for them).
#   local: rsync the repo + envs/uva/src to /workspace/dream-robot; HF_TOKEN set on the pod
#   pod:   bash /workspace/dream-robot/experiments/uva/scripts/pod_setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive UV_CACHE_DIR=/root/uv-cache
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd /workspace/dream-robot
bash experiments/uva/scripts/setup.sh

echo "== data $(date +%T)"
mkdir -p data/uva/pick_place_cube data/uva/t5  # empty: the cache is all the loader reads
envs/uva/.venv/bin/python -c "
from huggingface_hub import hf_hub_download
for name in ('pick_place_cube', 't5'):  # T5: 506 MB, both cameras at 224 px
    hf_hub_download('rgsk/dream-robot-data', f'uva/{name}_clip.zarr.zip', repo_type='dataset',
                    local_dir='/workspace/dream-robot/data')"
ls -la data/uva
echo "== pod setup done $(date +%T)"
