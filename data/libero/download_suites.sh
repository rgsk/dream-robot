#!/bin/bash
# LIBERO suites (original hdf5 with sim states), Long first. Skips files already present.
cd "$(dirname "$0")/../.."
for s in libero_10 libero_spatial libero_object libero_goal; do
  envs/libero/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('yifengzhu-hf/LIBERO-datasets', repo_type='dataset', allow_patterns='$s/*', local_dir='data/libero')
print('done $s', flush=True)"
done
echo ALL_DONE
