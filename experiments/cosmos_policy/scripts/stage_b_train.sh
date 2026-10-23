#!/usr/bin/env bash
# Stage B on the pod: fine-tune Cosmos Policy (from the LIBERO policy) on T5. After pod_setup.sh.
#   smoke:  ITERS=20 BATCH=16 NAME=smoke bash stage_b_train.sh   (loads? memory? s/iter?)
#   run:    ITERS=2000 BATCH=16 NAME=b1 bash stage_b_train.sh "scheduler.cycle_lengths=[2000,100000000000000]"
# Output: /root/out/cosmos_policy/cosmos_policy_t5/<NAME>/checkpoints/iter_N (DCP), log on stdout.
set -euo pipefail
export PYTHONUNBUFFERED=1 HF_HOME=/root/hf UV_CACHE_DIR=/root/uv-cache
export PATH="$HOME/.local/bin:$PATH"
export T5_DATA_DIR=/root/data/cosmos_policy/t5_224 IMAGINAIRE_OUTPUT_ROOT=/root/out
export WANDB_MODE=disabled
ITERS=${ITERS:-20} BATCH=${BATCH:-16} NAME=${NAME:-smoke} SAVE=${SAVE:-500}

cd /root/cosmos-policy
uv run --extra cu128 --group libero --python 3.10 \
    torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.train \
    --config=cosmos_policy/config/config.py -- \
    experiment=cosmos_policy_t5_from_libero \
    trainer.max_iter="$ITERS" dataloader_train.batch_size="$BATCH" \
    checkpoint.save_iter="$SAVE" job.name="$NAME" "$@"
