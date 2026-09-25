#!/usr/bin/env bash
# T5: fine-tune SmolVLA on the 150 clean demos, on a pod of its own.
#   local: ssh ... 'bash /workspace/setup.sh'   (once per pod: uv, env vars)
#          ssh ... 'nohup bash /workspace/dream-robot/experiments/t5/scripts/pod_smolvla.sh \
#              > /workspace/dream-robot/experiments/t5/pod_smolvla.log 2>&1 < /dev/null &'
#
# Everything heavy lives on the container disk (/root): venv, HF cache, dataset,
# checkpoints. The network volume is at its quota (24 GB of it is uv's cache), and
# it is slow for imports; only this script and its log live there. Checkpoints are
# pulled to local at the end -- /root is wiped when the pod is terminated.
# First attempt (shared pod, venv on the volume) died of "Disk quota exceeded":
# experiments/t5/notes.md, step 5.
set -euo pipefail
source /etc/profile.d/workspace.sh             # UV_PROJECT_ENVIRONMENT=/root/venv: ours, own pod
export HF_HOME=/root/.hf-cache
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /workspace/dream-robot
echo "== uv sync $(date +%T)"
uv sync --no-dev

DATA=/root/data/robosuite/instructed_pick_place_clean_n150
mkdir -p "$(dirname "$DATA")"
[ -d "$DATA" ] || cp -r data/robosuite/instructed_pick_place_clean_n150 "$DATA"

echo "== train $(date +%T)"
uv run --no-dev python -m dream_robot.policies.smolvla.train \
    --root "$DATA" --repo-id dream_robot/instructed_pick_place \
    --out /root/runs/t5/smolvla_clean_n150 \
    --batch-size 32 --workers 8 2>&1 | grep --line-buffered -v "Map:"
echo "== done $(date +%T)"
echo "------ FINISHED ------"
