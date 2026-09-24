#!/usr/bin/env bash
# T2-fixed flow matching on RunPod: train 100 epochs from pod-local disk.
#   on the pod: nohup bash -lc "bash /workspace/dream-robot/experiments/t2_fm/scripts/pod_fixed_fm.sh" > /workspace/dream-robot/experiments/t2_fm/pod_fixed_fm.log 2>&1 &
set -euo pipefail
cd /workspace/dream-robot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
# Own venv: the pod is shared with llm, whose jobs use /root/venv (workspace.sh default).
export UV_PROJECT_ENVIRONMENT=/root/venv-dream-robot

DATA=data/robosuite/pick_place_two_bins_fixed_noise0025_n100
LOCAL=/root/$DATA
OUT=experiments/t2_fixed_fm_half100

echo "== copy dataset to local disk $(date +%T)"
mkdir -p "$(dirname $LOCAL)" && rm -rf $LOCAL && cp -r $DATA $LOCAL

echo "== train $(date +%T)"
uv run python -m dream_robot.policies.flow_matching.train --root $LOCAL \
    --repo-id dream_robot/pick_place_two_bins --out $OUT --workers 5 2>&1 \
    | grep --line-buffered -v -i "svt\|warn"

echo "== done $(date +%T)"
echo "------ FINISHED ------"
