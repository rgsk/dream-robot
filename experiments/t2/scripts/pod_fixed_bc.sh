#!/usr/bin/env bash
# T2-fixed on RunPod: record 100 half-shake demos with a fixed cube pose and an exactly 50/50 coin,
# train BC (60 epochs, as every 100-demo BC cell), eval with half shake, path diagnostic.
#   on the pod: nohup bash -lc "bash /workspace/dream-robot/experiments/t2/scripts/pod_fixed_bc.sh" > /workspace/dream-robot/experiments/t2/pod_fixed_bc.log 2>&1 &
set -euo pipefail
cd /workspace/dream-robot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1

DATA=data/robosuite/pick_place_two_bins_fixed_noise0025_n100
LOCAL=/root/$DATA                       # train from pod-local disk, not the network volume
REPO=dream_robot/pick_place_two_bins
ENV=robosuite/pick_place_two_bins_fixed
BC=experiments/t2_fixed_bc_half100

# Record onto pod-local disk. The recorder writes one PNG per frame per camera before encoding,
# and the network volume is slow on many small files: recording straight onto it ran at ~21 s
# per episode (first attempt, killed) vs ~2.5 s on the local machine.
echo "== record to local disk $(date +%T)"
rm -rf $LOCAL $DATA && mkdir -p "$(dirname $LOCAL)"
uv run python -m dream_robot.sims.robosuite.tasks.pick_place_two_bins.record \
    --episodes 100 --noise-sigma 0.025 --fixed-cube --coin balanced --root $LOCAL \
    --verify --verify-out experiments/t2/videos/fixed_dataset_roundtrip.mp4 2>&1 \
    | grep --line-buffered -v "Svt\|WARNING\|INFO\|Map:\|moov"

echo "== copy dataset to the volume $(date +%T)"
mkdir -p "$(dirname $DATA)" && cp -r $LOCAL $DATA

echo "== expert eval, half shake $(date +%T)"
uv run python -m dream_robot.core.run --env $ENV --policy expert --noise-sigma 0.025 \
    --experiments experiments/t2_fixed 2>&1 | grep --line-buffered -v "Svt\|WARNING\|INFO"

echo "== BC train $(date +%T)"
uv run python -m dream_robot.policies.bc.train --root $LOCAL --repo-id $REPO \
    --out $BC --epochs 60 --workers 16 2>&1 | grep --line-buffered -v Svt

echo "== BC eval, half shake $(date +%T)"
uv run python -m dream_robot.core.run --env $ENV --policy bc --checkpoint $BC/checkpoint.pt \
    --noise-sigma 0.025 --experiments $BC/eval 2>&1 | grep --line-buffered -v "Svt\|WARNING\|INFO"

echo "== paths $(date +%T)"
S=experiments/t2/scripts/rollout_paths.py
uv run python $S run --env $ENV --policy expert --noise-sigma 0.025 \
    --out experiments/t2/paths/fixed_expert.json 2>&1 | grep --line-buffered -v "WARNING\|INFO" | tail -12
uv run python $S run --env $ENV --policy bc --checkpoint $BC/checkpoint.pt --noise-sigma 0.025 \
    --out experiments/t2/paths/fixed_bc.json 2>&1 | grep --line-buffered -v "WARNING\|INFO" | tail -12
uv run python $S plot experiments/t2/paths/{fixed_expert,fixed_bc}.json \
    --out experiments/t2/paths/fixed_paths.png

echo "== done $(date +%T)"
