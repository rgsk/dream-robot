#!/usr/bin/env bash
# RunPod setup + smoke test for dream-robot. No real training: one epoch each, a 2-episode eval.
# Run on the pod (after rsync):  bash -lc "bash /workspace/dream-robot/experiments/t2/scripts/pod_smoke.sh"
set -euo pipefail
cd /workspace/dream-robot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1

DATA=data/robosuite/pick_place_two_bins_noise0025_n100
REPO=dream_robot/pick_place_two_bins
OUT=/root/pod_smoke          # pod-local disk, thrown away with the pod

echo "== uv sync $(date +%T)"
time uv sync 2>&1 | tail -3

echo "== GPU / torch $(date +%T)"
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

echo "== T2 sim tests, headless EGL $(date +%T)"
time uv run pytest -q tests/test_pick_place_two_bins.py 2>&1 | tail -3

echo "== BC, 1 epoch $(date +%T)"
time uv run python -m dream_robot.policies.bc.train --root $DATA --repo-id $REPO \
    --out $OUT/bc --epochs 1 --workers 16 2>&1 | command grep -v Svt | tail -4

echo "== ACT, 1 epoch $(date +%T)"
time uv run python -m dream_robot.policies.act.train --root $DATA --repo-id $REPO \
    --out $OUT/act --epochs 1 --workers 16 2>&1 | command grep -v Svt | tail -4

echo "== eval BC, 2 episodes with video $(date +%T)"
time uv run python -m dream_robot.core.run --env robosuite/pick_place_two_bins --policy bc \
    --checkpoint $OUT/bc/checkpoint.pt --episodes 2 --experiments $OUT/eval 2>&1 | command grep -v "Svt\|WARNING\|INFO" | tail -5
ls -la $OUT/eval/*/videos/

echo "== done $(date +%T)"
