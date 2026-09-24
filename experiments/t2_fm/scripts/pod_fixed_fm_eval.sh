#!/usr/bin/env bash
# T2-fixed: sampling test (one start repeated, policy seed varied) for FM / BC / ACT, then FM standard eval.
#   on the pod: nohup bash -lc "bash /workspace/dream-robot/experiments/t2_fm/scripts/pod_fixed_fm_eval.sh" > /workspace/dream-robot/experiments/t2_fm/pod_fixed_fm_eval.log 2>&1 &
set -euo pipefail
cd /workspace/dream-robot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
# Own venv: the pod is shared with llm, whose jobs use /root/venv (workspace.sh default).
export UV_PROJECT_ENVIRONMENT=/root/venv-dream-robot

ENV=robosuite/pick_place_two_bins_fixed
S=experiments/t2/scripts/rollout_paths.py
P=experiments/t2_fm/paths
declare -A CKPT=(
  [flow_matching]=experiments/t2_fixed_fm_half100/checkpoint.pt
  [bc]=experiments/t2_fixed_bc_half100/checkpoint.pt
  [act]=experiments/t2_fixed_act_half100/checkpoint.pt
)
quiet() { grep --line-buffered -v -i "warn\|info\|svt"; }

for policy in flow_matching bc act; do
  for start in 1000 1010 1013; do
    echo "== sampling $policy start $start $(date +%T)"
    uv run python $S run --env $ENV --policy $policy --checkpoint ${CKPT[$policy]} \
        --repeat-seed $start --out $P/${policy}_repeat$start.json 2>&1 | quiet
  done
done

echo "== FM standard eval, half shake $(date +%T)"
uv run python -m dream_robot.core.run --env $ENV --policy flow_matching \
    --checkpoint ${CKPT[flow_matching]} --noise-sigma 0.025 \
    --experiments experiments/t2_fixed_fm_half100/eval 2>&1 | quiet
uv run python $S run --env $ENV --policy flow_matching --checkpoint ${CKPT[flow_matching]} \
    --noise-sigma 0.025 --out $P/flow_matching_shake.json 2>&1 | quiet | tail -12

echo "== done $(date +%T)"
echo "------ FINISHED ------"
