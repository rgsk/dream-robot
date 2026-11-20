#!/bin/bash
# Evaluate a training run's snapshots as they appear (2026-11-20): rollout.py on ckpt/weights_step<N>.pt for each N in
# STEPS, then weights.pt; EPISODES per task. One summary line per snapshot in RUN_DIR/eval_progress.log.
#   experiments/libero/scripts/eval_watch.sh RUN_DIR SUITE [EPISODES] [STEPS...]
RUN=$1; SUITE=$2; EPS=${3:-10}; shift 3; STEPS=${@:-4000 8000 12000 16000}
cd "$(dirname "$0")/../../.."
for w in $(for s in $STEPS; do echo ckpt/weights_step$s.pt; done) weights.pt; do
  until [ -f "$RUN/$w" ]; do sleep 20; done
  sleep 5  # let the save finish
  tag=$(basename $w .pt)
  PYTHONUNBUFFERED=1 VERNE_LIBERO_SUITE=$SUITE MUJOCO_GL=egl envs/libero/.venv/bin/python \
    experiments/libero/scripts/rollout.py "$RUN" "$w" "$EPS" 1 > "$RUN/rollout_$tag.log" 2>&1
  echo "$(date +%T) $tag $(grep -a '^total' "$RUN/rollout_$tag.log")" >> "$RUN/eval_progress.log"
done
echo "$(date +%T) DONE" >> "$RUN/eval_progress.log"
