#!/usr/bin/env bash
# T5 step 9: flow matching told the target as a one-hot, then seen + held-out evals.
set -euo pipefail
export PYTHONUNBUFFERED=1
OUT=experiments/t5/fm_onehot_n150
uv run python -m dream_robot.policies.flow_matching.train \
    --root data/robosuite/instructed_pick_place_clean_n150 \
    --repo-id dream_robot/instructed_pick_place \
    --out "$OUT" --condition-words red green blue yellow left right
for pairs in seen held_out; do
  MUJOCO_GL=egl uv run python experiments/t5/scripts/eval_t5.py \
      --policy flow_matching --checkpoint "$OUT/checkpoint.pt" \
      --pairs "$pairs" --wording template --episodes 60 --out "experiments/t5/eval_fm_onehot/$pairs"
done
echo "------ FINISHED ------"
