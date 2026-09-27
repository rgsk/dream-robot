#!/usr/bin/env bash
# T5 step 13: flow matching, one-hot FiLMed into the encoders, then seen + held-out evals.
set -euo pipefail
export PYTHONUNBUFFERED=1
OUT=experiments/t5/fm_film_n150
uv run python -m dream_robot.policies.flow_matching.train \
    --root data/robosuite/instructed_pick_place_clean_n150 \
    --repo-id dream_robot/instructed_pick_place \
    --out "$OUT" --condition-words red green blue yellow left right --film
for pairs in seen held_out; do
  MUJOCO_GL=egl uv run python experiments/t5/scripts/eval_t5.py \
      --policy flow_matching --checkpoint "$OUT/checkpoint.pt" \
      --pairs "$pairs" --wording template --episodes 60 --out "experiments/t5/eval_fm_film/$pairs"
done
echo "------ FINISHED ------"
