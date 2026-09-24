# T5 -- "put the {color} cube in the {side} box"

Plan: T5 env -> SmolVLA zero-shot (plumbing only, expect ~0%) -> SmolVLA fine-tune -> eval on
seen pairs / held-out pairs (yellow->left, green->right) / eval-only paraphrases.
Chance for a policy that ignores the words: 25%.

## Step 1 · env, expert, per-episode recorder (2026-09-24)
- Scene: T2's bins (grey) + 2 cubes at x = ±0.07, colours drawn from {red, green, blue, yellow}
  per episode, slot order random. "left" = image-left in top and front cameras (robot's right).
- Policy sees top + wrist (128²). Front camera is render-only.
- core/record.py: `task_prompt` may be a callable, asked after each reset; T1/T2/T6 unchanged,
  30/30 recorder tests pass.
- Smoke record, 6 eps -> scratch/t5/smoke: expert 6/6, prompt round-trips through the dataset,
  distractor untouched in ep 0.
- Decision: go to SmolVLA zero-shot next.
