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

## Step 2 · SmolVLA base, zero-shot (2026-09-24)
- `lerobot/smolvla_base`: 873 MB, 450M params, pretrained on SO-100 (6-dim state/action, 3 cams).
- Python 3.14 breaks draccus (LeRobot's config parser; argparse rejects its types), and no
  release fixes it. Fix: build SmolVLAConfig from config.json ourselves. `lerobot-train` CLI
  also goes through draccus, so it won't run here either.
- Base config keeps SO-100 input features unless cleared; cleared -> ours (8-dim state, top, wrist).
  State/action pad to 32 dims, so no new layers.
- Result: 0/5, all no_grasp. Arm swings up and back, away from the table, never approaches a
  cube. Plumbing works: prompt -> VLM -> 8 joints -> env.
- Cost: load 120 s (first load, cold cache), 0.86 GiB GPU, 122 env steps/s (one inference per
  50-step chunk).
- Decision: fine-tune. Pending: own training loop (stays on 3.14) vs py3.12 venv for lerobot-train.

## Step 3 · 150 clean demos (2026-09-24)
- Clean, not shaky: README T1 matrix has ACT (chunked) at 100% on 100 clean demos; shake only
  mattered for single-step BC. SmolVLA runs 50-step chunks. Shake is the follow-up if it drops cubes.
- Expert 150/150, 33,385 frames, 65 MB -> data/robosuite/instructed_pick_place_clean_n150.
- Kept per pair: red->left 32, yellow->right 33, green->left 27, blue->right 22, blue->left 20,
  red->right 16. Uniform draw, uneven at n=150.
- By construction of the held-out pairs, yellow only ever goes right and green only left in
  training. A colour->side shortcut fits the data perfectly; the held-out eval
  (yellow->left, green->right) is what catches it.

## Step 4 · own training loop, smoke (2026-09-24)
- dream_robot/policies/smolvla/train.py: lerobot-train's step, written out (build at step 0 ->
  delta_timestamps chunk dataset -> preprocessor -> policy.forward flow loss -> AdamW + cosine,
  clip 10). VLM frozen: 100M of 450M params train. Checkpoint = lerobot's pretrained_model/ layout,
  config.json written by hand (draccus again).
- Batch 32 OOMs on the 4060 (7.6 GiB, ~1.1 GiB held by VS Code + browser).
- Batch 8, 40 steps: loss 1.03 -> 0.57, val 0.64 -> 0.58, peak 2.9 GiB, ~3 it/s (val pass ~40 s).
  Checkpoint reloads into the eval wrapper; first action sits next to the current joints (sane).
- Decision: full 20k-step run on the pod (32 GB), batch 32 (~10 GiB by scaling).

## Step 5 · full fine-tune on a pod (2026-09-24)
- Try 1, shared llm pod (RTX PRO 4500): container disk 8.7 GB free < 9.2 GB venv, and
  /root/venv is llm's. Venv on the network volume -> "Disk quota exceeded" (uv cache = 24 GB of it).
  llm job untouched.
- Try 2, own pod: RTX 4090 24 GB, 30 GB container disk; everything heavy on /root.
- Batch 32, 20k steps: peak 7.2 GiB, ~3.6 it/s steady -> ~1.5-2 h (~$1.3-1.5).
  Loss 2.14 (step 50) -> 0.28 (step 250).

## Step 6 · early read, 10k checkpoint (2026-09-24)
- 20 seen-pair episodes (seeds 1000-1019), local 4060: **9/20 (45%)**; chance 25%, zero-shot 0/5.
  Failures: 9 wrong_target, 1 dropped, 1 no_grasp.
- Seed 1002 ("yellow -> right"): put the red distractor in the right bin -> wrong cube.
- These 20 seeds put the target in slot 0 17 times (200 other seeds: 45%; demos 65/150).
  Coincidence, not a bug, but 20 episodes are too few: final eval = 60 per split, by slot.
- core/run.py crashed writing results.json (provenance assumed a .pt file); fixed for dirs.
- New experiments/t5/scripts/eval_t5.py: same evaluate(), plus per-episode instruction and
  wrong_cube / wrong_bin split, success by slot and by pair.

## Step 7 · 10k checkpoint, three splits x 60 episodes (seeds 1000-1059)
| split | success | wrong_cube | wrong_bin | no_grasp/dropped | right cube when it chose one |
|---|---|---|---|---|---|
| seen pairs, template | 22/60 (37%) | 27 | 0 | 11 | 22/49 = 45% |
| held-out pairs, template | 12/60 (20%) | 36 | 0 | 12 | 12/48 = 25% |
| seen pairs, paraphrases | 9/60 (15%) | 15 | 0 | 36 | 9/24 = 38% |
- **Side word: obeyed every time** (0 wrong_bin in 180). **Colour word: ~chance** (45% right
  cube on seen pairs). At 10k it has learnt "left/right" and not "red/blue".
- Held-out pairs go left/right correctly too; they fail on the cube, like seen pairs, only more.
- Paraphrases mostly break the *motion* (35 no_grasp): unseen wordings push the action expert
  off-distribution before any grounding question arises. "move the X cube to" (closest to the
  template) 5/19; "pick up ... and drop it in ... container" 0/8.
- Slot: no strong bias (seen: slot 0 15/39, slot 1 7/21).
- Guess for colour: the cubes are ~5 px wide in the 128x128 top camera; SmolVLA compresses each
  image to 64 tokens (8x8 grid, ~16 px each), so a cube is a fraction of one token. Wait for 20k.

## Step 8 · 20k checkpoint, same three splits, same seeds
| split | 10k | 20k | right cube when it chose one (10k -> 20k) |
|---|---|---|---|
| seen pairs, template | 22/60 | 22/60 | 45% -> 42% |
| held-out pairs, template | 12/60 | 11/60 | 25% -> 21% |
| seen pairs, paraphrases | 9/60 | 10/60 | 38% -> 30% |
- wrong_bin still 0 of 180. Grasping improved (no_grasp 10 -> 6 seen, 35 -> 25 paraphrase).
- **Colour did not improve with 2x training.** Still chance. Longer training fixed the motion,
  not the grounding. Not a training-length problem.
- Training: 20k steps, 97 min on the 4090 (~$1.20), final val loss 0.047. Checkpoint pulled
  (865 MB); pod can go.
