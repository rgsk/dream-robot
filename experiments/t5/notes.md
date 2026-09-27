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

## Step 9 · one-hot flow-matching baseline, code (2026-09-26)
- Question: can a policy see the colour at all in 128x128, given the target as an index?
- flow_matching: optional `--condition-words red green blue yellow left right`; each prompt ->
  colour one-hot + side one-hot, joined to image+state features. No words = old net, old
  checkpoints load. Checked on the 150 demos: 6 distinct vectors = the 6 seen pairs.
- eval_t5.py: `--policy flow_matching`. Paraphrase split is meaningless here (same words
  -> same vector); eval seen + held-out only.
- Caveat: the image encoder is BC's spatial softmax on raw 128px, not SigLIP's 16px patches.
  One-hot at chance -> the pixels are the problem. One-hot works -> SmolVLA fails at
  language *or* at its own coarse vision; not yet which.

## Step 10 · one-hot flow matching, result (2026-09-26)
- Train: 100 epochs, 29 min local 4060, sampled first-action 3.5 mrad (5.5M params).
| split | one-hot FM | SmolVLA 20k | right cube when it chose one (FM / SmolVLA) |
|---|---|---|---|
| seen pairs | 19/60 | 22/60 | 49% / 42% |
| held-out pairs | 13/60 | 11/60 | 36% / 21% |
- wrong_bin 0 of 120 again. no_grasp 20 seen, 22 held-out (worse than SmolVLA's 6).
- **Handed the colour as an index, still chance on the cube.** Not a language problem, at
  least not first: neither policy gets colour out of these images.
- Open: pixels (5 px cubes) vs encoder (spatial softmax keeps *where*, drops *what*).
- Looked at the policy's own input frames (scratch/t5/policy_view/): cubes are ~10 px in top,
  ~20-25 px in wrist, saturated, unmistakable. The "5 px" guess was wrong. **The colour is in
  the pixels**; the failure is in what the networks do with them.

## Step 11 · colour probe on top frames (2026-09-26)
- 3600 top frames at reset (seeds 50000+, arm clear), 3000 train / 600 test; "which slot holds
  colour c?" over the two present colours, chance 50%. Script: scripts/colour_probe.py.
| probe | held-out |
|---|---|
| A: plain CNN + global avg pool | **100%** (99.7% by epoch 10) |
| B: policy's CameraEncoder (spatial softmax) + one-hot after it | **52%** |
- **Pixels carry the colour, completely.** The 128x128 top camera is not the problem.
- **Our encoder loses it.** B does not even fit its training set (51%, loss flat at 0.69):
  keypoint (x, y) coordinates plus a one-hot joined afterwards never learnt to tell red from green.
  Matches the policy: side (a position) perfect, colour (an identity) at chance.
- SmolVLA's failure is its own (SigLIP sees these pixels too); this probe does not explain it.

## Step 12 · probe C: one-hot inside the encoder (FiLM) (2026-09-26)
- Same encoder as B; after every GroupNorm the one-hot sets per-channel scale + shift.
- **C: 100% held-out.** Stuck at 51% for the first ~5 epochs, then escapes: 77% at 10, 95% at 15,
  100% at 20. B never left 51% in 30 epochs.
- Reading: the colour has to reach the convs, not just the head. The stall before escaping says
  it is a hard start even with FiLM; B might escape too given longer (one seed each, not tested).
- Decision pending: FiLM the condition into flow matching's encoders, retrain, rerun T5 evals.

## Step 13 · flow matching with FiLM, result (2026-09-26)
- `--film`: one-hot scales/shifts every conv block of top + wrist encoders (+ still joined after).
  100 epochs, 33 min, sampled first-action 3.51 mrad (same as without FiLM).
| split | FiLM | one-hot after (step 10) | right cube when it chose one (FiLM / after) |
|---|---|---|---|
| seen pairs | 16/60 | 19/60 | 16/27 = 59% / 49% |
| held-out pairs | 5/60 | 13/60 | 6/21 = 29% / 36% |
- no_grasp up: 20 -> 28 seen, 22 -> 37 held-out. Colour not clearly better (27 and 21 choices:
  too few to separate 59% from 50%).
- **The probe's 100% did not carry over.** Worse overall; the change mostly hurt grasping.
- Not yet known why: stuck before choosing a cube, or bad motion. Watch failures before guessing.
- Watched: both filmed failures are wrong_cube (asked yellow, went straight to red, no
  hesitation). The no_grasp episodes are not in the video.
- **Next:** condition-swap check -- same start frame, colour one-hot swapped, does the sampled
  plan head to the other cube? Answers "ignores the colour input" vs "reads it but badly".
  Then film no_grasp seeds (film_policy_view.py) to see why grasping got worse.
