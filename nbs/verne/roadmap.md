# Verne, small: roadmap

Step 0 (`dream_robot/policies/verne/model.py`) is Verne V0.5's shape with every box at its dumbest.
Each item below is one box: what step 0 does, what Verne does, and why it might matter.
Boxes open in the order data flows through Verne ([architecture.md](architecture.md)), one step
each, with a results notebook and an explained notebook (`NN_<box>.ipynb`, `NN_<box>_explained.ipynb`).
Scored on T1, 10 labelled demos, rollouts seeds 1000-1019 at n_exec 8 and 16 (step 0: 17/40).

## Boxes to open

### 01 · Camera canvas (input) — **DONE**

- step 0: the top camera only, 128×128 → 64 patches of 16×16.
- Verne (architecture §1): each camera resized, placed side by side as one wide image, so the
  video model sees one picture per moment.
- why: the wrist view shows the cube big just before the grasp, so the model can correct
  in closed loop.
- test: [top | wrist] as one 128×256 canvas → 128 patches, same patch embedding, positions over
  the whole canvas.

#### Results: 01 · Camera canvas

From `01_canvas.ipynb` (commit e16af3e). Successes out of 150 (3 training seeds × 50 episodes).

| T1 | n_exec 8 | n_exec 16 |
|---|---|---|
| step 0 | 56 | 54 |
| step 01 | 59 | 48 |

| T1-wide | n_exec 8 | n_exec 16 |
|---|---|---|
| step 0 | 4 | 3 |
| step 01 | 18 | 21 |

- Only 1 of the 3 runs finds the cube (probe x slope +0.61).
- Which run finds it depends on the GPU it trained on.

---

### 02 · VAE latents (input) — **DONE** on T1-wide

- step 0: raw pixel patches, one linear layer each.
- Verne (§2-3): a frozen VAE turns each frame into a latent grid; a small conv patches the grid.
- why: the latent keeps edges, objects and colours and drops pixel detail. In WAMz, moving
  from pixels to latents went 24 → 34/40.
- test: the frozen VAE from the dreamzero notebook (128×128 → 16×16×4), patched to tokens.
- later: Verne's own VAE shape (16× down, 48 channels, attention, average-pool shortcuts) or the
  pretrained Wan2.2 encoder; T1 (10 demos).

#### Results: 02 · VAE latents

From `03_vae_latents.ipynb` (step 03). A small VAE (`vae.py`, 1.9M params) trained on both tasks'
frames: canvas 3 × 128 × 256 → latent 4 × 16 × 32, val reconstruction 0.0015; the policy reads
2 × 2 latent squares (128 tokens). Successes out of 150 (3 training seeds × 50 episodes), L40S:

| T1-wide | n_exec 8 | n_exec 16 |
|---|---|---|
| step 02, pixels | 29 | 14 |
| step 03, VAE latents | 112 | 112 |

- Every seed finds the cube (probe x slope +0.95 to +0.99, miss 1.1-1.4 cm); in step 02 only 1 of 3.
- Seeds agree: 34 / 36 / 42 at n_exec 8.
- What's left is the grasp: most failures are `no_grasp`.
- T1 not run.

---

### 03 · Memory (input)

- step 0: only the current frame.
- Verne (`memory.py`, `mot.py` `joint_attention_mask`): past frames packed by a schedule of
  (frames, time kernel, space kernel), newest first; recent frames fine, older ones merged up to
  4× in time and 4× in space; one learned attention bias per slot on its key columns; slot i
  reads slots 0..i, the present frame reads memory + itself, actions read everything.
  The test config has 16 past frames, 8 video frames apart, schedule
  [2,1,2] [2,1,4] [4,2,4] [8,4,4]; the checkpoint's own config.json isn't in the repo.
- why: remembering "I already tried and the cube is still there" (the WAMz 1013 failure).
- test: a few past frames as extra tokens with a causal mask; then the coarse packing.

---

### 04 · Time embedding + adaptive norm (action side)

- step 0: raw τ (one number, 0..1) → MLP (1 → 128 → 128), added to every action token.
- Verne (`action_dit.py`): τ → sinusoid features (cos/sin at many frequencies) → MLP → 6 vectors
  per layer (shift, scale, gate for attention and MLP: adaptive layer norm).
- why: sinusoids let the network tell nearby noise levels apart (0.90 vs 0.95) and act
  differently at each; raw τ through one SiLU only gives smooth, gentle functions of τ. Step 0
  gets away with it because it samples just 10 evenly spaced levels.
- test: swap in sinusoids (+ adaptive norm), compare held-out loss and success.

---

### 05 · Sampler (inference) — **DONE**

- step 0: 10 evenly spaced Euler steps.
- Verne (`sampler.py`): Euler steps on σ = s·u / (1 + (s−1)·u), u even from 1 to 0, so with
  s > 1 more of the steps sit at high noise; the test config uses 4 steps, s = 1.5.
- why: fewer steps = faster plans; the shift puts the steps where the hard choices are made.
- test: 4 shifted steps vs 10 even, same weights.

#### Results: 05 · Sampler

From `04_sampler.ipynb` (step 04). Step 03's three T1-wide policies, unchanged, local RTX 4060.
Successes out of 150 (3 training seeds × 50 episodes), and episodes won / lost against
10 even steps on the same 300:

| sampler | n_exec 8 | n_exec 16 | won / lost |
|---|---|---|---|
| 10 even (steps 0-03) | 112 | 112 | — |
| 4 even | 118 | 120 | 17 / 3 |
| 4, shift 1.5 (Verne) | 120 | 122 | 23 / 5 |
| 10, shift 1.5 | 110 | 114 | 6 / 6 |

- Fewer steps is the gain; the shift alone does nothing.
- (4, 1.5) is now the package default: 2.5× fewer action-side passes per replan.

---

### 06 · Action format (output)

- step 0: absolute joint positions + gripper, 16 steps.
- Verne (§10, §12): gripper pose relative to the current hand (xyz + 6-number rotation + jaw),
  plus progress 0..1; IK outside the model turns poses into joints.
- why: "move 2 cm down" looks the same anywhere; robot-agnostic; progress says when a step is done.
- test: relative end-effector targets + IK, progress head; compare success.

---

### 07 · Future-video loss (training only)

- step 0: actions are the only target.
- Verne (training, inferred): some future frames are added as noisy tokens and the video side
  learns to denoise them; at run time that branch is never used.
- why: predicting the future forces the video side to understand dynamics (WAMz's gain).
- test: noisy future-frame tokens + a frame loss in training; run time unchanged.

---

### 08 · RoPE positions (both transformers)

- step 0/01: learned absolute positions, one vector per token added to its input
  (`video_pos` over state + canvas patches, `action_pos` over the chunk).
- Verne (`wan_dit.py` `frame_rope`, `wam.py`, `memory.py`, action `chunk_rope`): rotary
  positions in attention, video tokens by (frame, row, column) of the whole canvas, memory
  slots by their own grid, action tokens by step in the chunk.
- why: attention then sees relative offsets (this patch is 2 to the right of that one) rather
  than learning every absolute slot; it carries over to grids of other sizes and to more frames.
- test: 3-axis RoPE on the video side, 1-axis on the action side, drop the learned positions.

---

### 09 · State as cross-attention context (both transformers)

- step 0/01: the joints are one more token in the video self-attention; the action side
  sees them only through the video keys and values.
- Verne (`mot.py` `project_context`, `transforms.py`): the state is one row read through
  cross-attention. Video blocks cross-attend to it, action blocks to [text rows, state row].
  It holds the current and previous step (a sense of velocity), as poses (58 numbers).
- why: the action side reads the state directly, not filtered through the video layers;
  the previous step gives velocity.
- test: state (now + previous) as a cross-attention row on both sides, drop the state token.

---

### 10 · Layer pairing (action side)

- step 0/01: 4 video layers, 4 action layers, action layer i reads video layer i.
- Verne (`config.py` `paired_video_blocks`, `mot.py`): fewer action layers than video
  layers, each reading a chosen video block (test config [0, 1, 3]); the video side's last
  block is never run, since nothing reads its output.
- why: the action side can be cheaper than the video side and read the depths that help.
- test: 2-3 action layers paired to chosen video layers; compare success and ms/step.

---

### 11 · Cosine LR + EMA weights (training) — **cosine LR DONE**; warmup, EMA later

- step 0/01: AdamW at a constant 3e-4 for 20k steps; the last step's weights are the model.
- Verne: the training code isn't released, so unknown. Both are standard for diffusion and
  flow policies.
- why: at a constant LR the weights keep jittering to the last step; final train loss 0.010,
  0.027, 0.051 across seeds. Decaying the LR (cosine to ~0) or keeping an exponential moving
  average of the weights settles them, which should cut the part of the seed-to-seed spread
  that comes from where training happened to stop (step 0 T1: 13-24 of 50). It won't remove
  the part from each seed memorising its own solution of 10 demos.
- test: cosine alone, EMA alone (decay ~0.999), both; 3 training seeds each; compare the
  spread across seeds and the mean against 54/150.
- later: cosine with warmup; EMA alone and with cosine.

#### Results: 11 · Cosine LR

From `02_cosine_lr.ipynb` (commit 14d7003).

T1-wide, successes out of 150 (3 training seeds × 50 episodes):

| T1-wide | n_exec 8 | n_exec 16 |
|---|---|---|
| step 01 | 18 | 21 |
| step 02, cosine | 29 | 14 |

T1, successes out of 50 (seed 0 only):

| T1 | n_exec 8 | n_exec 16 |
|---|---|---|
| step 01 | 10 | 15 |
| step 02, cosine | 5 | 10 |

- Still only 1 of the 3 runs finds the cube, and it's a different seed than in step 01.
- T1 (10 demos) overfits harder: held-out loss ends at 0.53, against 0.30.
- When a run fails, the gripper drives forward to x ≈ +16 cm and parks past the cube.

---

Not opened: text encoder (T1 has one instruction), real-time chunking, Wan's size, the head-camera crop (§1: centre 87.5%; our top camera already frames the table).

## Data steps (not boxes: the model is unchanged, the demos change)

- **05 · Square up to the cube** — **DONE** (`05_square.ipynb`): the expert turns the wrist to the cube's
  nearest face. Seed 0, T1-wide, n_exec 8 / 16: clean squaring 28 / 25, **+ full shake 42 / 44** (step 04:
  36 / 39), + ±45° aim bias 26 / 33, both 35 / 33. Next: DAgger.

## Side work, parked

- T1-wide blindness (probes.ipynb cells 52-56, notes.md "Box 3"): the step 0 gripper closes in the
  same spots whatever the cube does (slope ≈ 0). Guess: the joints token shortcut. Not part of
  this series.
