# Verne, small: roadmap

Step 0 (`dream_robot/policies/verne/model.py`) is Verne V0.5's shape with every box at its dumbest.
Each item below is one box: what step 0 does, what Verne does, and why it might matter.
Boxes open in the order data flows through Verne ([architecture.md](architecture.md)), one step
each, with a results notebook and an explained notebook (`NN_<box>.ipynb`, `NN_<box>_explained.ipynb`).
Scored on T1, 10 labelled demos, rollouts seeds 1000-1019 at n_exec 8 and 16 (step 0: 17/40).

## Boxes to open

- **01 · Camera canvas** (input)
  - step 0: the top camera only, 128×128 → 64 patches of 16×16.
  - Verne (architecture §1): each camera resized, placed side by side as one wide image, so the
    video model sees one picture per moment.
  - why: the wrist view shows the cube big just before the grasp, so the model can correct
    in closed loop.
  - test: [top | wrist] as one 128×256 canvas → 128 patches, same patch embedding, positions over
    the whole canvas.

- **02 · VAE latents** (input)
  - step 0: raw pixel patches, one linear layer each.
  - Verne (§2-3): a frozen VAE turns each frame into a latent grid; a small conv patches the grid.
  - why: the latent keeps edges, objects and colours and drops pixel detail. In WAMz, moving
    from pixels to latents went 24 → 34/40.
  - test: the frozen VAE from the dreamzero notebook (128×128 → 16×16×4), patched to tokens.

- **03 · Memory** (input)
  - step 0: only the current frame.
  - Verne (§4): 16 past slots; recent slots fine, older ones merged coarsely in time and space;
    a learned attention bias per slot; time-ordered mask.
  - why: remembering "I already tried and the cube is still there" (the WAMz 1013 failure).
  - test: a few past frames as extra tokens with a causal mask; then the coarse packing.

- **04 · Time embedding + adaptive norm** (action side)
  - step 0: raw τ (one number, 0..1) → MLP (1 → 128 → 128), added to every action token.
  - Verne (`action_dit.py`): τ → sinusoid features (cos/sin at many frequencies) → MLP → 6 vectors
    per layer (shift, scale, gate for attention and MLP: adaptive layer norm).
  - why: sinusoids let the network tell nearby noise levels apart (0.90 vs 0.95) and act
    differently at each; raw τ through one SiLU only gives smooth, gentle functions of τ. Step 0
    gets away with it because it samples just 10 evenly spaced levels.
  - test: swap in sinusoids (+ adaptive norm), compare held-out loss and success.

- **05 · Sampler** (inference)
  - step 0: 10 evenly spaced Euler steps.
  - Verne (§9): 4 Euler steps, shifted so more of the effort goes to the noisy early steps.
  - why: fewer steps = faster plans; the shift puts the steps where the hard choices are made.
  - test: 4 shifted steps vs 10 even, same weights.

- **06 · Action format** (output)
  - step 0: absolute joint positions + gripper, 16 steps.
  - Verne (§10, §12): gripper pose relative to the current hand (xyz + 6-number rotation + jaw),
    plus progress 0..1; IK outside the model turns poses into joints.
  - why: "move 2 cm down" looks the same anywhere; robot-agnostic; progress says when a step is done.
  - test: relative end-effector targets + IK, progress head; compare success.

- **07 · Future-video loss** (training only)
  - step 0: actions are the only target.
  - Verne (training, inferred): some future frames are added as noisy tokens and the video side
    learns to denoise them; at run time that branch is never used.
  - why: predicting the future forces the video side to understand dynamics (WAMz's gain).
  - test: noisy future-frame tokens + a frame loss in training; run time unchanged.

Not opened: text encoder (T1 has one instruction), real-time chunking, Wan's size.

## Side work, parked

- T1-wide blindness (probes.ipynb cells 52-56, notes.md "Box 3"): the step 0 gripper closes in the
  same spots whatever the cube does (slope ≈ 0). Guess: the joints token shortcut. Not part of
  this series.
