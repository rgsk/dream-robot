# Pending walkthroughs

Probes and checks done in scripts that haven't been walked through yet. Each entry says what was asked,
how, and what came out, so it can be rebuilt step by step later.

## T5 colour binding probe (2026-11-13, step 07)

- **Question:** is "which cube spot holds the named colour" learnable from the VAE latents at all?
  The T5 policy picks the named cube at chance (25/50, then 22/50 with the instruction also on the
  action side), while the side word is perfect (0 wrong_bin).
- **Data:** 1,300 fresh T5 scenes from env resets (seeds 5000-6299), first frame only, [top | wrist]
  canvas through the step 07 VAE (`vae.WEIGHTS`, 4 x 16 x 32 latent). Label = the named cube's slot
  (0 = near spot, 1 = far spot; the two spots are front/back from the robot, 14 cm apart).
  Cached: `scratch/verne07/t5_scenes.npz`.
- **Model:** small conv net: input = latent (4 ch) + the colour one-hot tiled as 4 constant channels
  (8 ch) -> 3 conv 3x3 (64 ch, GELU) -> average pool -> Linear(64, 2). AdamW 1e-3, 3,000 steps,
  batch 64, cross-entropy.
- **Runs:** trained on the first 135 scenes (= the demo count) and on the first 1,000; held-out =
  scenes 1000-1299. Chance 50%.
- **Result:** 135 scenes: train 100%, held-out **89%**; 1,000 scenes: train 100%, held-out **100%**.
- **Reading:** the latents carry colour and place well enough; the binding is easy for a model asked
  only that. The policy has the information but doesn't learn to use it (its loss is mostly motion).
- **Script:** `experiments/verne/scripts/t5_colour_binding.py`; log `scratch/verne07/t5_colour_binding.log`.
- **Related probes (same day):** `t5_pick_rule.py` (which cube the policy grasps: a red preference in
  the first run, a far-slot preference in the second), `t5_instruction_probe.py` (swapping the colour
  word flips 12/50 picks; zeroing it 7/50), `vae_colour_check.py` (step 03 VAE: green 7%, blue 0% of
  cube pixels kept; step 07 VAE: 97-99%).
