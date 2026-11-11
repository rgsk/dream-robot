# Verne, small: Verne V0.5's shape rebuilt piece by piece on T1

Reference: github.com/vernerobotics/v0.5 (cloned at ../verne-v0.5). Code: dream_robot/policies/verne/.
Notebooks: nbs/verne/. Comparison: dreamzero WAMz, 10 labelled T1 demos, 18 + 16 = 34/40.

## Step 0 · dumb whole model (2026-11-10)
- Video transformer (4 layers, 256 wide) over 64 pixel patches + a joints token, run once;
  action transformer (4 layers, 128 wide, same 4 heads x 32) attends to [video k,v of its layer,
  own k,v]; flow matching on 16 joint actions, 10 Euler steps. 3.66M params, 7.2 ms/step (4060).
- Smoke on random frames only so far.
- T1, 10 labelled, 20k steps B32: train 2.03 -> 0.01, val best 0.17 (step 5k), end 0.25 = overfit.
- Rollouts seeds 1000-1019: n_exec 8 **8/20** (no_grasp 9, wrong_target 2, dropped 1),
  n_exec 16 **9/20** (no_grasp 10, dropped 1) = **17/40** vs WAMz 34/40. One seed.
- Missing vs WAMz: VAE latents, future-frame loss (WAMz pixels -> latents went 24 -> 34).

## Probes (nbs/verne/probes.ipynb): opening step 0's boxes
- Box 1 patches: cube = ~2 of 64 patches, half the tokens are empty table.
- Box 1b: grey frame instead of the real one, n_exec 8: **0/20** (all no_grasp) vs 8/20. NOT proof
  the cube is located from the image (see box 2b): a grey frame is just out of distribution.
- Box 2 attention (one held-out frame, tau 0.9): cube patches at chance (0.01-0.02) in all 4
  action layers; joints token 0.10-0.28; weight peaks on top-row background + bin edge (layer 3
  top patch 0.11 = 7x even). Guess: register-like scratch tokens carry the scene summary. Untested.
- Box 2b (n_exec 8): cube erased (table colour, red-pixel patches) 9/20; action tokens blocked
  from top-row tokens 9/20; bottom row (control) 7/20; real 8/20. Register guess wrong.
  T1 cube_xy_range is +-3 cm: the model grasps the average spot, works ~half the time. It does
  not localize the cube. Checks: erase hits 46-54 red px in sim frames; masked sampler == sample.

## T1-wide: a T1 where the eyes have to find the cube (2026-11-11)
- task.yaml's cube_xy_range was read but never applied (robosuite Lift's built-in +-3 cm did it);
  env now builds Lift's sampler from it, per axis half-width or [min, max]. T1 spawns identical
  (10 seeds checked).
- Expert (experiments/verne/scripts/expert_wide.py, 100 seeds): +-10 cm both axes 95/100, all 5
  "stuck in descend" with cube y >= +0.076 (fingers on the bin's near wall, bin starts y +0.134);
  x [-10, 10], y [-10, 6] cm: 100/100.
- Recorded data/robosuite/pick_place_cube_wide_n100: x [-0.10, 0.10], y [-0.10, 0.06], 100/100
  expert, 22,409 frames (record.py --cube-x/--cube-y).
- Step 0 model trained on 90 T1-wide demos, 20k steps: held-out loss 0.05 (no overfit).
  Rollouts seeds 1000-1019: n_exec 8 **1/20**, n_exec 16 **1/20**, cube erased **1/20**, almost
  all no_grasp. The model doesn't use the cube. Unchecked: rollout path bug; where the gripper goes.
- Box 3 (probes cells 52-56): gripper xy at first commanded close vs cube spawn xy, n_exec 8,
  seeds 1000-1019: slope x -0.09, y +0.07, corr ~0.08 -> **blind**. Mean miss ~10 cm (films
  looked closer than they were). Close points in two clumps regardless of cube: x +4..+9 cm
  (14/20) or -4..-8 cm (6/20). Check: expert's close point within 0.1 cm of cube (5 seeds).
