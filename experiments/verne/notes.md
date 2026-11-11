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

## Protocol from 2026-11-11: 50 rollout seeds (1000-1049), training seeds 0 first, 1 and 2 later
- rollouts.py (env, sweep, close-point probe) shared by every step's notebook; frame_of in data.py.
- Step 0, train seeds 0 + 1 + 2 (00_end_to_end): T1 n_exec 8 **25 + 18 + 13 = 56/150**, 16
  **25 + 17 + 12 = 54/150** (val end 0.246, 0.196, 0.201, doesn't predict order). T1-wide
  **3 + 1 + 0 = 4/150**, **1 + 1 + 1 = 3/150**, all no_grasp. Probe: closes 50/50 each, x slopes
  -0.23/-0.27/-0.11, y +0.04/-0.10/-0.02, intercepts x +3.8/+3.9/+3.1 y -2.9/-1.9/-1.1 cm, miss
  10.5/10.1/10.1 cm = blind. Seed-to-seed on T1: 12-25 of 50.
- Rollouts now one torch seed per episode (seed*100 + n_exec), 8 spawn workers (pod: 50 eps ~1 min;
  one thread per worker, else 128-thread thrash). First pass (one seed per sweep): T1 54/150,
  59/150, wide 5/150, 5/150: same models, within noise. Each worker holds ~2.3 GB GPU: two
  notebooks sweeping at once filled the 4090, a worker hung; pool now has a 10 min timeout.

## Step 01 · camera canvas (2026-11-11), nbs/verne/01_canvas.ipynb
- First pass, old protocol (1 train seed x seeds 1000-1019); to be redone on rollouts.py, 3 x 50.
- [top | wrist] side by side, 128x256 -> 128 patches, same patch_in, learned positions over the
  canvas (Verne transforms.py build_canvas; head-camera crop skipped). 3.68M params.
- T1, 10 labelled, 20k steps: val best 0.167 (2k), end 0.302 (step 0: 0.168, 0.246).
- Rollouts seeds 1000-1019: n_exec 8 **4/20** (no_grasp 15, wrong_target 1), n_exec 16 **6/20**
  (no_grasp 14) = **10/40** vs step 0 17/40. One seed.
- Cache lesson: also=['00_end_to_end'] replayed step 0's train/sweep (identical cells, package
  not in the key). Each step's notebook now reads only its own cache.
- T1-wide (01_canvas step 01.5): 90 train demos, val as probes, 20k steps: val end 0.054 (step 0
  wide 0.05). Rollouts seeds 1000-1019: n_exec 8 **3/20** (no_grasp 17), n_exec 16 **2/20**
  (no_grasp 18) = **5/40** vs step 0 wide 2/40. One seed, within noise.
- Close-point probe on the step 01 wide model (01_canvas cells 28-31), n_exec 8: closes 17/20,
  x slope +0.96 corr 0.97, y slope +0.49 corr 0.55, intercept y -3.3 cm, mean miss **3.6 cm**
  (step 0 wide: slopes ~0, ~10 cm). The wrist view makes it localize; y is half and short.

## Step 01 · camera canvas, 3 x 50 on the pod (2026-11-12), nbs/verne/01_canvas.ipynb
- Train seeds 0, 1, 2 on an RTX 4090 pod (RunPod), rollouts 1000-1049, one torch seed per episode.
- T1 (10 demos): n_exec 8 **10 + 30 + 19 = 59/150**, 16 **15 + 16 + 17 = 48/150** (step 0: 56, 54).
  Val end 0.304, 0.255, 0.239 (step 0: 0.246, 0.196, 0.201). No gain; seed spread 10-30 of 50.
- T1-wide (90 demos): n_exec 8 **3 + 3 + 12 = 18/150**, 16 **3 + 10 + 8 = 21/150** (step 0: 4, 3).
- Probe (wide, n_exec 8): x slopes +0.25 / +0.03 / **+0.61**, y +0.07 / +0.02 / +0.30, miss 9.4 /
  10.2 / **7.5 cm**, all close 50/50. Step 0: all blind.
- Seed 0 on the local 4060 (first pass) = bit-identical when retrained locally, probe x **+0.95**,
  miss 3.8 cm, closes 31/50; the pod's seed 0 (same code) probes blind both on the pod and locally.
  Pod vs local decoded frames identical. -> GPU arithmetic decides which solution a run lands
  on; finding the cube is fragile (2 of 4 canvas runs). Same seed != same run across GPUs: train
  a step's seeds on one machine.
- Parallel sweeps: a worker hung twice (not reproducible serially, 50/50 eps fine). Sweeps now
  collect what finishes in 5 min and run leftovers in the kernel (same per-episode seed).


## Step 02 · cosine LR (box 11), seed 0 only (2026-11-12), nbs/verne/02_cosine_lr.ipynb
- train.py: CosineAnnealingLR(T_max=steps), 3e-4 -> 0, no warmup. Trained on the pod 4090.
- T1 seed 0: n_exec 8 **5/50**, 16 **14/50** (step 01 seed 0: 10, 15). Train 0.002, val **0.544**
  (step 01: 0.012, 0.304): on 10 demos cosine deepens the overfit.
- T1-wide seed 0: n_exec 8 **11/50**, 16 **7/50** (step 01 seed 0: 3, 3). Val 0.042 (0.063).
- Probe seed 0: closes 28/50, x slope **+0.74** (corr 0.91), y **+0.63**, miss **3.0 cm** (step 01
  pod seed 0: +0.25, 9.4 cm, blind). Same seed, same GPU: cosine turned a blind run into a seeing
  one. One seed; seeds 1, 2 pending.
- load_cameras now decodes once per dataset into data/robosuite/<name>/decoded/{top,wrist}.npy.
- Kernel friction: the 02 kernel (~10.8 GB: two canvas sets + training) left no GPU for 8 x 2.3 GB
  workers -> every probe episode fell back to the kernel. Decision (2026-11-12): runs become jobs
  (one command per task/seed -> files under experiments/verne/runs/<step>/, meta = commit + code
  hash, a detached queue on the pod), notebooks become readers. Start of next session.
