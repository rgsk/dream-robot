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

## Runs as jobs (2026-11-12): dream_robot/policies/verne/run.py
- `run job <step> <task> <seed>` trains + sweeps (n_exec 8, 16) + probes (wide) -> experiments/verne/runs/<step>/<task>/seed<k>/
  (meta: commit, dirty, package hash, GPU; log, sweep_*, probe JSON committed; weights.pt, job.log ignored).
  `run queue` = jobs one by one, fresh process each. Pod has no .git: sync writes synced_from.json (`run synced-from`).
- RunPod L40S (US-TX-4, 16 vCPU, 320 MB/s): load_cameras decode ~6 min per dataset on the pod (not 74 s);
  one job ~13 min wide (train ~6), ~10 min T1.

## Step 02 seed 0 rerun as jobs on the L40S; cosine LR closed (one seed)
- T1: n_exec 8 **5/50**, 16 **10/50**, val end 0.531 (4090: 5, 14, 0.544). Val min 0.167 at 2k steps.
- T1-wide: **9/50**, **4/50**; probe x **+0.73**, y +0.55, miss **3.3 cm** (4090: 11, 7; +0.74, 3.0 cm).
  Seed 0 sees the cube on both GPUs (step 01's seed 0: blind on the 4090).
- Decision (user): seeds 1, 2 not run; cosine LR done for now. Spread-across-seeds question open.
  Wrap-up in nbs/verne/02_cosine_lr.ipynb (step 02.7 + wrap-up).
- T1-wide seeds 1, 2 (L40S): seed 1 **11/50, 6/50**, probe x +0.05, miss 8.7 cm (blind); seed 2 **9/50, 4/50**,
  x +0.23, miss 7.6 cm. Totals n_exec 8 **29/150** (step 01: 18), 16 **14/150** (21). 1 of 3 sees, as step 01.
- Films (scripts/films.py, seed 0 vs 1, scratch/verne_films/) + reach_track.py (10 eps each): every failure
  drives the gripper forward to x ≈ +16-17 cm (robot base at -56 cm; cube spawns to +10) and parks 11-24 cm
  past the cube; seed 0 never comes back (<= 1.7 cm), seed 1 sometimes 11-18 cm. Successes stop at x ~+1.5.
  Blind seed 1 wins when the cube spawns near its spot. Open: do demos ever reach x +16 cm?
- Decision: cosine LR (box 11) done; warmup and EMA later. Roadmap updated.

## Step 03 · VAE latents (box 02), T1-wide 3 seeds (2026-11-12), nbs/verne/03_vae_latents.ipynb
- vae.py: small conv VAE, Verne's encoder shape (res convs, RMSNorm + SiLU, stride-2, mean/logvar, mean
  normalised per channel); canvas 3x128x256 -> 4x16x32; 1.88M params. `run vae 03_vae_latents` on the L40S:
  39,327 frames of both tasks' train demos, 5k steps B64 lr 1e-3 beta 1e-3, ~20 min; val recon **0.0015**
  ([-1,1]; dreamzero top-only 0.0022), cube sharp in both views, gripper a bit soft. weights.pt gitignored!
- Latents encoded once per dataset (decoded/latents_<hash>.npy, fp16, ~90 MB; mean 0 std 1 per channel;
  decode of stored latents 0.0013). Model reads 2x2 latent squares (16 numbers) -> 128 tokens; 3.49M params.
  Rollout encodes the live canvas with the frozen VAE. Job ~6 min with sweeps (was ~13).
- T1-wide: n_exec 8 **34 + 36 + 42 = 112/150**, 16 **36 + 37 + 39 = 112/150** (step 02: 29, 14). Probe x
  +0.99 / +0.95 / +0.99, miss **1.4 / 1.4 / 1.1 cm**: all 3 seeds find the cube (step 02: 1 of 3).
  Failures mostly no_grasp (7-14 per seed).
- Not run: T1 (user: pod terminated next). Notebook wrap-up done; explained notebook to come.
- 03_vae_latents_explained.ipynb (seed 0 models, held-out demo 82): cube squares among the loudest latent
  tokens (rank 0-17 of 128 at start); live vs stored latent rms 0.057 (10-step move 1.04); action tokens
  don't attend to cube tokens (<= even share), joints token 0.25-0.66 in layers 2-3. Linear probe on 1,000
  home-pose first frames (pre-grasp demo frames leak: joints alone 2.5 cm): joints token after video layer 0
  reads cube xy to **1.4 cm** (step 02 seed 0 pixels: 6.2 cm, patches 5.0; mean guess 7.1).

## Step 04 · sampler (box 05), inference only (2026-11-12), nbs/verne/04_sampler.ipynb
- model.sample(steps, shift): Verne's σ = s·u/(1+(s−1)u) grid; (10, 1.0) reproduces the old loop (7e-7).
  scripts/sampler_sweep.py on step 03's 3 T1-wide policies, local 4060 (10 even rerun = L40S scores exactly).
- n_exec 8 / 16 of 150: 10 even 112/112; 4 even 118/120 (won 17, lost 3 of 300 same episodes);
  **4 shift 1.5 120/122 (won 23, lost 5)**; 10 shift 1.5 110/114 (6/6). Fewer steps = the gain, shift ~0.
- Default now (4, 1.5). Sweep: 8 workers OOM on the 8 GB card (parent 3.5 GB) -> 4 workers + empty_cache,
  ~25 s per sweep; EGL render hang 2x (faulthandler dump, leftovers rerun in parent, scores unaffected).
- Close-point probe (scripts/sampler_probe.py, n_exec 8): miss 1.38 -> **1.25 cm** (10 even -> 4 shift 1.5), successes
  1.06 -> 0.97, failures 2.32 -> 2.38 (|dy| ~2.0); > 2 cm share 16% -> 15%; no_grasp 69 -> 50 (n_exec 8+16).
  Borderline closes pulled in; the sideways misses stay. rollouts: worker wait 300 -> 90 s, stack dump 120 -> 30 s
  (EGL render hang in parallel workers, ~1 in 500 eps; never seen serially).

## Step 05 · square up to the cube, a data step (2026-11-12/13), nbs/verne/05_square.ipynb
- Why: failures miss sideways and the fingers close along y; room each side 1.8 cm square, ~1.0 cm diagonal.
- expert.py square_up (turn <= 45 deg to the nearest face, straighten after lift), aim_bias_deg (executed-only
  per-episode wrist bias, clean labels); env.cube_yaw; record.py --square-up / --aim-bias-deg. Expert closes
  1.9 deg off square (fixed wrist 21.3), 100/100 (expert_square.py).
- Seed 0, T1-wide, n_exec 8 / 16: step 04 36/39; a square 28/25; **b + shake 42/44 (miss 1.0 cm)**; c + bias45
  26/33 (miss 2.4); d bias + shake 35/33 (expert 72% of attempts).
- wrist_check.py: a off square 15.2 deg, turn slope 0.18; c 11.5 deg, slope 0.41 (corr 0.63). a hangs on crooked
  holds (wrong_target); c aims off (labels aim at the latched spawn, cube nudged); d hovers square, never closes.
- Next: DAgger with a state-based labeller (StateExpert, uncommitted: alone 30/30, rescues 21/21 policy failures).

## Step 06 · DAgger, a data step (2026-11-13), nbs/verne/06_dagger.ipynb
- StateExpert (expert.py): phase from the scene + memory (settled grip stays held, release final, 2-step phase
  changes) + retries (closed on nothing / slipped -> reopen, rise clear, re-aim). Alone 30/30; takes over step 05
  clean-square failures 21/21. dagger.py + `run dagger`: recorder records the labeller, executes the policy.
  run.py load_task: 'base+round[:fail]', val = base's 10.
- r1: base policy, seeds 2000-2099, placed 79%, 21 failed. r2: r1:fail policy, seeds 2100-2299 (200), placed 76%, 47 failed.
- Seed 0 n_exec 8 / 16: base 42/44; r1 all 45/44; r1 fail 45/47; **r1+r2 fail 48/47** (no_grasp at n_exec 8: 6 -> 0,
  y slope 0.86 -> 0.97, miss 0.9 cm). Failures-only >= all with a third fewer frames.
- Left: all 5 failures hit 600 steps; 4 of 5 place on replay (also at a 40 s limit, within 600 steps): knife-edge
  nondeterminism, not time. Ep 1001 real: slip, crooked re-grip, never lifts. rollouts.make_env(horizon_seconds).
