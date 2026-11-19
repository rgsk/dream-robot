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

## Step 07 · grasp lower, a data step (2026-11-13), nbs/verne/07_low_grasp.ipynb
- Expert closes ~1 cm above its target while descending; at grasp_dz +0.5 cm it caught the cube's top quarter (+1.5 cm).
  T5 + shake (grasp_height_t5.py, 180 seeds): +0.5 150, **-0.5 163**, -0.5 tight 160, 0 tight 143. grasp_dz -0.005 in T1 + T5
  task.yaml; clean 50/50 both. T1-wide low data: expert 93% (was 85%).
- Package: new VAE (step 08), instruction token (zeros on T1), clipping 1.0, checkpoints; Verne LR WSD -> 29/23 (miss 3.1 cm,
  slopes 0.61/0.69), **back to cosine: 43/33** (miss 1.5, 0.86/0.84); step 05 42/44. Held-out loss ~1.1 = ep 82 outlier action (10 sd).
- Failures: close beside the cube, carry empty, park (no retry) -> DAgger. Runs on Modal (scripts/modal_app.py).

## Step 08 · the instruction on T5 (2026-11-13), nbs/verne/08_t5_instruction.ipynb
- T5 expert squares up to the named cube (T2 expert reuses T1 __call__; T5 cube_yaw = named cube); data
  instructed_pick_place_square_noise005_n150 (low grasp + shake, 91%).
- Step 03 VAE paints T5 cubes red (green 7%, blue 0% of cube pixels); new VAE on T1 + T1-wide low + bias45 + T5: 97-99%.
  VAE runs: local cosine 0.0015 lost (crash in latent stats, no ckpt); WSD 1e-3 spiked 20x at 3500/7000; WSD 5e-4 + clip spiked
  5x at 7000 (same batches) -> step6000.pt (0.0017) is vae.WEIGHTS.
- Instruction one-hot (6 words) as a video token: T5 22/22 of 50, 0 wrong_bin, wrong_cube ~21; + added to action tokens 21/17.
  Picks: run 1 "red" (named red 16/19), run 2 "far" (31/50). Colour swap flips 12/50 picks, zeroing 7/50.
- Binding probe (t5_colour_binding.py): picture + colour -> near/far, 135 scenes 89% held-out, 1000 -> 100%. Info is there; the
  policy doesn't bind it. Next: near/far aux head (train-only), then future-video loss; more T5 demos.

## Step 09 · slot head on the instruction token, T5 seed 0 (2026-11-13), runs/09_slot_head
- Head: Linear(256 -> 2) on the instruction token at the input of the last video layer (its k,v are what the last
  action layer reads; the last layer's output feeds nothing). Label = target_slot (instructions.json), only frames
  before the first gripper close (34% of rows); loss flow + 0.1 CE; logs held-out first-frame accuracy (home pose).
- First labels ran to the episode end: 98% held-out in 1k steps = read off the arm's pose (user caught it), cut at close.
- Head, held-out first frame (15 eps): 0.60 at 0 -> 0.87 at 4k -> 0.93 (14/15) at 16k-20k. Val flow 0.091 (step 08 ~0.1).
- Rollouts: 28/26 of 50 (step 08: 22/22, + action side 21/17); wrong_cube 17/15 (~21). Named cube grasped 32/50
  (step 08 25 and 22). Bias toward slot 1: picked 29/50; target slot 0 right 18/33, target slot 1 right 14/17.
- Read: the head binds (14/15 at home pose) but the arm follows it in only ~2/3 -> binding helps, reading it is partial.
  One seed. Next: head's first-frame answer vs the cube grasped per rollout; seeds 1-2.
- Head vs arm per rollout (slot_head_vs_pick.py -> runs/.../head_vs_pick.txt): arm = head 44/50 (head right + arm right
  32, head wrong + arm wrong 12, head right + arm wrong 6). The arm reads the head; the head is the limit:
  38/50 in the sim (vs 14/15 held-out demos), wrong ones confident (p(named) ~0), 11/12 say slot 1 when it's 0
  (train prior 63% slot 1). Train data balanced per prompt (no word -> slot shortcut). Matches the binding probe at
  135 scenes (89%); 1000 scenes -> 100%. Next: more labelled scenes for the head (first frames need no demo).

## Step 09b · slot head + 1000 demo-less scenes (2026-11-14), runs/09_slot_head/t5-scenes
- data.load_scenes: T5 resets seeds 5000-5999 (train) / 9000-9199 (val), first frame through the VAE + target_slot,
  cached data/robosuite/t5_scenes/<part>_<vae tag>.npz (also on the data volume). Task t5-scenes: each step the head
  also gets 32 scenes, its loss = mean(demo rows, scenes); actions untouched.
- Head on val scenes: 0.49 -> 0.985 at 5k, 0.985-0.995 after. Demo first frames 15/15. Val flow 0.092 (09: 0.091).
- **Rollouts 44/43 of 50** (09: 28/26, 08: 22/22). Wrong cube 1/1; named cube grasped 49/50; head right 50/50 on the
  rollout seeds, arm = head 49/50. Left: no_grasp 4/4, dropped 1/2 (the T1-wide failure modes).
- Read: binding was the missing piece, and it needed more scenes than 135 demos give; the arm reads the head's answer
  through the last layer's k,v. Label is sim-only here. One seed.

## Step 10 · future frame on the video side (box 07), T5 seed 0 (2026-11-14), runs/10_future_video/t5
- Slot head + scenes removed from the package (sim-label test, step 09 is its record). Package = step 08 + future frame:
  noisy latent of frame now+16 as 128 more video tokens (own tau), read the present, never read by it (actions'
  k,v unchanged, diff 1e-6); x-prediction, weight 1.0. Binding probe = binding_probe.py on frozen weights (linear,
  instruction token entering the last video layer, 1000 reset scenes fit / 200 scored): step 08 66%, 09 81%, 09b 98%.
- Losses: flow val 0.100 (08 ~0.1); future val 1.27 -> 0.20.
- **Rollouts 19/16 of 50** (08: 21/17 same architecture); wrong cube 28/27; named cube 21/50 (chance); picks slot 1 30/50.
- **Probe 79.5%** (08: 66%): the future loss puts more binding in the token, about step 09's level, but the arm doesn't
  follow it (09 at 81% picked 32/50).
- future_frames.png: top camera predicted well (arm, cubes, bins); wrist view one-shot = blank table, samples = blobs.
  At t=0 the real future (0.5 s) shows the arm barely moved: the frame carries almost no which-cube signal at the
  moment the choice is made. Result as feared: 22-32 band, actually at the bottom.

## Step 10b · future clip: top camera at +16..+80, open mask, T5 seed 0 (2026-11-14), runs/10_future_clip/t5
- Why: at t=0 the +16 frame shows the arm barely moved; real demos (commit_horizon.png) commit only at +64..+80
  (+48: gripper over the upper cube either way; first close at 81-114). Clip = 5 frames x 64 top patches, one shared
  tau, future frames read each other both ways (user's pick; causal / independent masks not run).
- Losses: flow val 0.096; future val 1.18 -> 0.18.
- **Rollouts 19/13 of 50, named cube 22/49 (chance)**; picks slot 1 28/49. **Probe 79.5%** (same as 10).
- future_frames.png (4 val eps x t 0/40; real / one-shot / sampled): the arm path is right in shape (leave home, descend
  to the cubes); for named-slot-1 episodes the dream hovers at the upper cube; cubes change colour in the dream
  (green -> blue, red -> orange/green), as rough/t5_video's dreams recoloured the held cube.
- Read: the future loss doesn't bind colour to cube in this model; the dream itself doesn't know which cube.
  The mask isn't the limit (probe unchanged from 10, dream already wrong at high noise).

## Step 11 · instruction words as cross-attention rows, T5 seed 0 (2026-11-14), runs/11_cross_attn/t5
- Package = step 08 with the words moved out of the token sequence: each side has a 7-row word table ('no word' + WORDS),
  every video and action block cross-attends to [no word + the prompt's words] (masked), Wan's block order
  self -> cross -> MLP. Instruction token, instruction_act and step 10's future branch removed. 4.28M params (08: 3.49M).
  Walkthrough: nbs/verne/11_cross_attn_explained.ipynb (shapes; 1/130 attention share vs a softmax over the words alone).
- Losses: train 0.025, val 0.095 (08 ~0.1).
- **Rollouts 14/13 of 50** (08: 22/22 same seed); wrong_cube 20/23 (08: ~21-23), no_grasp 16/13 (08: 5/8).
  Probe (gripper at first close vs named cube): slopes ~0, miss 8.7 cm (08: 7.7) -> still blind to which cube.
- Read: the wiring alone doesn't bind colour to cube (wrong cube unchanged at ~chance) and grasping got worse (one seed).
  Binding probe not run: its read point (instruction token 1) no longer exists.

## Step 11b · zero rows instead of mask + 'no word' (Verne's padding), T5 seed 0, runs/11b_zero_rows/t5
- words_of only: 6 word rows, absent words zero, all attended (no mask, no 'no word' row). Reverted after the run.
- Losses: train 0.021, val 0.101.
- **Rollouts 13/13 of 50** (11 masked: 14/13; 08: 22/22); wrong_cube 29/26 (11: 20/23), no_grasp 8/11; miss 9.8 cm.
- Read: padding style isn't the lever; both cross-attention variants bind colour no better than chance, below step 08.

## Step 12 · state as a cross-attention row (roadmap box 09), T5 seed 0 (2026-11-18), runs/12_state_row/t5
- Package = step 11 with the state token gone: state [previous control step, now] (16 numbers, previous = now at a
  demo's first step / reset) through a linear-GELU-linear per side, one more cross-attention row next to the word
  rows on both sides. The action side reads the state directly. 4.37M params (11: 4.28M).
  Check: scripts/state_row_check.py (shapes, prev clamp, video k,v fixed + state changed -> velocity changes).
- Losses: train 0.028, val 0.094 (min 0.081; 11: 0.095, min 0.084).
- **Rollouts 23/20 of 50** (11: 14/13; 08: 22/22). no_grasp 6/6 (11: 16/13), wrong_cube 20/23 (11: 20/23, unchanged),
  dropped 1/1. Probe (gripper at first close vs named cube): miss 7.0 cm (11: 8.7), slopes x 0.10 y 0.28 -> still ~blind.
- Read: the direct state read fixes the grasp step 11 lost (no_grasp 16 -> 6) and gets back to step 08's level;
  colour-to-cube binding is untouched (wrong cube still ~chance). One seed.

## Step 13 · near / far words, prompts by colour / place / both, T5 seed 0 (2026-11-18), runs/13_prompt_kinds/t5
- Package = step 12 + WORDS = near far | red green blue yellow | left right (near = slot 0, x -0.07, nearer the
  robot base at x -0.56). Each T5 training sample's prompt is drawn uniformly: colour ('red left', as recorded),
  place ('near left'), both ('near red left'); slots from instructions.json (43% near). Val loss colour-only.
  T5 swept with each kind (sweep_<n>.json = colour, _position, _both). Check: scripts/prompt_kinds_check.py.
- Losses: train 0.029, val 0.093 (12: 0.094).
- **Rollouts (n_exec 8 / 16, of 50):** colour 16/19 (12: 23/20), wrong cube 27/23 (~chance); **place 40/39,
  wrong cube 0/0**; both 39/40, wrong cube 1/0. Place failures are grasps: no_grasp 8/5, dropped 2/6.
  Probe (colour prompts): miss 8.6 cm, slopes ~0.
- Read: told where the cube is, the arm goes there every time; ~40/50 is the grasp ceiling of this policy.
  The whole colour gap is the colour -> place lookup. With both words the model uses near/far. One seed.
- Next: step 14 = near/far head pretraining on 1000 demo-less scenes (colour-only prompts), head on the video
  side's last-layer input, then fine-tune on demos (13a); target ~40/50 on colour prompts.

## Step 14 · stage 1: captioned denoising pretraining (Wan-style), seed 0 (2026-11-18), runs/14_caption_pretrain/pretrain
- Probe first (scripts/patch_binding_probe.py): linear, patches entering the last video layer flattened
  (128 x 256), colour prompt -> named slot, 1000 reset scenes fit / 200 scored. Untrained 73%, step 13 71%;
  'near red left' ceiling 100% for both. Floor ~70-73%, not 50% (32,768 features). Rule fixed before: >=85%
  stage 2, <=76% stop.
- Stage 1 (pretrain.py): video side gets a noise time + frame_out; T5 reset scenes noised (tau ~ U(0, 1)),
  one-cube caption ('near red', place + colour), x-prediction; 20k steps, B 64, cosine 3e-4.
  Scenes re-rendered with both colours (data.load_scenes, *_colours.npz, also on the data volume).
- Log: wrong-caption gap ~0.09 at tau 0.1 from step 1000 on (caption used). Overfits: tau 0.1 val best at
  6k (0.248) -> 0.352 at 20k while train 0.075 -> 0.029.
- denoise.png: top camera paints the named cube's colour in the named place, 4/4 val scenes both captions;
  wrist blurrier.
- **Probe: pretrained 53%** (untrained in the same job 68%), ceiling 100% -> stop, no stage 2.
- Read: captioned painting builds place -> colour ('near? paint red') but not colour -> place, the lookup the
  arm needs; at this scale Verne's route to binding doesn't carry over. Next: near/far head pretraining (13a).

## Step 15 · erase pretraining: a colour word erases its cube, then demos, T5 seed 0 (2026-11-18), runs/15_erase_pretrain
- Step 14's denoising replaced. Stage 1 (pretrain.py): reset scene + one colour word ('red') -> the change that
  hides that cube (latent(erased) − latent(scene); data.load_erased, alpha 0 + re-render, both cubes per scene),
  read by edit_out from the patches entering the last video layer (the last action layer's read point; the last
  video layer's output no longer computed). 5k steps, local 4060. Bug caught by erase_check.py: the loader first
  hid cubes in a stale sim model (robosuite rebuilds it each reset) -> erased == scene; now asserted.
- Stage 1: erase accuracy 100% from step 1000; val right 0.0039, wrong word 0.37; no overfit. edit.png: named
  cube erased in both cameras, 4/4. **Binding probe 99.5%** (untrained 69%, step 14 53%).
- Stage 2 (run.py job --init, local 4060): whole model fine-tuned on the 150 demos, three prompt kinds.
  Val 0.108 (13: 0.093).
- **Rollouts (n_exec 8 / 16): colour 22/24** (13: 16/19), wrong cube 17/18 (13: 27/23); place 42/40, both 41/43.
  Close-point: miss 6.2 cm, slopes 0.21 / 0.27 (13: 8.6, ~0). **Binding probe after fine-tune 87.5%** (stage 1 99.5%).
- Read: erasing (colour -> place) builds the binding where the arm reads; the arm uses some of it (+6/+5,
  wrong cube -10/-5), and demo fine-tuning erodes it (99.5 -> 87.5%). One seed.
- Next: keep the erase loss on during stage 2 (co-train, as 09b's head: 44/43), or freeze the video side.

## Step 16 · erase pretraining + colour-only demo prompts, T5 seed 0 (2026-11-18), runs/16_colour_only/t5
- Package = step 15 with the prompt mix removed: demos train (and val) with their recorded prompt ('red left');
  sweeps colour-only. Note: steps 13-15 val batches also mixed prompts (get_batch's mix defaulted on), so their
  val losses were not colour-only; rollouts unaffected. Stage 2 only, init = step 15's stage 1
  (15_erase_pretrain/pretrain/seed0, probe 99.5%), local 4060.
- Losses: train 0.020, val 0.114 (min 0.089).
- **Rollouts 42/42 of 50** (12 colour-only, no pretraining: 23/20; 15 mixed: 22/24; place-prompt ceiling ~40;
  09b sim-label head: 44/43). wrong_cube 4/4, no_grasp 1/3, dropped 3/1.
- Close-point: miss 2.0 cm, slopes x 0.82 / y 0.52 (12: 7.0 cm, ~0.1-0.3). **Binding probe after demos 94.5%**
  (15: 87.5%).
- Read: with near/far prompts in the demo mix the arm took the place-word route and the binding eroded; with
  colour-only prompts the action loss needs the binding, so it is kept and used. Colour now reaches the
  place-prompt ceiling. One seed.
- Bug: the binding probe OOMs right after the sweeps (rollout workers still releasing GPU memory), twice;
  reran the job to finish it. Fix: probe before the sweeps.

## Step 17 · roadmap box 04: sinusoid noise level + adaptive norm (action side), T5 seed 0 (2026-11-18), runs/17_adaln/t5
- Package = step 16 + Wan's action time path: 1000·σ -> 256 sinusoids -> MLP -> per action layer shift/scale/gate
  (attention, MLP); video side unchanged. Init: video side of 15_erase_pretrain/pretrain/seed0, action side fresh.
  Also: binding probe now runs before the sweeps (no OOM).
- Losses: train 0.026, val 0.102 (min 0.086) (16: 0.114, min 0.089): lower.
- **Rollouts 19/22 of 50** (16: 42/42). wrong_cube 17/16 (16: 4/4), no_grasp 12/12 (16: 1/3).
- Close-point: miss 6.7 cm, slopes 0.16 / 0.20 (16: 2.0 cm, 0.82 / 0.52): the arm barely follows the cube.
  Binding probe after demos 84% (16: 94.5%).
- Read: a regression, and not in the loss (val is lower): closed loop the arm stops tracking where the cube is.
  One seed.
- Seeds (local 17 s1; Modal 16 s1 with step 16's own code, 2cec91a): 16 = 42/42, 36/36; 17 = 19/22, 25/25.
  Slopes 16: 0.82/0.52, 0.58/0.42; 17: 0.16/0.20, 0.18/0.15. Val loss 0.098-0.114 for all four: val doesn't see it.
- scripts/time_use_check.py (val loss at fixed τ with another sample's frame / state): 17 uses the frame 35-50%
  less at τ 0-0.4 (both seeds), state about the same there, more at τ 0.9. Suspect: Wan's gate init (±0.1-0.3)
  starts the frame's path (self-attention, gated) nearly shut while words + state (cross-attention) are ungated.

## Step 17b · box 04 with gates starting at 1 (gate = 1 + learned, modulation and time_out's last layer zero), runs/17b_gate1/t5
- At init the action side computes exactly step 16's.
- Seed 0: **40/39 of 50** (16: 42/42, 36/36; 17: 19/22, 25/25). wrong_cube 6/9, no_grasp 4/2. Miss 2.9 cm, slopes
  0.67 / 0.44. Probe 94.5%. Val 0.112.
- Read: the regression was the gate init; with gates at 1, box 04 is inside step 16's band. One seed.
- Seed 1: **20/21**, wrong_cube 21/22, miss 6.9 cm, slopes 0.17 / 0.17, probe 89%. So seed 0 was the lucky one.
- time_use_check, frame gap at τ 0 / 0.18 / 0.40: 16 s0 .046/.063/.088 (42), 16 s1 .057/.080/.120 (36),
  17b s0 .046/.063/.085 (40), 17b s1 .024/.034/.044 (20); 17 s0/s1 .030/.043/.061, .029/.041/.055 (19-25).
  The frame gap at high noise sorts all 6 runs into the two modes (>= ~.045: tracks the cube, ~40; <= ~.03:
  ignores it, ~20). State gap at τ 0.9 is ~2x step 16's in every adaptive-norm run, either mode: adaptive
  norm's signature, not the failure.
- Read: two outcomes per training run, same val loss; the demo loss doesn't force the frame read at high noise.
  Adaptive norm lands in the bad one more often (3 of 4 vs 0 of 2). The gate start alone isn't the cause.

## Step 17c · gate 3: cross-attention (words + state) gets shift/scale/gate too, all start at 0 (gate = 1 + learned), runs/17c_gate3/t5
- Seed 0: **34/38 of 50**, wrong_cube 7/4, no_grasp 9/7. Miss 3.5 cm, slopes 0.59 / 0.47 (tracking mode). Probe 91.5%.
  Val 0.117.
- First sweep was 0/50: bogus. The rollout workers imported model.py after I edited it to 17d mid-job; restored
  17c (same package hash), reran the sweeps only.

## Step 17d · gate 3 with Wan's small start on all three gates (modulation randn/√dim, gates as is), runs/17d_gate3_small/t5
- Seed 0: **19/23 of 50**, wrong_cube 23/24, no_grasp 8/3. Miss 8.3 cm, slopes -0.08 / 0.11 (frame-blind mode).
  Probe 84%. Val 0.098.
- Read: starting every path shut (frame included) didn't let the frame earn its way in; same as step 17. One seed.

## Step 17e · frame's path opens first: attention gated by 1 + gate (starts ~1), MLP + cross-attention gates Wan-small, runs/17e_attn_open/t5
- Seed 0: **36/36 of 50**, wrong_cube 4/4, no_grasp 8/9. Miss 2.8 cm, slopes 0.71 / 0.53 (tracking mode). Probe 94%.
  Val 0.121.
- Read: tracking, with wrong cube as low as step 16's (4/4); no_grasp a bit higher. One seed.
- Seed 1: **43/42 of 50**, wrong_cube 4/6, no_grasp 2/2. Miss 2.2 cm, slopes 0.85 / 0.63 (best tracking yet).
  Probe 95%. (17b broke on seed 1; 17e didn't.) Seed 2 on Modal.
- Seed 2 (Modal, A100-40GB): **38/38 of 50**, wrong_cube 6/5, no_grasp 5/4. Miss 3.1 cm, slopes 0.69 / 0.46. Probe 94.5%.
  17e: 3 of 3 seeds track the cube (36, 43, 38; step 16: 42, 36).

## Step 17f · attention + MLP start open (1 + gate), cross-attention Wan-small, runs/17f_mlp_open/t5
- Seed 0: **38/36 of 50**, wrong_cube 9/8, no_grasp 1/4. Miss 3.3 cm, slopes 0.64 / 0.60 (tracking). Probe 91.5%.
- Seed 1: **37/31 of 50**, wrong_cube 9/9, no_grasp 4/8. Miss 3.5 cm, slopes 0.60 / 0.52 (tracking). Probe 92%.
- Seed 2: **36/36 of 50**, wrong_cube 7/9, no_grasp 7/5. Miss 3.1 cm, slopes 0.68 / 0.58 (tracking). Probe 93.5%.
- Read: 3 of 3 track (seeds 0-1 at first: 2 of 2), where 17b (everything open, cross-attention too) went frame-blind on seed 1: what tipped
  runs was the words + state path starting open. 17e vs 17f: equal within seed noise (wrong cube 4-6 vs 8-9 is
  inside step 16's own seed spread, 4 vs 7-8). Keep 17e: the smaller departure from Wan (one gate changed, not two).

## Box 04 summary (frame-blind = close-point slopes ~0.1-0.2, ~20/50)
| start of gates                                   | runs (n_exec 8)  | frame-blind |
| no gates (16)                                    | 42, 36           | 0 / 2       |
| all small, cross-attn ungated (17)               | 19, 25           | 2 / 2       |
| all at 1 (17b)                                   | 40, 20           | 1 / 2       |
| all at 1, gate 3 (17c)                           | 34               | 0 / 1       |
| all small, gate 3 (17d)                          | 19               | 1 / 1       |
| attention at 1, MLP + cross small (17e)          | 36, 43, 38       | 0 / 3       |
| attention + MLP at 1, cross small (17f)          | 38, 37, 36       | 0 / 3       |
Rule: the frame's path (self-attention) must start at least as open as the words + state path (cross-attention).

## Open: Modal speed
- 17e seed 2 on Modal got an A100-40GB (L40S out of capacity; fallback list): ~40 s per 1000 steps vs ~12 s on
  the local 4060. User: earlier L40S runs on Modal were much faster. Settle later: time an L40S vs A100 run
  (per-1000-step wall time from the dashboard), and check CPU count / workers on each.

## Step 18 · roadmap box 08: RoPE on both sides (video (frame, row, column) 12/10/10 channels; action chunk step), learned positions gone, runs/18_rope
- Stage 1 (erase pretraining redone, step 15's used learned positions): erase acc 100% from step 1000, val right
  0.0035 (15: 0.0039), **probe 100%** (15: 99.5%; untrained 69.5%).
- Stage 2 seed 0: **41/40 of 50**, wrong_cube 5/5, no_grasp 4/5. Miss 2.7 cm, slopes 0.72 / 0.58 (tracking).
  Probe 93.5%. Val 0.110. (17e: 36/43/38.)
- Close-point probe: one rollout worker hung in EGL render (GPU 7.7/8 GB); killed by hand, the job finished it.
- Seeds 1-2 (Modal, A100): **42/40, 39/39**, wrong_cube 5/4, 6/4; slopes 0.76/0.39, 0.70/0.48. RoPE: 3 of 3 track,
  41/42/39 vs 17e 36/43/38: no regression (slightly up).

## Step 19 · video pretraining by clips (image-to-video, Wan I2V-style), no actions / labels / erase
- 19a clips (data.clip_rows): present at a random step + 7 future frames 32 steps apart (224 steps; T5 episodes
  192-324, median 251), held at the episode's end. scripts/clip_film_check.py on the T5 demos: from the reset a
  clip shows the whole task (named cube -> named box); from a random step most clips start after the grasp, so
  only ~1/3 of clips need the colour -> cube lookup. Knob if the probe doesn't rise: bias the present early.
- 19b model.clip_velocity: present (frame 0) attends to itself only, future (1-7) to all; noise level = an
  embedding added to the future tokens (no adaptive norm on the video side yet); gen_out after the last layer;
  RoPE frame axis. Checked: the present's tokens are identical with any future and equal the policy path's.
  scripts/clip_overfit_check.py (8 clips, 1500 steps): loss 2.50 -> 0.11, dream mse 0.048 (copying the present:
  1.35); dreams match the real clips (scratch/clip_overfit.png).
- 19c (run.py clips, Modal H100, 20k steps, B 32, the 150 demos' clips; train eps for training, val eps for loss):
  binding probe on the present every 4k steps: 69.5 (step 0), 68, 79, 81, 81.5, **81.5%** (20k): between the
  stop line (76) and the gate (85); flat from 12k. Caption use (val loss right / shuffled caption) grows all run:
  τ 0.5 0.35 / 0.81 at 20k; right loss flat from 8k (τ 0.2 slightly up: 0.54 -> 0.58, mild overfit on 140 eps).
- films.png (val reset scenes, each cube's colour word, same side): **6 of 6 dreams move the named cube** (wrist
  view shows the named colour in the gripper, the other cube stays), blurry. The model does the lookup when it
  generates; the present's tokens carry part of it (81.5% vs erase pretraining's 100%).
- 19 stage 2 (run A: init = the clip-pretrained video side, action side fresh; as step 18 otherwise), seed 0:
  **25/22 of 50**, wrong_cube 23/19, miss 7.1 cm, slopes 0.07 / 0.29: frame-blind outcome. Binding probe after
  demos **85.5%** (up from 81.5%). The present carries more of the binding than before, but the arm doesn't
  follow the cube's position. One seed; no-pretraining control (B) not run yet.
- 19b (rows_to='present': only the present cross-attends to words + state, the future reads neither; else = 19c,
  Modal H100): probe 69.5 (0), **86 (4k)**, 87.5, 86.5, 88, **88.5% (20k)**: passes the 85% gate, and fast (19:
  68% at 4k, 81.5% at 20k). Caption use as 19 (τ 0.5 right 0.33 / shuffled 0.79). films.png 6 of 6 named cube in
  the gripper, a little messier than 19.
- 19 stage 2 seed 1: **27/28**, wrong_cube 16/17, miss 5.4 cm, slopes 0.33 / 0.42 (in between: weaker tracking than
  the erase runs' 0.7, not flat), probe after 79.5%. 19 (rows to all) init: 25, 27 vs erase init (18) 41/42/39.
- 19b stage 2 seed 0: **14/18**, wrong_cube 30/22, miss 8.7 cm, slopes -0.06 / 0.09: frame-blind. Probe after
  demos 81.5% (pretrained 88.5%). A higher probe after pretraining did not give a better arm (19: 25, 27; erase: 41,
  42, 39). Clip-pretrained inits 0 of 3 track well. Missing: the no-pretraining control on this package.
- Modal speed (open item): stage 2 on an H100 also ~30-40 s / 1000 steps (A100 40 s, local 4060 12 s): the GPU
  doesn't matter for the 4.5M-param action training, host per-step overhead does. Clip pretraining (1024 tokens) is
  real GPU work: H100 ~2.7x the 4060. Rule for now: stage 2 local (Modal only for parallel seeds), clips on Modal.
  To try later: pin torch CPU threads, CUDA graphs / torch.compile for the train step.
- Run B, no pretraining (fresh Verne, stage 2 only, current package), seed 0: **20/22**, wrong_cube 23/20, miss 7.5
  cm, slopes 0.10 / 0.14: frame-blind. Probe after demos 79.5%. So clip pretraining is neutral-ish (19: 25, 27;
  19b: 14), not harmful; erase pretraining (41/42/39) is what made the arm follow the cube. One seed each.
- scripts/patch_locate_probe.py (per patch: does the token entering the last video layer mark 'I am the NAMED cube'?
  one linear head shared over patches; labels from the erased renders; val 'named > other' = mean score on the named
  cube's patches beats the other cube's; chance 50%): untrained 53.5%; erase pretrain **100%**, erase stage 2 s0/s1
  **77 / 76%**; clips 19 pretrain 55%, 19b 54.5%, their stage 2 54-55%; no pretraining stage 2 52.5%. AUC (named vs
  all patches) 0.86-0.99 everywhere (finding cubes is easy; telling which one is named is the test).
  Read: this separates the arms that track (erase) from the frame-blind ones (all others); the whole-frame probe
  (79-88% for everyone) doesn't. Clip pretraining binds (films 6/6, whole-frame 81-88%) but NOT on the cube's own
  patches: likely the two-hop route, the present relays the word and the future's queries find the colour.
  Results in runs/19_clips/patch_locate_probe.json.
- 19d (future_by='heads': video side runs the present only, as the policy; two word- and joint-blind heads, top
  and wrist cameras, arm-shaped (d 128, 4 layers, layer i reads video layer i's present keys / values) predict the
  7 future frames; Modal H100, 20k): locate probe (named > other) **51.5%** at 16k-20k (chance; erase 100%);
  whole-frame probe 88.5-89%. Val loss higher than 19 (τ 0.2 0.78 vs 0.58), caption still used (shuffled 1.06).
  Read: no mark on the named cube's patches. The heads still use the word (relayed through the present's tokens)
  and do the matching in their own 4 layers.
  Trajectory: locate 52.5 (0), 52.5, 51, 51.5, 51.5, 51.5; whole-frame 68.5, 80, 90.5, 91, 89, 88.5. Val right τ 0.2
  0.60 (4k) -> 0.78 (20k): the heads overfit the 140 episodes after ~4k. films.png 6 of 6 named cube in the gripper.
- 19e (future_by='change': no generation; one linear layer per present token predicts its own patch's change at each
  of the 7 future frames, read where the arm's last layer reads; local 4060, 20k): locate probe (named > other) 53
  (0), 56.5, **67, 66.5, 68, 67.8%**: the first video-only objective to mark the named cube above chance (19-19d:
  51-55%; erase 100%, erase stage 2 76-77%). Whole-frame 87-90.5%. Caption used (val 0.77 right / 0.95 shuffled).
  films.png = change picture: heat dominated by the arm's area and the moving wrist view; the named cube isn't
  visibly singled out by eye.
- 19e stage 2 seed 0: **29/27**, wrong_cube 16/15, miss 5.5 cm, slopes 0.35 / 0.36 (partial tracking, like 19 seed 1).
  Locate probe after demos 68.5% (kept). Best of the video-only inits (none 20; 19: 25, 27; 19b 14) but far from
  erase (41/42/39, locate 76-77%). One seed.
- 19f (future_by='change_top': 19e's head, loss on the top camera's patches only; local, 20k): locate 52.7 (0),
  61.8 (4k), 59, 57, 59.5, **60%** (19e 67.8%); whole-frame 93-95%. Change picture: heat on the arm's path, same for
  either colour word. Stage 2 seed 0: **27/28**, wrong_cube 14/16, miss 5.1 cm, slopes 0.43 / 0.42; locate after
  demos 59%. ~ 19e (29/27). Read: top-only didn't sharpen the mark (the probe also scores wrist patches, which got
  no loss); change targets plateau at 60-68%, arms at ~28 vs erase 41.
- 19g co-training (stage 2 from 19's clip weights; every step action loss + 1.0 x the clip loss ('all'), same
  train episodes; Modal H100 for training, sweeps local after the H100's rollout workers hit GPU graphics faults):
  locate 53.7 (0), 54.5, 60.2, 61.5, 60.5, **61.5%** (20k): the arm's pull lifts the mark from 54% and it levels off
  by 8k; whole-frame 87%; clip loss holds ~0.29. Rollouts **27/25**, wrong_cube 18/18, miss 6.2 cm, slopes 0.32 /
  0.12. ~ 19e / 19f (29, 27); erase 41/42/39. Read: co-training moves some binding onto the present (locate +7)
  but stalls around 61%, like the change heads; marks of 60-68% give arms of ~27. One seed.
- 19h co-training from scratch (fresh Verne, action + 1.0 x clip loss ('all') from step 0; Modal L40S/A100):
  locate 52.5 (0), 54.2, 56, 54.8, 54.5, **53.7%** (20k); whole-frame 81-82.5%; clip loss 0.37 (19g: 0.29).
  Rollouts **16/18**, wrong_cube 26/24, miss 8.5 cm, slopes -0.15 / 0.20: frame-blind, like actions alone (B: 20,
  locate 52.5%). Read: from scratch, the future pressure didn't put the binding on the present at all, nor did it
  help the arm; 19g's 54 -> 61.5% needed the pretrained clip knowledge first. One seed.
- 19i paired co-training (= 19g, but the clip batch is the demo batch's own moments: same present, its future;
  one joint Verne-style pass in effect; Modal L40S, 1 CPU): locate 53.7, 54.2, 56.5, 58.5, 59.2, **61.8%** (20k,
  slower climb, still rising); whole-frame 87%. Rollouts **26/24**, wrong_cube 15/18, miss 5.7 cm, slopes 0.36 /
  0.15. = 19g (61.5%, 27/25): pairing makes no difference. One seed.
- Modal: nvidia-smi in the container (modal container exec <id> -- nvidia-smi): L40S 75% busy, 4 of 46 GB, but
  nproc = 1 CPU for training + 8 rollout workers. Set cpu=8 on job and clips for the next runs.
- 19j (clip pretraining as 19 ('all', 20k, B 32), but half the clips start in the decision window: first third of
  the steps before the grasp (cut step 14-47, median 29); ~57% of presents there vs ~12%; job clips --early 0.5;
  Modal H100, 8 CPUs, GPU 96% busy): locate 50.5 (0), 63.2, 69.7, **73.2** (12k), 72.5, **71.7%** (20k) (19: 55%,
  change head 68%, erase stage 2 76-77%); whole-frame 92% (19: 81.5%). Caption used more (τ 0.5 right 0.31 /
  shuffled 0.68). Caveat: the probes test on reset scenes, the frames this mix trains on more. Next: stage 2 /
  co-training from these weights.
- 19j plain stage 2 (demos only from 19j's clip weights, local): **29/29**, wrong_cube 14/14, miss 5.3 cm, slopes 0.33 /
  0.48. After demos: whole-frame 83%, locate **56%** (from 72%): plain demo training wears the mark off (erase too:
  100 -> 77). Same arm as the other video-only starts (19: 25, 27; 19e 29).
- 19k paired co-training, 40k steps, early mix 0.5 in both batches, from 19j's weights (Modal), running. run.py eval
  (new: a checkpoint scored like a run's end, into <run>/at<N>/) on its step20000.pt, local: **21/20**, wrong_cube
  15/18, dropped 5/5, no_grasp 9/7, miss 6.1 cm, slopes 0.27 / 0.25; whole-frame 92%, locate 67.3%. Mid-schedule
  (lr not decayed), so noisier than a run ending at 20k. Training log: locate 69.5-69.7% through 32k; val action
  loss rising after ~30k (0.10 -> 0.155): overfitting the 140 episodes.
- 19k final (40k): locate **70.5%** (held: 69.5-70.5 from 12k on); whole-frame 93%. Rollouts **23/21**, wrong_cube
  18/19, no_grasp 8/7, miss 6.8 cm, slopes 0.20 / 0.28. Read: the highest video-only mark kept through action
  training, yet wrong cube no better than any video-only run (14-19). The mark at ~70% isn't what the arm uses, or
  isn't enough; erase's arms (77%) pick right 46/50. One seed.
- scripts/arm_use_check.py (val action loss at fixed τ, true vs another moment's frame vs the colour word changed;
  decision-window moments and uniform ones). Decision moments, gap at τ 0 / 0.4 / 0.67:
  frame:  erase 0.075 / 0.188 / 0.419; 19k 0.035 / 0.096 / 0.290; 19j plain 0.033 / 0.087 / 0.215; none 0.020 / 0.045 / 0.078
  colour: erase 0.047 / 0.126 / 0.267; 19k 0.018 / 0.045 / 0.134; 19j plain 0.013 / 0.029 / 0.077; none 0.002 / 0.005 / 0.004
  Read: the arm's use of the colour word at decision moments follows the mark (erase > 19k > 19j > none) but even
  19k's arm leans on it ~2.5-3x less than erase's, and on the frame ~2x less. 19k's arm also fits decision moments
  worst (true loss τ 0.67: 0.353 vs erase 0.195): the 40k run overfit (val action loss 0.10 -> 0.12-0.165).

## Step 20 · roadmap box 06: actions as hand moves relative to the hand at the chunk's start (+ progress), IK outside
- Decisions (user): relative to the chunk start (Verne), keep progress (t / (len − 1)), compare on the erase recipe.
- 20a (hand.py: forward kinematics of the T5 arm on a private MjData, relative <-> absolute; scripts/hand_check.py):
  FK vs the sim's grip site 0.11 mm max; hand moves median 6.4 mm per control step (max 25 mm), 7.8 cm median 15
  steps ahead (max 24 cm); relative -> absolute round trip exact. One IK step (the expert's: damping 0.15, max
  0.06 rad/joint) toward the recorded target's hand pose undershoots (5.7 mm median, 9.6 mm 95%); iterating
  closes it: 3 steps 0.5 mm, 10 steps 0.00 mm (or damping 0.05, 3 steps: 0.00 mm). Joints then differ from the
  recorded ones by 0.2 deg median, 2 deg 95% (the 7-joint arm reaches the same hand pose with other joints).
  Run time: iterate IK (10 steps, damping 0.15) on a private MjData per executed action.
- 20b (data.targets / chunk_rows: 11 numbers per chunk step, from cached FK; model action in/out 11; rollouts and
  dagger: hand.Driver = chunk -> world poses at its start -> 10-step IK per executed step from the current joints).
  scripts/hand_replay_check.py + a diagnosis: replaying the recorded joints 19/20; world hand poses through the
  Driver's IK **8/8** (hand within 15 mm of the recorded path); the same moves relative to the ACTUAL hand, open loop,
  0/8 (drift 8-26 cm): the demos were recorded with joint noise (sigma 0.05), so each relative label holds a
  correction for that step's nudge, which open loop (no nudge, no anchor) piles up. Closed loop that correction is
  the recovery signal the noise was for. Conversion OK.
- 20c seed 0 (step 18 recipe: init 18_rope/pretrain/seed0, colour-only demos, 20k; hand-move actions + IK): **41/40**,
  wrong_cube 5/3, no_grasp 3/6, dropped 1/1, miss 2.1 cm, slopes **0.76 / 0.80** (18 s0: 0.72 / 0.58); locate after
  demos 75.5%, whole-frame 94.5%. = step 18 (41/40, 42/40, 39/39). Val loss 0.18 -> 0.26 (different units than
  joints; rising after ~10k).
- 20c seed 0, ckpt step10000 (run.py eval; val loss lowest there): **39/40**, wrong_cube 4/4, no_grasp 6/4, miss 2.2 cm,
  slopes 0.74 / 0.67, locate 76.7%. = the 20k end (41/40): the rising val loss after 10k didn't cost rollouts.
- 20c seed 1 (local): **43/42**, wrong_cube 4/6, no_grasp 2/1, dropped 1/1, miss 1.6 cm, slopes 0.85 / 0.68, locate 83%.
  Seed 2 on Modal (L40S, cpu=8) was at step 10k when seed 1 had finished locally: stopped, rerun locally.
- 20c seed 2 (local): **48/47**, wrong_cube 0/1, no_grasp 1/1, dropped 1/1, miss **0.9 cm**, slopes 0.96 / 0.71, locate 76.5%.
- Box 06 on 3 seeds: **41/40, 43/42, 48/47** (step 18 joints: 41/40, 42/40, 39/39); miss 2.1 / 1.6 / 0.9 cm (18: 2.0-3.4).
  Read: hand moves + IK are at least as good, likely better (mean 44 vs 40.7 at n_exec 8; closer closes). Done.

## Step 21a · erase co-training: step 15's erase loss kept on during demos (weight 1.0, B 64 reset scenes), runs/21a_erase_cotrain
- Step 20 recipe (init 18_rope/pretrain/seed0, colour-only demos, hand moves + IK, 20k, local 4060) + job --erase 1.0.
  Step 15's open item (demo training wore the mark 100 -> 76-83%).
- Seed 0: **49/49 of 50**, wrong_cube **0/0**, failures no_grasp 1 / dropped 1; miss **0.5 cm**, slopes 0.99 / 0.75
  (step 20 s0: 41/40, wrong 5/3, miss 2.1 cm). Locate 100% all run (step 20: 75.5-83% after demos), whole-frame 100%.
  Erase loss 0.0057 (1k) -> 0.0025 (20k, below stage 1's 0.0035). Val action 0.21-0.24 (step 20 ~0.18-0.26). One seed.
- Rollout workers OOM'd on the 8 GB card (VS Code GPU process + job + 8 workers); unfinished episodes ran in the main
  process (6 / 21 / 49 of 50), same seeds, so results are complete.
- Seed 1 (Modal L40S, eager): **44/48**, wrong_cube 0/0, dropped 4/2, no_grasp 2/0, miss 0.6 cm, slopes 0.97 / 0.73.
- Seed 2 (Modal L40S, compiled): **43/50**, wrong_cube 0/0, no_grasp 5/0, dropped 2/0, miss 0.6 cm, slopes 0.97 / 0.78.
- Seed 4 (local, compiled; a speed check vs seed 0): **47/49**, wrong_cube 0/0, dropped 2/1, no_grasp 1/0, miss 0.5 cm.
  One worker hung in EGL render (faulthandler dump, 5 episodes rerun in-process); compiled main process holds
  ~1.2 GB + 8 workers ~0.7 GB each on the 8 GB card.
- **21a on 4 seeds: 49/49, 44/48, 43/50, 47/49; wrong_cube 0 in all 8 sweeps (400 rollouts)** (step 20: 9 / 10 of 150;
  41/40, 43/42, 48/47). Locate 100% held in every seed. Left failures = grasping (dropped, no_grasp).
- torch.compile (mode reduce-overhead: CUDA graphs) on the per-step losses (train.py): s / 1000 steps, from ckpt
  times (alternating values = probe every 4k): local 4060 eager 25-33 -> compiled 15-23 (517 -> 334 s, 1.55x);
  Modal L40S eager 35-38 -> **6-10** (651 -> 142 s, **4.6x**; now ~2x faster than local). Step-0 losses match
  eager (same draws). Modal image + gcc (triton). Modal jobs on L40S only.
