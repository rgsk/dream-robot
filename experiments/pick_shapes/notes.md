# Pick different shapes -- a motion-level policy, verb 1 = pick

Question: can one learned policy pick objects of shapes it never trained on? First verb of a
motion-instruction executor (LLM says "rotate lid left", policy does it). Charter discussion 2026-11-05.

Plan (stages):
1. One object, no instruction. Tests grasping new shapes only.
2. Several objects + a point (u, v) in the top view, as a blurred-dot image channel. Tests listening:
   point at the far one, count right-object picks.
3. Maybe language.

Hypotheses from T5 (experiments/t5/notes.md), to check results against (2026-11-05):
- **Stage 2 (pick the specified object), point not name.** T5 *was* "pick the specified object":
  by colour it sat at chance (SmolVLA, FM + clean one-hot, FM + FiLM); by side (left/right) it
  was obeyed every time. The spatial-softmax encoder keeps *where*, drops *what colour*. A
  blurred dot at the target pixel is a "where" instruction, and a bright spot is exactly what
  spatial softmax reads -> predict: obeyed like side, not chance like colour. If it lands at
  chance, the T5 explanation was incomplete. Test: 2+ objects, dot on the far one, right-object rate.
- **Verbs (pick + lift lid in one model, verb one-hot).** Pick and lift-the-lid are different
  motions, not a subtle visual difference between two similar targets. The instruction that
  worked in T5 was the one that changed where and how to move -> a verb input should be easier
  than the colour word was. Risk: a scene where both verbs fit (pot with lid). First check there:
  same start frame, swap the verb, does the plan change (T5's planned condition-swap, never run).
- Order per new verb: specialist first (reference), then joint model, compare each verb to its
  specialist. Grow the model only if the joint one underfits. Pick bar before lid: rung 2 ~80%+.

Stage 1 as rungs (each ends in a video + number; next rung only when this one works):
- 0: one apple mesh, test new positions (pipeline base case)
- 1: some apple meshes, test other apple meshes
- 2: several spheres, test tomato / onion (new category, seen shape)
- 3: sphere + cylinder + box (+ irregular), test long_thin / handled (new shape family)
Scripted picker grows with the rungs: apples first, other families when a rung needs them.
Policies: our FM (chunks, pretrained ResNet18) first -- grasps are multimodal; ACT alongside as baseline.
Demos: scripted top-down picker using sim state (data only, never drives at test time).

## Step 0 · RoboCasa inventory + contact sheet (2026-11-05)
- 198 object categories, 157 graspable, ~2,500 meshes (10-30 per category). No pick-only task;
  pick is inside the PnP tasks. Our task: "pick up the X", success = lifted ~10 cm and held.
- envs/robocasa/src/{robocasa,robosuite} had all .py deleted in the working copy (876 + 233
  files); restored with `git checkout -- .`, same commits. envs/cosmos_policy, envs/vpp2 have
  the same deletions -- left alone (parked).
- Six families (sphere, cylinder, box, long_thin, handled_hollow, irregular_soft), 4 categories
  each: `contact_sheet.png` (scripts/contact_sheet.py, standalone MJCF, 5 cm checker).
  Category scales not applied (wooden_spoon renders huge); in-kitchen render will apply them.

## Step 1 · rung 0 scripted picker (2026-11-05)
- Own task `PickOne` (one object, counter, no distractors), layout/style (1, 1), no robot spawn
  jitter, mesh pinned to objaverse/apple_0. Recipe: 15 cm above -> centre -> close -> lift 15 cm.
  Sim-state aiming. scripts/rung0_scripted_pick.py, videos rung0/seed*.mp4 (scene | wrist).
- **4/5 lifted** (seeds 0,1,3,4: 14.3-14.9 cm, holding). Seed 2: closed, not holding, 0.2 cm --
  apple sits next to an appliance and ends up beside the fingers (squeezed out or nudged).
- The seed also picks *which counter* (apple x 0.58 .. 3.79 m): more scene change than "positions
  only". ~60-70 control steps per pick (3-3.5 s).
- Seed 2 (user, from the video): the appliance hit the apple and it fell. Scene, not grasp --
  skipped; demos keep successful episodes only.
- Counter pinned (user: yes): layout 1 main counter, long clear top (nearest the stove).
  Robot base fixed at (2.23, -0.80); apple relative to base x -0.11..+0.16, y 0.24..0.42 m.
- **20/20 lifted** (seeds 0-19, 14.2-15.0 cm, all holding). Seed 7: 'down' stalled on the
  counter, still grasped. rung0/run20.log, rung0/seed*.mp4.

## Step 2 · demo recording with touch (2026-11-05)
- Touch = 8 numbers: normal force per finger (summed contacts) + wrist F/T (robosuite's
  force_ee/torque_ee, tared at step 0). Recorded now; policies train with and without it (user).
- robocasa venv's LeRobot is 0.3.3 (old format), main is 0.5.1 (v3.0) -> rung0_record.py writes
  raw .npz, rung0_to_lerobot.py (main venv) writes the dataset. Smoke: 3 kept of 4 (seed 1002
  'down' blocked, missed), 210 frames, loads back in main venv.
- touch_seed1000.png: **left finger hits the apple at step 36, ~6 steps before the close** (21 N
  spike, wrist fz 20 N) -- the descent bumps it; the video doesn't show it. Holding: ~20 N each finger.
- Wrist F/T does not see the apple's weight (~0 after lift): sim apple is ~20 g (density 100).
  A real apple is ~150-200 g. Sim artifact to remember for transfer.
- Descent bump fixed: align above to 4 mm, then straight down in 2 cm waypoints. bump_N (finger
  force while still commanded open) 0.0 on every episode; picks ~90 steps (was ~70).
- Cost: the straight descent stalls near the robot base. **30 kept of 36**; all 6 drops have the
  apple at y < -0.57 (nearest 2-3 cm band). Demos never cover that band -> report test seeds
  there separately. data/robocasa/pick_apple_rung0_n30 (2.6k frames), log rung0/record30.log.
- Front camera loses the apple behind the arm from the descent on (demo_videos/grid4.mp4).
  Wrist sees the apple at step 0 in 30/30 demos (~300 red px; front ~900). -> first run is
  **FM on wrist + touch only** (user); front stays in the data for later.

## Step 3 · FM wrist + touch, 30 demos (2026-11-05)
- Added `--cameras` and `--extra-state` to FM training (touch appended to state, saved in config).
- 100 epochs, 5 min, 5.3M params. Held-out first-action error 292 -> 23 mrad, still falling.
- **Kitchen test: 0/20** (seeds 2000-2019). Apple never touched in any episode.
- Seed 2000 trace: hand drifts down toward the apple but stays 3-9 cm off in x, gets to 3 cm
  above centre, never closes, then drifts. Test apple is +5.8 cm in x from the demos' mean
  apple; the hand error is -3..-6 cm in x -> it goes part-way toward the *average* demo spot,
  i.e. under-uses the image. Wrist view loses the apple once off-centre (eval/seed2000.mp4).
- Candidates: state shortcut (joints predict the next action in 25 near-identical demos), wrist
  view too small at start (~17 px apple), too little position diversity / training.
- Image-swap check, seed 2000, hand-apple at step 80: real 3 cm / frozen other-seed frame 11 cm /
  blank 18 cm -> the policy *does* use the camera; it was imprecise, not blind.

## Step 4 · same, 300 epochs (2026-11-05)
- 14.6 min. Held-out action error 11.1 mrad (100 ep: 23). Val loss V-shaped, min ~ep 210
  (0.047 -> 0.054 at 300) while action error kept falling: FM val loss is a weak proxy.
  Only the last epoch is saved -> add --save-every to test epochs in the kitchen.
- **Kitchen test: 16/20** (100 ep: 0/20), same seeds 2000-2019. Training length was the fix.
- Fails 2006, 2008, 2011, 2017: three at the edges of the demo spread (2006 near-base band
  y -0.584, never in demos; 2008 / 2011 far right x 2.38 + near side y -0.55); 2017 central.
- One training seed only.

## Step 5 · rung 1 zero-shot: rung-0 policy on unseen apple meshes (2026-11-05)
- Split fixed: test = objaverse apple_1..10, training pool for later = apple_11..22 (+ apple_0).
  apples_rung1.png. Same seeds 2000-2019, mesh i % 10. No retraining.
- **12/20** (apple_0 on same seeds: 16/20). Seeds 2006, 2008, 2011 failed with apple_0 too
  (position) -> on the 16 seeds apple_0 passed: **11/16**.
- Mesh-attributable fails: apple_6 0/2, apple_10 0/2, apple_3 1/2. apple_10 and apple_3 are the
  narrowest (5.5-5.8 cm vs apple_0 6.8). apple_6 is the same size and look as apple_4 (2/2), so
  2 episodes per mesh can't separate colour / size / chance.

## Step 6 · thesis test setup: camera-only UWM, video-only clips (2026-11-05)
- Goal (user): the gripper policy improves from *any* video, ultimately the user's own hand used
  as a 2-finger gripper. Rung 2 on hold. Approach 4, joint training (UWM): one transformer,
  frame-t+16 head trained by all clips, action head by labelled ones only.
- Camera-only from the start (user): wrist frames t-12/-8/-4/t at 128 px, no joints, no touch.
  Arm action = change from q[t] over the 16-step chunk; run time adds q[t] back. Gripper = command.
- Video-only episodes: action tokens pure noise (tau_a = 0), action loss masked.
- scripts/uwm_cam.py (train, main venv; 1.45M params, ~23 s / 500 steps) and
  scripts/uwm_cam_eval.py (kitchen, robocasa venv). Split fixed: 5 val eps, then labelled, then
  video-only, from the same 30 rung-0 demos (same robot, actions dropped = mechanism test).
- Bar: untrained net (says the average change) scores 24.9 mrad first-action error.
- Smoke: 500 steps 23.7 mrad; eval runs end to end (0/1, as expected at 500 steps).
- Plan: anchor l25_v0 (10k steps, save every 2.5k) vs FM joints+touch 16/20. Then the curve:
  labelled 3 / 10 with and without the remaining train episodes as video-only.

## Step 7 · camera-only anchor, 25 labelled, no video-only (2026-11-05)
- l25_v0_s0 (flat lr 1e-3, no EMA, 10k): first-action 16.0 mrad, **1/20**. Wrist view shows the
  hand tilting (brick wall in view) and losing the apple; joint changes drift with no anchor.
  The one success (2005) needs that run's exact noise; alone it fails.
- Added cosine lr (1e-3 -> 1e-5) + EMA 0.999 (validated and saved). l25_v0_s0_30k:
  step 4000 already 10.6 mrad (flat run 26.8 there) -- the EMA alone did most of it.

| step | first-action mrad | val action | kitchen |
|---|---|---|---|
| 5000 | ~9 | ~0.05 | 3/20 |
| 15000 | ~5 | ~0.04 | **10/20** |
| 20000 | 4.9 | 0.045 | **10/20** |
| 30000 | 4.9 | 0.056 | 9/20 |

- Plateau ~10/20 from 15k to 30k (joints+touch FM: 16/20); 9/20 at 30k as val action climbs.
  Seeds 2006, 2012, 2013, 2015, 2017 fail at every checkpoint -> the anchor's hard set.
- Anchor for the curve: **20k steps** (peak, before most of the memorising). Val action rises from ~10k (0.035 -> 0.048 at 26k)
  while train action -> 0.002: memorising 25 demos.
- Successes **over-lift** 13-48 cm (demos 15): no sense of "lifted enough". 2/40 knocked the apple
  off the counter (-0.92 m).
- **Unprompted recovery** (step20000/seed2001.mp4): grasp misses, hand rises, then re-descends
  and re-centres on the apple, twice. No demo has a retry (failed picker episodes were dropped).
  Camera-only acts on what it sees: "apple below, empty fingers" = approach again.

## Step 8 · the video comparison, camera-only UWM, 20k steps (2026-11-05)
| run | labelled | video-only | 10k | 20k |
|---|---|---|---|---|
| l25_v0_s0_30k | 25 | 0 | -- | 10/20 |
| l10_v0_s0 | 10 | 0 | 3/20 | **6/20** |
| l10_v15_s0 (unbalanced) | 10 | 15 | -- (5k: 0/20) | **0/20** |
| l10_v15bal_s0 (balanced) | 10 | 15 | 2/20 (5k 2/20, 15k 2/20) | **5/20** |
- l10_v0: val action climbs to 0.27 (l25: ~0.045) -- heavy memorising with 10 demos.
- Fairness issue (user): random batches over all train rows -> in l10_v15 only ~26 of 64 rows
  carry actions, 2.5x fewer action examples than l10_v0. A null result there is ambiguous.
  Balanced version: each step 64 labelled rows (as l10_v0) + 64 video-only rows (frame loss only).
- l10_v15 (unbalanced) 20k: **0/20**, apple never touched. Arm rises and swings toward the wall,
  never approaches (step20000/seed2000.mp4). Training end: first-action 16.6 mrad (l10_v0: 6.5),
  val action 0.109 (l10_v0: 0.276) -- lower val loss, far worse policy. Video-only rows hurt here,
  confounded by 2.5x fewer action examples. Balanced run (l10_v15bal_s0) training.
- Balanced (64 labelled + 64 video-only rows per step): action side matched l10_v0 (first-action
  8.4 vs 8.0 mrad at 5k), val action ~10% lower at every step, but kitchen **5/20 vs 6/20**.
  -> pixel-target video clips don't help here; the unbalanced 0/20 was action starvation.
  Successes lift 20-27 cm (l10_v0: 13-51) -- maybe steadier, small sample. One seed each.
- Parallel eval (5 workers, net on CPU): 266 s with training on the GPU, 248 s without -- no
  faster than serial; each worker ~5x slower per episode. CPU-side contention, not profiled yet.
- Next for the thesis: frozen-feature video target (DINOv2 features of frame t+16 instead of pixels).
- Parallel eval fixed: the slowness was the net on CPU (43 s per rollout vs 3.5 s on GPU), not
  contention. GPU workers need ~1.34 GB each -> 3 fit on the 8 GB 4060. **3 workers: 56 s for 20
  seeds** (serial ~4-5 min). Defaults now --device cuda --workers 3; per-episode build_s / run_s logged.
- Noise check from the timing runs: l25_v0_s0_30k step20000 with per-episode noise seeds = **7/20**
  (serial-noise run: 10/20). Same weights, same seeds, only the plan noise differs -> a 20-episode
  score moves by ~3 from sampling noise alone. Differences like 5 vs 6/20 are inside that.

## Step 9 · re-score at 60 seeds (2000-2059), step 20000, per-episode noise seeds (2026-11-05)
| run | labelled | video-only | 20 seeds | **60 seeds** |
|---|---|---|---|---|
| l25_v0_s0_30k | 25 | 0 | 10/20, 7/20 | **18/60 (30%)** |
| l10_v0_s0 | 10 | 0 | 6/20 | **9/60 (15%)** |
| l10_v15_s0 (unbalanced) | 10 | 15 | 0/20 | **0/60** |
| l10_v15bal_s0 (balanced) | 10 | 15 | 5/20 | **4/60 (7%)** |
- Halving labels (25 -> 10) halves success (30% -> 15%). Pixel-target video does not win it back;
  balanced is *below* no-video (4 vs 9 of 60; ~1.7 binomial sd, suggestive not certain).
- Pixel video target: no help, maybe mild harm. Next: frozen DINOv2 feature target (plan in chat),
  then frozen-feature inputs (the full video-features thesis) if the target alone doesn't move it.

## Step 10 · variant B, step 1: frozen DINOv2 features cached (2026-11-05)
Plan (B): wrist frames -> frozen DINOv2-S -> features NOW -> predictor -> features at t+16;
action head reads NOW + predicted FUTURE -> 16-step chunk. Trained in sequence: predictor on all
clips (labelled + video-only), then the head on labelled only with the predictor frozen. Arms at
10 labelled: A (NOW only) / B (predictor on 10) / B+video (predictor on 10 + 15).
- [dino_cache.py](scripts/dino_cache.py): raw lossless 256 px -> 224 -> 16x16x384 + CLS,
  averaged 2x2 -> 8x8x384. 2715 frames in 7 s on the 4060 (model load 17 s first time).
  Output data/robocasa/pick_apple_rung0_n30_dino.npz (136 MB, f16), same row order as uwm_cam.load.
- Apple check ([features.png](dino/features.png)): the apple is its own PCA colour at 16x16 and
  still at 8x8; similarity to the apple cell lights the apple and nothing else from step 18 on.
  At step 0 (apple ~1 patch, far away) it is faint at 8x8. Acceptable to start; 16x16 is the fallback.
- Arm A ([uwm_dino.py](scripts/uwm_dino.py), 10 labelled, 0.87M params, 6.4 min): train loss -> ~0,
  val action min 0.114 at 5k, 0.138 at 20k; first-action 5.1 mrad (pixel l10: ~8).
  Kitchen: step 10000 **5/20**; step 20000 **4/60** (pixel l10_v0: 9/60). 56/60 never move the apple.
  Better first-action, worse kitchen -> memorising the 10 demos; fix A before building B on it.
- Arm A by checkpoint, 60 seeds: step 5000 **9/60**, 10000 **7/60**, 20000 **4/60**. Never-touched
  failures 42 -> 50 -> 56. Longer training = worse (trend, each step inside noise). At its best
  checkpoint A ties pixel l10_v0 (9/60): frozen DINOv2 input neither helps nor hurts at 10 demos.
- Predictor ([dino_predict.py](scripts/dino_predict.py), 10 labelled clips, 3.4M params): val/copy
  best 0.454 at 3k (saved best: step 5000, 0.458); train 0.04 vs val 0.39 -> overfits. Picture
  ([ep2](uwm_dino/pred_l10_v0/predicted_step5000_ep2.png)): apple's future position right, counter
  blurred to an average. Stopped at 17.5k; second (B+video) predictor skipped for now.
- Arm B (head reads now + predicted t+16, predictor frozen): val best at 5k (0.160; A 0.114).
  Future replaced by "now" -> val ~12% worse at every step -> **the head uses the future**.
  Kitchen step 5000: **5/20** (A step 5000 on the same seeds: 5/20; only 2 seeds shared).
  Caveat: predictor near-perfect on its own train clips (0.04) vs 0.39 at test -> head may over-trust it.
- Arm B step 5000, 60 seeds: **10/60** vs A step 5000 **9/60** (pixel l10_v0 9/60). Tie, inside noise.
  Different policies: 4 seeds won by both, 6 only B, 5 only A. Never-touched 40 (A 42).
  -> with a predictor trained on the same 10 clips, a predicted future doesn't add kitchen wins.
- **B+video predictor** (10 labelled + 15 video-only clips): val/copy best **0.403** at 3k, **0.410**
  at 5k (B's predictor: 0.454 / 0.458); lower at every step, train 0.061 vs 0.037 at 5k -> overfits
  later. **Video without actions makes the imagined future ~11% better.** Stopped at ~6k.
- B+video head (same as B, reads the 10+15 predictor's step 5000): val best 0.161 at 4-5k, then
  rises (stopped ~11k); no-future still ~13% worse. Kitchen step 5000: **7/20** (A 5/20, B 5/20,
  same seeds). Wins all 5 of B's seeds + 2011, 2013 (A's). Inside noise -> needs 60 seeds.
- **B+video step 5000, 60 seeds: 13/60** (B 10/60, A 9/60, pixel l10_v0 9/60). Never-touched 32
  (B 40, A 42). vs B: 7 seeds both, 6 only B+video, 3 only B (sign test on 6 vs 3: p~0.5).
  Direction matches the predictor (video -> 11% better future -> more lifts, fewer whiffs), but
  one training seed and +3/60 is not proof. Next: training-seed repeats and/or more video clips.

| arm (10 labelled, step 5000) | predictor val/copy | 60 seeds | never touched |
|---|---|---|---|
| A: now only | - | 9/60 | 42 |
| B: now + future, predictor on 10 clips | 0.458 | 10/60 | 40 |
| B+video: now + future, predictor on 10 + 15 video | 0.410 | **13/60** | **32** |
