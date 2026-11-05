# Pick different shapes -- a motion-level policy, verb 1 = pick

Question: can one learned policy pick objects of shapes it never trained on? First verb of a
motion-instruction executor (LLM says "rotate lid left", policy does it). Charter discussion 2026-11-05.

Plan (stages):
1. One object, no instruction. Tests grasping new shapes only.
2. Several objects + a point (u, v) in the top view, as a blurred-dot image channel. Tests listening:
   point at the far one, count right-object picks.
3. Maybe language.

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
