# UVA on T5 -- video + actions in training, actions only at run time

Question: a small video-action model whose run-time path never draws or carries video -- does it
pick the named cube on held-out pairs (Cosmos Policy 27/30, SmolVLA 11/60), at Cosmos-beating
speed? Parked behind UWM (user, 2026-10-21): UWM first, because its video-drop variant decides
whether we bet on UWM; UVA is the ready-made version of that run-time design to compare against.

## Step 0 · read from the code (2026-10-21)
- Repo `ShuangLI59/unified_video_action` (MIT), cloned to envs/uva/src (gitignored), commit
  fe3dec8. Built on MAR (masked autoregressive image generation, `mar_base`) + a KL-16 VAE.
- LIBERO-10 setup (`config/task/libero10.yaml`, `config/model/uva.yaml`):
  - **instruction wired**: CLIP text embedding (`language_emb_model: clip`), multi-task;
  - current scene as **patches**: 256 VAE tokens per frame, 4 past frames as condition (as
    Cosmos sees it, unlike UWM's one vector per image);
  - **top camera only** (agentview 128 px, resized 256); a `predict_wrist_img` option exists;
  - actions 10-D (abs position + 6-D rotation + gripper), horizon 32, 8 executed;
  - action diffusion head (MLP, 6 x 1024), 100 sampling steps; actions masked in training at
    ratio 0.5 (no released path for action-free datasets);
  - run time: one backbone pass, then the action head; video not generated (~95 ms per 16
    actions, paper).
- Checkpoints on Google Drive: PushT, PushT-M, LIBERO-10, UMI multitask; MAR + VAE pretrained
  weights via their download script. Training: "at least 4 GPUs" for video + action.

Compared with UWM (experiments/uwm/notes.md step 1):
| | UWM | UVA |
|---|---|---|
| instruction in LIBERO setup | no (we wire CLIP) | yes (CLIP) |
| current scene | 1 vector per image (AdaLN) | 256 patches per frame |
| video at run time | in the input as noise (fix: drop it) | skipped by design |
| cameras | top + wrist | top only |
| size | ~100M trainable | MAR-base backbone, bigger |
| action-free video | built in | not released |

Ladder (each stage asked first):
| stage | what |
|---|---|
| A | LIBERO-10 checkpoint loads; memory + seconds per call on the 4060 |
| B | fine-tune on the 150 T5 demos (top camera, 8-D joints), ~10 eval episodes |
| C | 30 seen + 30 held-out (Cosmos 27 / 27) |

## Step 1 · stage A setup (2026-10-26)
- Clone had 181 tracked .py files deleted from the working tree (also uwm 54, cosmos_policy 498,
  vpp2 94; cause unknown); uva + uwm restored with `git checkout -- .`, the other two left for later.
- Env `envs/uva/.venv`: Python 3.9, their conda pins as pip (torch 2.1.0 cu121; av 10.0 has no
  wheel -> 12.3.0). `scripts/setup.sh`, 33 min at ~3-5 MB/s.
- Downloaded: libero10.ckpt 1.9 GB (Drive), KL-16 VAE 266 MB + MAR-base 1.6 GB (Dropbox, their
  download.py). `scripts/stage_a.py` written (random 0..255 frames; seconds per call + peak memory).

## Step 2 · stage A: LIBERO-10 checkpoint on the 4060 (2026-10-26)
- Loads clean (EMA weights from the checkpoint; VAE missing 0 / unexpected 0). 247M MAR + 67M VAE
  (frozen) + 151M CLIP (frozen). Run-time config: 16 obs steps -> 4 frames, 100 action-diffusion
  steps, num_iter 1, chunk 16 x 10-D, 8 executed.
- `scripts/stage_a.py`, batch 1, fp32, random frames (speed only): **0.296 s per call** (first 0.53),
  peak GPU 2.4 GB. vs UWM 0.094 (policy) / 0.065 (dropped) on the same GPU; paper's ~95 ms was on
  a bigger GPU. Not yet split into VAE / CLIP / transformer / action head.
- Split of the 0.30 s call: action head (100 diffusion steps) 0.133 (45%), VAE encode of the 4
  past frames 0.081 (27%), MAR encoder 0.038 + decoder 0.041 (27%), CLIP 0.003.
  -> the transformer alone (0.08) ~ UWM's whole call. Easy cuts, untested for quality: 10-step
  action sampling (~0.013), encode only the new frame and reuse 3 cached latents (~0.02) -> ~0.12 s.

## Decision · stage B setup (2026-10-26)
- Easy task first: pick_place_cube (25 demos, BC/ACT already succeed), so a failure is a hookup
  bug, not the colour question. T5 after it works.
- Arm state: feed the robot's joint readings (8-D `observation.state`) through a new input layer,
  not past actions. On a real robot commands and readings differ (gripper stalls on the object,
  contact, lag); readings carry the grasp. Same choice as ACT / DP / pi0.

## Step 3 · stage B data: pick_place_cube in UVA's loader (2026-10-26)
- `scripts/to_hdf5.py`: 25 demos -> data/uva/pick_place_cube (one file, "put the cube in the bin");
  raw top + wrist frames, `obs/joint_states` (8), actions (8).
- Their loader flips every frame vertically (LIBERO renders are upside down); ours are upright,
  so the converter stores them flipped. Checked: loader output upright, bin on the right
  (data_check.png). At run time: feed upright frames, no flip.
- Their loader also colour-jitters (hue 0.05): fine here, revisit for T5 (colour is the task).
- pytorch3d (rotation conversion, abs_action only) replaced by a stub (`scripts/stubs`); +h5py,
  tensorboard missing from setup, added.
- Through their loader: 4,713 windows of 32 steps; frames (32,3,128,128) in [0,1], joint_states
  (32,8), language tokens (32,2,30), action (32,8).

## Step 4 · stage B model: joints in, 8-D actions out (2026-10-26)
- `scripts/uva_patch.py`: joint readings -> Linear(8, 768), zero init, added per frame (the 4
  past frames' readings, each over its 256 positions) after their per-position projection. Also
  scales actions to [-1, 1]: with abs_action false their normalizer is identity (ours are radians,
  -2.6..3.0). Joint key renamed `joint_qpos` (their normalizer rejects other names).
  `scripts/task_cfg.py`: the checkpoint's own cfg, re-pointed (task name keeps "libero").
- `scripts/check_model.py`: from LIBERO-10, only 4 tensors not loaded (action_proj_cond,
  action head input / output layers, all 10-D -> 8-D). Scaled action and joints in [-1, 1].
  Zero joint layer: two different readings, same seed -> actions identical (0.0).
- One fp16 AdamW step: batch 2 fits (5.5 GB peak, 0.67 s); batch 4 out of memory on the 4060.
  -> training goes on a pod (their batch was 32). That step drew a video mode (action loss 0);
  action loss in policy mode still to see.
- Their live policy expects pixels in [0, 1] (stage_a fed 0..255; timing unaffected, fixed).

## Step 5 · stage B training script + pod (2026-10-26)
- `scripts/train_uva.py`: plain one-GPU loop (bf16, cosine LR, their EMA), all 5 training modes
  as theirs; val_action_err = EMA policy as at run time on 2 held-out demos (radians, mean over
  the 16-action chunk); checkpoint = EMA + normalizer + cfg, pushed to rgsk/dream-robot-checkpoints.
- Local smoke impossible: batch 1 OOMs on the 4060 once the EMA copy + AdamW state are on it.
- Data to the pod as the loader's own zarr cache (26 MB, rgsk/dream-robot-data uva/); the
  497 MB HDF5 never leaves home. `scripts/pod_setup.sh`.
- Pod: secure A100 SXM 80 GB, EU-RO-1, $1.79/h, 32 vCPU (US-KS-2 had none left), id udjqd6iodyszxj,
  created 18:14 IST.
- Pod smoke (batch 32): 39.6 GB. First ~7 s/step, GPU at 0%: the loader. Their __getitem__ calls
  threadpool_limits(1) per sample = ~1,750 stat() calls, 1.8 s/sample on /workspace
  (`scripts/profile_loader.py`; JPEG2000 decode and colour jitter were NOT it). Made it a no-op,
  one thread per worker instead; also decode frames into RAM once (13 s) and jitter off.
  -> **1.46 s/step**, data wait ~0. Losses move, val_action_err 0.23 rad at step 30, checkpoint
  saves (1.9 GB, EMA incl. frozen VAE + CLIP).
- Pitfall again: `pgrep -f` / grep on a pattern inside the ssh command matches that ssh's own
  shell (killed it once, hid a finished job once). Match on `[t]rain_uva.py`, launch separately.
- Run b1 launched 19:18 IST: 5,000 steps x 32, warmup 500, val every 500, checkpoint every
  1,000 to rgsk/dream-robot-checkpoints uva/b1/. Log mirrored to pod_logs/b1.log.

## Step 6 · eval hookup (2026-10-26)
- `scripts/serve.py` (UVA env): loads train_uva's model.pt (their model from the saved cfg, our
  joint input, EMA weights, saved normalizer), answers 16 actions per call.
- `dream_robot/policies/uva` + registry `uva`: keeps the last 16 top frames + joint readings every
  step (first frame repeated at episode start), asks every 8 steps (their n_action_steps).
- Eval can run locally (2.4 GB, ~0.3 s per call on the 4060, one call per 8 steps) -> pod only
  for training.

## Step 7 · first closed loop: b1 @ 1,000 steps (2026-10-26)
- b1 @1,000: action_loss 1.7 -> 0.08, val_action_err 0.23 -> 0.028 rad.
- Local eval (server on the 4060, `dream_robot.core.run --policy uva`), seeds 1000-1004:
  **3/5 success** (seeds 1000, 1002, 1004 in 198 / 209 / 205 steps), 2 NO_GRASP (1001, 1003). eval/b1_1000/ (results.json, videos).
  First success for a video-action model in this repo; the hookup (frames upright, joints,
  16-step window, 8 executed) works end to end.
- b1 @3,000 (val_action_err 0.016), 20 eval seeds 1000-1019: **10/20**, all 10 failures NO_GRASP.
  BC on the same seeds 5/20 (14 no_grasp); ACT walkthrough 136/140 with chunk ensembling (top +
  wrist). Lower offline error (0.028 -> 0.016) did not raise success (3/5 @1,000).
- Failure look (failures.mp4): the arm reaches toward the cube, closes short of / beside it, then
  carries on to the bin and hovers there empty -- the rest of the demo trajectory played
  without the cube. Not wrong-target: it misses the grasp position.
- Suspects: top camera only (UVA's LIBERO setup; the wrist view is what aligns a grasp, ACT
  had it), 23 train demos. To check: success vs cube position per seed.
- b1 stopped at step 4,050 (user: loss flattening; 1,000 -> 3,000 steps did not raise success);
  step-4,000 checkpoint on the hub (val_action_err ~0.015). Pod kept for T5.

## Decision · T5 next, with the wrist camera (2026-10-26)
- User: skip a pick_place_cube b2; go to T5 directly with the wrist camera. Failure modes keep
  the two changes apart: no_grasp = grasp still the problem, wrong_target = colour.
- Wrist path (`uva_patch.add_inputs(wrist=True)`): wrist frames at the 4 past steps -> their
  frozen VAE -> 16x16 patch codes -> Linear(16, 768), zero init, added per position after
  proj_cond_x_layer (= their own second-image fusion). Server / client carry a 16-step wrist window.
- T5 data: `to_hdf5.py` on instructed_pick_place_clean_n150_224 -> data/uva/t5 (150 demos, 6
  instruction files, both cameras at 224 px, 9.4 GB); kept at 224 so training and eval frames match.
- Eval video: `serve.py --imagine` adds UVA's expected top frame 16 steps ahead (their
  dynamic_model mode, given the chosen chunk) as the third column.
- b1 @4,000 eval cut short by a session restart: 7/14 seeds (1000-1013) succeeded; same picture.

## Step 8 · T5 + wrist: run c1 (2026-10-26)
- Smoke (30 steps x 32, --wrist, 224 px): frames into RAM 300 s (66k frames), 39.6 GB, ~1.85 s/step
  (vs 1.46 top-only); losses move; val_action_err 0.33 at step 30 (per-window instructions decoded
  from the tokens). 27,591 train windows, 2,344 val (12 of 150 demos held out).
- c1 launched 22:27 IST: 8,000 steps x 32 (~9 passes), warmup 500, val every 500, checkpoint every
  1,000 to the hub (uva/t5_c1/). ~4.2 h, ~$7.5. Log mirrored to pod_logs/t5_c1.log.

## Step 9 · b1 @4,000 with imagined frames (2026-10-26)
- 20 seeds: **13/20**, 7 NO_GRASP (1003, 1009-1011, 1013, 1014, 1016). Successes 196-222 steps.
  b1 @1,000 3/5, @3,000 10/20, @4,000 13/20.
- `serve.py --imagine` (top frame 16 steps ahead, forward mode given the chosen chunk) in the
  failure video (eval/b1_4000/failure_imagined_strip.png, one failure, top to bottom in time):
  it imagines the gripper coming down on the cube, and later imagines the arm lifting with the
  cube still on the table -- but then imagines a red cube in the gripper over the bin while also
  drawing the real one left behind: two cubes. The demos' "holding" prior wins over the scene.
- The real miss is small: in the top view the gripper is over the cube, in the wrist view the cube
  sits at the bottom edge, not between the fingers -- an alignment error the top camera can't
  resolve. Supports the wrist camera (c1).
- c1 @1,000 (val_action_err 0.034), local T5 eval, 10 + 10 episodes, seeds 1000-1009, --imagine:
  seen **0/10** (8 NO_GRASP, 2 wrong cube), held-out **1/10** (6 NO_GRASP, 2 wrong cube, 1 wrong bin).
  For scale: SmolVLA 22/60 + 11/60, UWM best 1/20, Cosmos 27/30 + 27/30. Early (1/8 of the run).
- Imagined frames (eval/c1_early/seen_failure_imagined_strip.png): the arm swings straight to the
  right bin with no cube, and the imagined frame shows it carrying a BLUE object (no blue cube in
  the scene) -- the LIBERO-10 video prior still in the video head at step 1,000. Cubes and bins are
  drawn right otherwise. Re-check at later checkpoints before judging.
- c1 @1,000 held-out seed 1001 (success, 384 steps): reached for the blue cube (imagined: gripper on
  blue), backed off empty (imagined: nothing held), came down on yellow (imagined: yellow held),
  placed it left. A retry the demos never show (expert always succeeds): re-planning every 8 steps
  from the current frames, not replaying a motion. b1 did the opposite (imagined holding, went on).

## Decision · c2 = wrist imagined too, if c1 @4,000 still fails by NO_GRASP (2026-10-26)
- c1 feeds the wrist camera as input only; its video head predicts the top camera alone, so no
  wrist frames can be imagined. b1's misses were wrist-visible misalignments, so a wrist video
  loss may be the better teacher for grasping. c2 = second video head on the future wrist frames
  (UVA has `predict_wrist_img`; ours would follow the wrist-input patch) + `future_wrist` in the eval.
  Needs the pod: c1 stopped at 4,000, or a second pod.
- c2 code built (`uva_patch.add_inputs(imagine_wrist=True)`, `train_uva.py --imagine-wrist`): a
  second video head (`diffloss_wrist`, starts as a copy of the pretrained top-camera head) on the
  future wrist frames' latents, same decoder output z and mask, video modes only; `serve.py
  --imagine` returns `future_wrist` too. `scripts/check_wrist_head.py` on the c1 checkpoint: only
  diffloss_wrist new; imagine gives both frames (the untrained head draws a top view, as expected:
  eval/c2_wrist_head_untrained.png); wrist loss ~1.5-1.7 in video / dynamic / full-dynamic modes
  with gradients into the head, 0 in policy mode; batch 1 peak 3.8 GB. Synced to the pod.
- **c1 stalled at step ~2,900 (23:14 IST)**: data_s 0.0003 -> 42.9 s/step. GPU fine; loader workers
  at 100% CPU. Host 94% idle, no steal, sy >> us. The same per-window numpy work (their
  __getitem__'s astype / rot90 / flip copies) took 5 ms locally and **1,763 ms on the pod**; with
  glibc told to reuse freed memory (MALLOC_MMAP_THRESHOLD_=33554432 MALLOC_TRIM_THRESHOLD_=17179869184
  MALLOC_TOP_PAD_=1073741824) **7 ms**. Fresh large allocations stall in the kernel on this host
  (fragmented after 6 days up, 1.5 TB page cache). Every pod launch sets those env vars from now on.
- c1 stopped at step ~3,000 after its checkpoint reached the hub (val_action_err 0.0224 @3,000).
  Hub keeps c1 @3,000 (uva/t5_c1/).

## Step 10 · c2: wrist in AND imagined (2026-10-27, 00:00 IST)
- `scripts/pod_train.sh` (sets the MALLOC_* fix, detached): NAME=t5_c2, --wrist --imagine-wrist,
  8,000 steps x 32, val every 500, checkpoint every 1,000 to uva/t5_c2/. Log pod_logs/t5_c2.log.
- c1 @3,000 eval running locally (10 seen + 10 held-out, imagined top frames) as c2's comparison.
- c1 was still improving when stopped: per 500 steps, action loss 0.077 / 0.060 / 0.054 / 0.048 /
  0.043 and val_action_err 0.034 / 0.028 / 0.025 / 0.024 / 0.022 (steps 1,000-3,000); lr at 75% of
  peak (cosine to 8,000). Full log: pod_logs/t5_c1.log.jsonl.
  -> the fair test of the wrist video head is **c2 @3,000 vs c1 @3,000** (same seeds); c2 beyond
  3,000 has no c1 partner (a full c1 rerun with the MALLOC fix would give one, if needed).
- c2 cancelled by the user (00:45 IST) to run tomorrow; pod udjqd6iodyszxj deleted (pod list empty).
  c2 decoded frames (352 s) and started, but logged no step in ~30 min: the MALLOC fix (250x on the
  benchmark) did not save the real run on that host. -> tomorrow: a FRESH pod, still via pod_train.sh;
  check s_per_step at step 50 before leaving it. Nothing lost: code + data local, data cache on
  rgsk/dream-robot-data uva/, checkpoints on the hub (b1 @4,000, c1 @3,000), logs in pod_logs/.
- Rerun c2: pod_setup.sh (+ T5 cache: hf_hub_download uva/t5_clip.zarr.zip, mkdir data/uva/t5), then
  NAME=t5_c2 bash experiments/uva/scripts/pod_train.sh --data data/uva/t5 --name t5 --image 224
  --wrist --imagine-wrist --steps 8000 --batch 32 --warmup 500 --log-every 50 --val-every 500
  --save-every 1000 --hf-repo rgsk/dream-robot-checkpoints

## Later · does imagining video change the actions? Measure, don't argue (user, 2026-10-27)
- Today `serve.py --imagine` picks the chunk in policy mode, then draws the future in a separate
  forward-model pass: a diagnostic, it cannot change the actions (only costs time per call).
- Inside UVA, the claim to test: with one sampling round (their num_iter = 1) both heads read the
  same z, so joint sampling (full_dynamic_model) should give policy-mode actions; with several
  rounds (num_iter > 1, MaskGIT-style) the filled video tokens feed the next pass and the actions
  are re-sampled from it -- the only way actions "look at" the imagined future (the video-thinking
  dial, experiments/uwm/notes.md step 3). UWM's T5 data point: joint = policy (0/10 + 1/10 both).
- Run on a UVA checkpoint that already grasps (c2 or later), same seeds, 30 + 30:
  | mode | num_iter | success seen / held-out | failure mix | s per call |
  | policy (as now) | 1 | | | |
  | joint (full_dynamic_model, act from it) | 1 | | | |
  | joint | 4 | | | |
  | joint | 8 | | | |
  serve.py needs a --mode joint --num-iter N path (act_out from the full_dynamic_model sample).
- **c1 @3,000 on T5: seen 0/10 (10 NO_GRASP), held-out 0/10 (9 NO_GRASP, 1 wrong bin)** -- worse than
  @1,000 (0/10 + 1/10, some grasps) although val_action_err fell 0.034 -> 0.022.
  Failure (eval/c1_3000/seen_failure_imagined_strip.png): the arm never goes to the cubes; it swings
  to the instructed bin and hovers. Imagined frames still show a BLUE object in the gripper over
  that bin (no blue cube in the scene): the LIBERO-10 prior did not wash out by step 3,000.
- Offline good + closed loop wrong from the first chunk = suspect the run-time hookup on T5 before
  more training (as UWM step 9). Candidates: the episode-start window (16 copies of the first frame;
  training windows pad at most 1 step), the wrist input at run time, 224 px path. Next: replay
  check -- a demo's own recorded window through serve.py vs its recorded actions, and the first
  chunk in the sim vs the demo with the same seed.
