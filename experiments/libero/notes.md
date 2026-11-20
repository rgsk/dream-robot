# LIBERO

Next rung (2026-11-20): one Verne policy on many short LIBERO-90 skills; the LLM (MCP) chains them for
LIBERO-10's long tasks. Published LIBERO numbers are saturated (~97-99% for big VLAs, trained per suite on the
test tasks); ours is a different protocol: skills -> chained long tasks, small model from scratch.

## Step 1 · install + film demos
- envs/libero/setup.sh: Python 3.11 venv, hf-libero (robosuite 1.4.0, mujoco 3.8.1); torch 2.7.1 hard-linked from
  RoboCasa's venv (uv's cache couldn't reuse it, 3.5 GB at 0.1 MB/s otherwise); repo-local LIBERO config
  (envs/libero/config/, LIBERO_CONFIG_PATH) since ~/.libero points at robot-vla. Object assets: ~/.cache/libero
  (was a partial copy, completed from HF: 586 files, 409 MB).
- Data: data/libero/libero_90/, 3 tasks of 50 human demos each (HF yifengzhu-hf/LIBERO-datasets):
  open the top drawer (67-177 steps, mean 95), turn on the stove (80-105, mean 93), moka pot on the stove
  (126-218, mean 151). Obs: agentview + wrist 128x128, joints 7, gripper 2, ee pose; actions 7 (ee delta + grip);
  20 Hz.
- Replay of demo_0 in our sim from its saved first state: **success on all 3**. scripts/film_demos.py; films in
  scratch_nov/libero/films/.
- Scene 8 "turn off the stove" (only turn-off task in LIBERO; same flat stove, two moka pots, no pan): 50 demos,
  101-283 steps (mean 176); replay success.

## Scene 3 plan (2026-11-20)
- Skills (LIBERO-90, scene 3): turn on the stove; moka pot on the stove; frying pan on the stove. Turn off from
  scene 8 (cross-scene). Chains never trained, LLM-driven: "turn on the stove and put the frying pan on it"
  (LIBERO-90), "turn on the stove and put the moka pot on it" (LIBERO-10).
- All 5 scene-3 tasks have the same bddl init: stove, pan, moka in the same regions, stove off. Goals differ only.
  So skill 1 of a chain starts in-distribution; skill 2 starts with the stove ON and the arm wherever skill 1 left
  it (the T5 second-move question again).

## Design · skill calls with fixed vocabularies (2026-11-20)
- The LLM picks from dropdowns; free language stays on the LLM side. One MCP tool per verb, so the schema only
  allows sensible combinations:
  `turn_on(object: stove)`, `turn_off(object: stove)`, `open/close(object: top_drawer | bottom_drawer | microwave)`,
  `put(object: moka_pot | frying_pan | bowl | ..., target: stove | plate | top_drawer | basket | ...)`.
- Policy input: 3 tokens, verb + object + target (target empty except put), each a learned embedding (fixed
  vocab; a new word = a new row) + a role embedding; the patches read them by cross-attention. No CLIP for now.
- Each object / target slot gets its own position target (step 22's spread loss, one per slot): the item's centre
  per camera, labelled from the sim's object poses. What the slots name is what the patches must mark.
- Start: scenes 3 + 8. Verbs turn_on, turn_off, put; objects stove, moka_pot, frying_pan; target stove. Skills:
  turn_on(stove), turn_off(stove) [scene 8], put(moka_pot, stove), put(frying_pan, stove).
- Limit, fine for now: two of the same object (scene 8's two moka pots, scene 2's three bowls) need a qualifier
  or a point-in-image slot later.
- Later tests: recombination (put(frying_pan, stove) in scene 3 from pan seen in scene 9), cross-scene verbs
  (turn_off from scene 8 used in scene 3), point slots for unnamed objects.

## Step 2 · demos -> npz with position labels (scripts/to_npz.py, LIBERO venv)
- data/libero/npz/<task>.npz: top + wrist 128x128 (upright), joints, gripper, ee pose, 7-d actions, episode,
  skill tokens (verb, object, target), and for EVERY name in the scene its centre per camera + an in-view flag
  (own projection: robosuite's rounds and clamps off-image points to the edge). Names: stove_knob (what turn
  on/off press), stove (cook region, a put target), moka_pot, frying_pan; scene 8's two moka pots unlabelled.
- turn_on 4645 steps, moka 7563, pan 9979, turn_off (scene 8) 8795; 50 demos each. In view: top 100% for all;
  wrist varies (the task's own object 98-100%, except the pan 29%: the camera looks along its handle).
- Check films (scratch_nov/libero/npz_check/): dots on the right objects in both cameras after two fixes (no row
  flip; wrist matrix per step). The pan's label is its body centre; the hand grasps the handle.

## Step 3 · VAE on LIBERO frames (runs/vae_modal)
- Verne's canvas VAE (top | wrist 128x256 -> 4x16x32), 14,039 train frames (every 2nd, 45 demos per skill),
  147 val (5 held-out demos per skill). Cosine lr (250 warmup, peak 5e-4), clip 1.0, 10k steps (vae.train got
  schedule='cosine'; WSD runs on T1/T5 spiked in their constant phase).
- Modal H100: 36 s / 500 steps, 12 min, ~$0.8 (99.7% util, 10.6 GB, 535 W); local 4060 330 s / 500 (stopped
  at 1500). Val recon 0.3625 -> 0.0050 (2k) -> 0.0031 (6k) -> **0.0028** (10k), no spikes; KL 1.34. T5's: 0.0017.
- recon: objects, arm, knob kept; the stove's red glow when on survives (turn_on's visible result); fine
  burner rings blur away in the top view. Good enough for pointing at stove / pan / moka.

## Step 4 · stage 1 on LIBERO: point at a named object, spread, pixels (runs/position_spread/seed0)
- dream_robot/policies/verne/libero.py: vocab 3 verbs + 4 names + 4 'to_<name>' (11 words); state 9; demo frames
  (31k, demos 0-4 of each task held out); every frame asked about any name in view; spread per camera, cameras
  where the name is out of view skipped. Verne(n_words, state_dim, frames) (T5 defaults unchanged). Modal
  libero_position (H100, 5k steps).
- Held-out (err patches / right = nearer the named object than every other object in view), step 5000:
  **all 0.54 / 97.2%**; stove_knob 0.54 / 97%, stove 0.60 / 98%, moka_pot 0.52 / 95%, frying_pan 0.45 / 98%.
  Moka lagged early (right 29% at 1k, 52% at 2k, 94% at 2.5k). position.png: crosses on the right objects in both
  cameras, arm anywhere, incl. stove-on frames and scene 8 (knob + cook region only).
- Occluded objects still count as in view (a knob hidden under the pan keeps its label).

## Step 5 · option B (Verne's hand-pose actions) on LIBERO: can hand poses drive the arm? (scripts/hand_check.py)
- Controller (env_args): OSC_POSE, control_delta, a in [-1, 1]: Δp = 0.05·a[:3] m, R_goal = exp(0.5·a[3:])·R_now.
  Driver: a[:3] = clip((p − p_now)/0.05), a[3:] = clip(rotvec(R·R_nowᵀ)/0.5), gripper = the recorded command.
  Hand pose = the controller's ee (recorded obs ee_pos is ~8 mm off it: don't use the stored one).
- Replay from each demo's first state, 5 demos per skill: recorded commands vs driving to the NEXT recorded hand pose
  each step (closed loop): turn_on 5/5 vs 5/5, moka 4/5 vs 5/5, pan 5/5 vs 5/5, turn_off 3/5 vs **1/5**.
  Total 17/20 vs 16/20. The hand trails the recorded path by 12-18 mm (commands: 5-6 mm): a one-step target
  is reached only partly per step (OSC lag). Turn-off (a knob turn) suffers most. Even recorded commands replay
  imperfectly (3/5 turn_off): the sim isn't fully deterministic from a saved state.
- Aim-ahead (target = recorded hand pose 1 / 2 / 3 steps ahead), 10 demos per skill, success · path distance:
  turn_on cmds 10/10 · 6.6 mm | +1 10/10 · 16.7 | +2 9/10 · 11.7 | +3 9/10 · 8.0
  moka    cmds  9/10 · 4.4    | +1 10/10 · 17.4 | +2 10/10 · 12.0 | +3 10/10 · 6.9
  pan     cmds 10/10 · 6.4    | +1  9/10 · 18.4 | +2  9/10 · 12.4 | +3  9/10 · 6.7
  turn_off cmds 9/10 · 6.2    | +1  7/10 · 13.6 | +2  6/10 · 10.1 | +3  6/10 · 6.4
- Read: +3 puts the hand back on the path (6-8 mm) but turn_off stays 6-7/10 vs 9/10: the miss isn't lag. A
  knob turn needs the hand pushed PAST where it ends up (contact); the achieved pose drops that push. T5 step 20
  labelled with the hand at each recorded joint TARGET (the goal), not the achieved hand. LIBERO's goal is
  recoverable: p_goal = p_t + 0.05·a[:3], R_goal = exp(0.5·a[3:])·R_t.
- Goal poses (target = the recorded command applied to the recorded hand), 10 demos per skill, success · path:
  turn_off cmds 7/10 · 6.5 mm | goal 7/10 · 3.7    turn_on cmds 10/10 · 5.5 | goal 10/10 · 3.3
  moka     cmds 9/10 · 4.4    | goal 10/10 · 0.6   pan     cmds 10/10 · 6.4 | goal 10/10 · 0.6
  Goal = commands on every skill (37/40 vs 36/40), path closer than the commands' own replay. turn_off's
  commands scored 9/10 last run and 7/10 now: replaying a saved state isn't repeatable; 10 demos is noisy.
- Decision: option B with GOAL poses as the training targets (as T5 step 20: the hand at the recorded target).
- to_npz.py now also stores the controller hand (hand_pos/rot) and the aiming pose (goal_pos/rot) per step.
  Aim vs the hand one step later: median 13-21 mm, 95% 24-35 mm, while the hand moves 2-6 mm per step (median):
  the demos always aim ~2 cm ahead and the hand covers a fraction each step (OSC kp 150), everywhere, not only at
  contact. Why aiming 3 steps ahead already put the hand on the path.

## Step 6 · stage 2 on LIBERO (runs/skills_spread/seed0) + rollouts (scripts/rollout.py)
- libero.py stage 2: skill words + option-B targets (aiming poses relative to the chunk start, grip, progress),
  demo B 32, position loss on (B 64, weight 1), lr 3e-4 cosine, init stage 1 seed 0, 20k steps (Modal H100);
  weights-only snapshots every 2k, full ckpt every 5k; stats.pt written before training.
- rollout.py (LIBERO venv, local 4060): LIBERO's own pruned init states, 10 settle steps, chunk every 8 steps,
  driver to the live hand, max 400 steps, task's own success check.
- **10k snapshot, 20 episodes per skill: 56/80.** turn_on **20/20** (78-103 steps), turn_off (scene 8) **15/20**,
  pan **15/20**, moka **6/20**. Moka fail film (ep0): reaches the pot, closes beside it, carries nothing to the
  stove and hovers there (T5's old failure: missed grasp, no retry).
- **20k (final), 20 episodes per skill: 66/80** (10k: 56/80). turn_on 20/20, turn_off 17/20 (15), pan 17/20 (15),
  moka **12/20** (6). Training: held-out action loss 0.209 (10k) -> 0.199 (20k), pointing right 98.5 -> 98.8%.
  Late training helped the arm, moka most; 20k stays the default for LIBERO stage 2. One seed.

## Step 7 · free readout on LIBERO (runs/position_free, runs/skills_free)
- libero.py readout option ('spread' / 'free'; free: squared distance per camera, out-of-view skipped).
- Stage 1 free seed 0, step 5000: **err 0.12 patches, right 99.6%** (every name 99-100%; moka 0.15 / 100%).
  Spread: 0.54 / 97.2% (moka 95%). Free reached 98.9% (moka 99%) by step 1000, spread 79.6% (moka 29%).
  position.png: crosses dead on, both cameras.
- Free stage 2 seed 0 (init free stage 1, free position loss on, 20k, Modal). **10k snapshot: 67/80** (spread 10k:
  56/80): turn_on 20/20, moka **15/20** (spread 6), pan 15/20 (15), turn_off 17/20 (15).
- **Free 20k (final): 70/80** (spread 20k: 66/80): turn_on 20/20, moka **17/20** (12), pan 17/20 (17), turn_off 16/20
  (17). Pointing at the end: free 0.11 patches / 99.6%, spread 0.48 / 98.8%.
- Odd: free's held-out action loss ends at 0.444 vs spread's 0.199, yet its arm is better. Not understood yet
  (both use the same val rows); check its log curve before trusting the loss as a guide.
- Read (one seed each): on LIBERO's multi-patch objects free beats spread, mostly on the moka pot (17 vs 12 at 20k,
  15 vs 6 at 10k): the opposite of T5's one-patch cubes. Your hunch held.
- Action-loss puzzle (log curves): free OVERFITS the demos from ~8k: train 0.12 -> 0.04, held-out 0.225 (8k, its
  best) -> 0.444 (20k); its position loss falls to 0.0013 (stage 1 already solved it). Spread: train ~0.13-0.16,
  held-out flat 0.19-0.21; its position loss stays ~0.12 (spread can't go below its snap floor), so the aux
  gradient never stops. Guess: that steady aux gradient regularises spread; free's vanishes. The arm doesn't
  suffer (free 10k 67, 20k 70): the held-out loss scores exact demo actions on 5 unseen demos per task, not
  closed-loop success.
- LIBERO's eval init states are NOT the demos' starts (checked moka / turn_on / pan): robot joints differ, the
  gripper starts less open (0.021 vs 0.034), objects in a narrower spread (only a few object entries vary). So
  the rollout scores are on fresh starts.
