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
