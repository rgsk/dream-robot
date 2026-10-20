# dream-robot

An LLM will handle planning. It will not handle the physical part. **This repo is the physical part.**

A modular bench for manipulation policies: one folder per policy, one folder per simulator, tasks under
each simulator, and **any policy x any task** runnable from one command. Every run emits a success rate,
a failure-mode histogram, and a video.

**Read [ROADMAP.md](ROADMAP.md) first** — the thesis, the skill ladder, the interface contract, and the
commercial filter that decides which rung comes next.

## Status

Rung 1 (reach + grasp a rigid cube) is closed in sim for both BC and ACT: record → dataset → train
→ eval, end to end, with a number and a video for every cell. The matrix is below.

| | |
|---|---|
| ✅ | `spec.md` — obs/action contract **and** the dataset layout on disk |
| ✅ | `core/schema.py` — canonical types, allowlist enforced by construction |
| ✅ | `core/env_api.py` — `Env` protocol, `FailureMode`, `check_env_conformance` |
| ✅ | `sims/robosuite/tasks/pick_place_cube/` — Panda, absolute joint control, cube → bin |
| ✅ | scripted expert — predicate-driven phase machine + differential IK |
| ✅ | `core/dataset.py` + `core/record.py` → LeRobotDataset — **the seam** |
| ✅ | `policies/bc/` — spatial-softmax visuomotor net, 0.75M params |
| ✅ | `core/eval.py` + `registry.py` + `run.py` — any policy × any task, one command |
| ✅ | noisy expert — shaky executed actions, clean recorded labels (`--noise-sigma`) |
| ✅ | `policies/act/` — action chunking + transformer, 4.8M params |
| ✅ | `sims/robosuite/tasks/pick_place_two_bins/` — T2: a second bin, coin-flip expert; `_fixed` variant pins the cube |
| ✅ | eval `--noise-sigma` — shake at test time, for scenes with no randomisation of their own |
| ✅ | [`runpod.md`](runpod.md) + `scripts/runpod_setup.sh` — running jobs on a RunPod pod, and when not to |
| ✅ | `sims/robosuite/tasks/two_arm_lift/` — T6: two arms, one loaded tray, four cameras, 16-dim action |
| ✅ | two-arm scripted expert — two phase machines and a synchronisation barrier |
| ✅ | `policies/flow_matching/` — Diffusion Policy with flow matching: action chunks sampled from noise, 5.5M params |
| ✅ | `watch/watch_file.sh` — live view of runs and results, marks each log in progress / finished |
| ✅ | `sims/robosuite/tasks/instructed_pick_place/` — T5: two cubes, two bins, a sentence picks one of each |
| ✅ | `policies/smolvla/` — SmolVLA (450M) fine-tuned with our own training loop; VLM frozen, action expert trains |
| ✅ | `envs/dream_yam/` + `sims/dream_yam/` — Dream-1 (Dream Labs' API policy) on its own YAM arms, Claude supervising through MCP: [below](#llm-plans-dream-1-executes) |

**Next:** our own executor, a video model that imagines the future plus an IDM that reads the
actions off it, in four steps starting with an oracle IDM on T5:
[ROADMAP.md](ROADMAP.md#next-our-own-executor--imagine-the-future-read-the-actions-off-it).

`uv run pytest -m "not slow"` runs the core suite without a simulator.

### Recording a dataset

```- 
MUJOCO_GL=glfw uv run python -m dream_robot.sims.robosuite.tasks.pick_place_cube.record \
    --episodes 25 --verify
```

25 successful episodes ≈ 5.3k frames ≈ 176 s of demonstration, 8.9 MB on disk, about a minute to
record. Add `--noise-sigma 0.025` to shake the executed arm joints while the dataset keeps the
expert's clean action as the label — every recovery becomes a training example. `--verify` reopens the dataset, decodes an episode out of it, and writes a video **from the
bytes on disk** — the seam is only real once you have looked through it. Provenance (every seed,
its outcome, its failure mode, the expert's success rate) lands in `recording_summary.json` beside
the data.

### Training and evaluating

```- 
uv run python -m dream_robot.policies.bc.train --epochs 60
uv run python -m dream_robot.policies.act.train --out experiments/act_pick_place_cube
uv run python -m dream_robot.policies.flow_matching.train --out experiments/fm_pick_place_cube

MUJOCO_GL=glfw uv run python -m dream_robot.core.run \
    --env robosuite/pick_place_cube --policy bc \
    --checkpoint experiments/bc_pick_place_cube/checkpoint.pt
```

`--policy expert` runs the scripted expert through the **same harness**, which is what makes the
ceiling row comparable to every other row.

Every run writes `results.json` plus **`videos/successes.mp4` and `videos/failures.mp4`** — up to two
of each, so you always get footage of the policy winning *and* losing. (Filming "the first three
episodes" is how the first BC run produced three failures and no evidence it ever worked.) An
outcome that never happened leaves no file, so the expert has no `failures.mp4`. To film specific
episodes instead, `--video-seeds 1000 1003 1009` writes just those to `videos/selected.mp4`.

## The matrix so far

`robosuite/pick_place_cube`, 20 evaluation episodes per run from seed 1000 — disjoint from every
recording seed. Where a cell says 3 runs, the policy was trained three times with different seeds.

| data | BC (0.75M) | ACT (4.8M) |
|---|---|---|
| 25 clean demos | 5/20 (25%) | **49/60 (82%)** · 3 runs |
| 25 half-shake demos | 4/20 (20%) | 14/20 (70%) |
| 100 clean demos | 32/60 (53%) · 3 runs | **20/20 (100%)** |
| 100 half-shake demos | **58/60 (97%)** · 3 runs | **20/20 (100%)** |

Scripted expert: 20/20, median cycle 6.9 s. Learned policies' successes run 7.0–7.8 s (BC on 25
half-shake demos: 11.1 s).
Full shake (sigma 0.05) hurts BC: 0/20 at 25 demos, 2/20 at 100.

**What it says.**

- **BC fails at the grasp, and not for lack of precision.** It lands 1–4 cm beside the cube and
  freezes with the gripper open. Clean demos never show an off-centre grasp — the expert is always
  within 0.25 cm — so BC has never seen what to do from there.
- **Shaky demos fix BC, but only with enough of them.** Half shake puts the gripper off-centre and
  records the expert correcting; at 100 demos that takes BC from 53% to 97%. At 25 demos it does
  nothing, and full shake makes the corrections too inconsistent to learn.
- **ACT doesn't need them.** Predicting a 50-step chunk and averaging overlapping plans removes the
  freeze by itself: 82% on the same 25 clean demos where BC gets 25%, and 100% on 100 clean demos.
  Its remaining failures are 3–4 cm edge pinches — it commits to a grasp it has mis-aimed.
- **Held-out action error does not predict success.** Clean ×100 BC has the best error of any BC run
  and scores 53%; ACT's error on 25 demos is about the same as BC's (9.2–11.2 vs 9.6 mrad) and it scores
  3× higher.

Caveats: the ACT 100-demo cells and ACT 25 half shake are single training runs, and 20/20 over 20
episodes means "≳ 85%", not "perfect". Every step, number and dead end is in
[experiments/noisy_expert/notes.md](experiments/noisy_expert/notes.md) and
[experiments/act/notes.md](experiments/act/notes.md).

## T2: two valid bins

Same scene with a second bin mirrored across the cube; the scripted expert flips a coin for which
one. 100 half-shake demos, 20 evaluation episodes from seed 1000. The question is multimodality: does
BC average the two choices into the gap between the bins, does ACT's latent carry the choice, and does
a policy that samples its actions represent both?

| scene | expert | BC (L1) | ACT | flow matching | ACT latent KL | random latent changes the bin |
|---|---|---|---|---|---|---|
| random cube spawn, fair coin | 20/20 · 12 L / 8 R | 20/20 · 6 L / 14 R | 20/20 · 7 L / 13 R | — | 0.00007 | 0 of 20 |
| fixed cube, exactly 50/50 demos, eval with shake | 20/20 · 9 L / 11 R | 20/20 · 14 L / 6 R | 20/20 · 9 L / 11 R | **20/20** | 0.00008 | 1 of 20 |

On the standard eval no cube ended between the bins, and all three policies score 20/20.

**The sampling test** separates them. One start, repeated 20 times: the same env seed, so the same
scene and arm pose, and only the policy's own random seed changes. Fixed-cube scene, no shake.

| one start × 20 runs | start 1000 | start 1010 | start 1013 |
|---|---|---|---|
| **flow matching** | 14 L / 6 R | 10 L / 10 R | 11 L / 9 R |
| BC | 20 L | 20 L | 20 R |
| ACT | 20 R | **20 between the bins (0/20)** | 20 R |

**What it says.**

- **Neither policy averages.** Both use L1, which follows the local majority instead of the mean.
- **With a random spawn, both invent a rule.** 100 fair coins leaned 54/46 by how far the cube spawned,
  and BC and ACT both turned that into "far cube → right bin" (7 of 7 eval seeds each).
- **With the cube fixed and the demos exactly balanced, noise picks the bin.** The shake nudges the
  arm to one side, the next frames look like that side's demos, and the policy commits. One BC episode
  hesitated ~100 steps first; none failed.
- **ACT's latent stays empty in both.** It never needs it: closed-loop tie-breaking already solves the
  task, so encoding the choice only costs KL.
- **Success rate can't tell "commits to one" from "represents both." The sampling test can.** From
  one start, flow matching uses both bins every time (60/60 successful); BC and ACT repeat one outcome.
- **The averaging failure exists, and only the sampling test found it.** At start 1010 ACT carries
  toward the right bin, turns back and drops the cube between them, on all 20 repeats. The 20-seed
  eval never landed on that start.
- **"Fixed cube" still randomises the arm's start joints** (up to 0.036 rad), and that, not the
  shake, is what picked BC's and ACT's bin.
- Flow matching trains in ~23 min on one GPU (100 epochs), about the same as ACT.

Predictions were written before each run and most were wrong; the full logs are in
[experiments/t2/notes.md](experiments/t2/notes.md) and
[experiments/t2_fm/notes.md](experiments/t2_fm/notes.md).

## T6: two arms, one tray

Two Pandas, a tray with a handle at each end and 25 loose balls on it. Each arm takes its own handle
and both lift together. Success is one physical statement: **the tray is above 10 cm, still above it
two seconds later, and still holding at least 19 of its 25 balls.** Tilt is measured but does not
score — it labels failures.

The failure this task exists to measure is one arm lifting while the other has not gripped. It tips
the tray, the balls go on the floor, and the histogram calls it `desynchronised`.

25 clean demos, 20 evaluation episodes from seed 1000.

| policy | three cameras | four cameras |
|---|---|---|
| scripted expert | 20/20 | 20/20 |
| BC (0.75M) | 6/20 (30%) | **15/20 (75%)** |
| ACT (4.8M) | 20/20 | **19/20** · identical across 4 evals |

**What it says.**

- **Bimanual cost the spine nothing.** 16-dim actions, four cameras, two arms — no change in
  `policies/`, `core/` or the dataset format. Rule 2 executed rather than asserted.
- **The expert's barrier is the task.** Neither arm leaves a phase until both have satisfied it.
  Remove it and the same expert scores 5-7/20, with 12 of 20 episodes `desynchronised` — half the
  demonstrations would have been the failure we are trying to measure.
- **BC's jump from 30% to 75% is one camera.** Its dominant failure was closing on nothing
  (`no_grasp` 10 of 14). A straight-down view cannot show how high a gripper is above the table; a
  room-level view can. Adding it took `no_grasp` to 2 of 5.
- **ACT is reproducibly imperfect, which is more useful than a round number.** Four evaluations of
  the same checkpoint fail the same scene (seed 1017) every time: the right arm grips, the left
  misses, and the right lifts anyway and dumps the tray.
- **Every demonstration is a success, so no policy has seen a recovery.** The left gripper closing to
  zero width is directly readable from the robot's own state, and neither policy does anything with
  it. That is the failure-detection gap, not a perception gap.

Cameras are ALOHA's set and each has a job: tight overhead for the tray's rotation and gripper
alignment, a wide room view for the whole scene and for height, one wrist camera per arm for the
close. Full log, including two placement bugs found by looking at renders rather than numbers, is in
[experiments/t6/notes.md](experiments/t6/notes.md).

## T5: "put the red cube in the left box"

Two cubes and two bins. Each episode draws two colours from {red, green, blue, yellow} and a side, and
the instruction names one of each. A policy that ignores the words gets 25%. Evaluation has three
splits: seen colour/side pairs, held-out pairs (yellow→left and green→right never appear in training),
and paraphrases the policy has never read.

SmolVLA base (450M), fine-tuned on 150 clean demos with the VLM frozen, 60 episodes per split:

| split | zero-shot | 10k steps | 20k steps | right cube, when it picked one |
|---|---|---|---|---|
| seen pairs | 0/5 | 22/60 | 22/60 | 45% → 42% |
| held-out pairs | | 12/60 | 11/60 | 25% → 21% |
| paraphrases | | 9/60 | 10/60 | 38% → 30% |

**What it says.**

- **The side word is obeyed perfectly.** No wrong-bin placement in 180 episodes, held-out pairs included.
- **The colour word is at chance.** Doubling the training did not move it, so training length is not
  the bottleneck.
- **Longer training fixed the motion.** Never-grasped dropped from 10 to 6 on seen pairs and from 35
  to 25 on paraphrases. Unseen wordings mostly break the motion before grounding is even tested.
- **Handing over the target as an index doesn't fix it.** Our flow-matching policy, told colour and
  side as a one-hot instead of words, picks the right cube 49% of the time (seen 19/60, held-out 13/60).
  So colour fails before language even comes into it.
- **The pixels carry the colour; our encoder drops it.** The cubes are ~10 px across in the top camera
  and plainly coloured. Asked "which slot holds colour c?" on unseen frames:

  | probe | held-out |
  |---|---|
  | plain CNN, channels averaged | 100% |
  | policy's encoder (spatial softmax), one-hot after it | 52% |
  | policy's encoder, one-hot inside every conv block (FiLM) | 100% |

  Spatial softmax reduces each map to one (x, y) point. With one cube in each slot in every frame, a
  map that fires on "a cube" gives the same point every time, so nothing pushes it to tell red from
  green. Side needs no vision at all, since the bins never move, which is why side is perfect.
- **FiLM fixes the probe but not the policy (yet).** With the one-hot inside the encoders the policy
  gets 16/60 seen and 5/60 held-out: colour is not clearly better (59% on 27 choices), and grasping
  got worse (never-grasped 28 and 37).

**Next:** check whether the one-hot changes the policy's actions at all: same start frame, colour
swapped, see if the planned arm path heads to the other cube. Then film the never-grasped episodes.

Zero-shot, the base model moves but never approaches a cube: it was pretrained on a different arm.
LeRobot's config parser breaks on Python 3.14, so the training loop is written out in
`policies/smolvla/train.py`; 20k steps took 97 min on a rented 4090 (~$1.20). Full log in
[experiments/t5/notes.md](experiments/t5/notes.md).

## LLM plans, Dream-1 executes

The ROADMAP split with a learned executor bought, not trained: **Claude plans, [Dream-1](https://dreamlabs.ai/research/dream-1/)
moves the arms.** The sim is Dream Labs' own *Dream YAM Sim* (it ships in the `dreamlabs` wheel): two YAM
arms, a top camera and a wrist camera on each arm, which is Dream-1's training robot. So Dream-1 answers
with joint targets directly, with no IK and no tool-centre-point guess between them.

```- 
envs/dream_yam/setup.sh                       # Python 3.12 venv, dreamlabs pinned; key: DREAM_LABS_API_KEY in .env
MCP_CONFIG=experiments/dream1/mcp_dream1.json SERVER=dream-yam \
    experiments/agent/scripts/headless_drive.sh 0 dream1       # a fresh claude -p drives one episode
npx @modelcontextprotocol/inspector@0.22.0 --config .mcp.json --server dream-yam   # or drive it yourself
```

The driver never moves a joint. It gets camera images and the gripper's own sensor, never the sim's
state:

- `policy(command)` starts Dream-1 on a short command ("put apple into the bowl") and returns. Dream-1
  keeps driving until `stop`, the next command, or `home`.
- `wait` returns the top camera when a gripper grasps or releases something, or after 10 s. The image
  decides when a step is done; grasp and release are only hints.
- `look` and `review` show the cameras now, or for any earlier stretch of time.

**The world runs on the wall clock, as a real robot's does.** While Dream-1 answers and while the LLM
thinks, time passes. Dream-1 is called the way Dream's own control loop calls it: the next chunk is
requested while the current one plays. Paid calls are capped per server, and a watchdog stops Dream-1
if the driver goes quiet. A referee scores from sim positions into a log that no tool can read.

Task: "put the apple and the orange into the bowl, and the banana on the plate" (3 steps).
Dream-1-beta costs $0.0014 per call.

| how the LLM drove it | result | Dream-1 calls | wall time |
|---|---|---|---|
| sim paused during every call and every LLM turn | 3/3 | 86 | 388 s |
| real time, Dream-1 run in fixed pieces | 3/3 | 94 | 244 s |
| real time, Dream-1 continuous, LLM waits on grasp / release + image | **3/3** | **64 (~$0.09)** | **133 s** |

In every run the LLM's verdict per object matched the referee. Not done yet: the safety layer Dream's
docs require on hardware (easing into each chunk, speed limits, holding on a stale camera), a scene
that changes between resets, and a driver for a real YAM in place of the sim. Full log in
[experiments/dream1/notes.md](experiments/dream1/notes.md).

## The shape

```- 
dream_robot/
  core/        schema · dataset · env_api · record · eval · registry · run
  sims/        robosuite/ · isaac/  → each with tasks/<task>/{env,expert,task.yaml}
  policies/    bc/ · act/ · flow_matching/ · rl_finetune/ · vla/
experiments/   <exp_id>/{config.yaml, results.json, videos/, notes.md}
results.md     the matrix, regenerated — never hand-edited
```

```- 
python -m dream_robot.core.run --env robosuite/pick_place_cube --policy bc
python -m dream_robot.core.run --env isaac/fold_napkin       --policy act
```

## The one rule that makes it work

**The seam between a policy and a simulator is a dataset on disk, never an import.** Policies depend on
torch + core only, and read their dimensions from dataset metadata. That is what lets Isaac (which pins
its own Python runtime) and robosuite coexist in one repo, and what makes bimanual "just a bigger action
dim" instead of a rewrite.

The other four rules are in [ROADMAP.md](ROADMAP.md#five-rules-that-keep-permutations-honest).

## Where the docs live

- `sims/<name>/README.md` — **setup only**: install, launch, quirks, which venv.
- `policies/<name>/walkthrough.py` — **algorithm only**, jupytext-paired, ending in a run-and-see payoff.

When something breaks, it is unambiguous which one to open.

## Related

- `../pytorch-practice` — the learning repo. From-scratch implementations of general techniques.
  Boundary: *general technique from scratch → there; runs against a sim task and emits `results.json` → here.*
- `../world_model` (branch `learn`) — Genie-style video world model; the eventual imagination engine.
