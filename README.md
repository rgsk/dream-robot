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
BC average the two choices into the gap between the bins, and does ACT's latent carry the choice?

| scene | expert | BC (L1) | ACT | ACT latent KL | random latent changes the bin |
|---|---|---|---|---|---|
| random cube spawn, fair coin | 20/20 · 12 L / 8 R | 20/20 · 6 L / 14 R | 20/20 · 7 L / 13 R | 0.00007 | 0 of 20 |
| fixed cube, exactly 50/50 demos, eval with shake | 20/20 · 9 L / 11 R | 20/20 · 14 L / 6 R | 20/20 · 9 L / 11 R | 0.00008 | 1 of 20 |

No cube ever ended between the bins.

**What it says.**

- **Neither policy averages.** Both use L1, which follows the local majority instead of the mean.
- **With a random spawn, both invent a rule.** 100 fair coins leaned 54/46 by how far the cube spawned,
  and BC and ACT both turned that into "far cube → right bin" (7 of 7 eval seeds each).
- **With the cube fixed and the demos exactly balanced, noise picks the bin.** The shake nudges the
  arm to one side, the next frames look like that side's demos, and the policy commits. One BC episode
  hesitated ~100 steps first; none failed.
- **ACT's latent stays empty in both.** It never needs it: closed-loop tie-breaking already solves the
  task, so encoding the choice only costs KL.
- **Success rate can't tell "commits to one" from "represents both."** A test that can: run a policy
  many times from one identical start with no shake and check it produces both bins. BC and ACT
  can't; Diffusion Policy should.

Predictions were written before each run and most were wrong; the full log is in
[experiments/t2/notes.md](experiments/t2/notes.md).

## The shape

```- 
dream_robot/
  core/        schema · dataset · env_api · record · eval · registry · run
  sims/        robosuite/ · isaac/  → each with tasks/<task>/{env,expert,task.yaml}
  policies/    bc/ · act/ · diffusion_policy/ · rl_finetune/ · vla/
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
