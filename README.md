# dream-robot

An LLM will handle planning. It will not handle the physical part. **This repo is the physical part.**

A modular bench for manipulation policies: one folder per policy, one folder per simulator, tasks under
each simulator, and **any policy x any task** runnable from one command. Every run emits a success rate,
a failure-mode histogram, and a video.

**Read [ROADMAP.md](ROADMAP.md) first** — the thesis, the skill ladder, the interface contract, and the
commercial filter that decides which rung comes next.

## Status

The first cell of the matrix is closed: record → dataset → train BC → eval, end to end, with a
number and a video at the end of it. What comes next is making that number better.

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
| ⬜ | noisy expert — jittered waypoints, recorded recoveries (**next**, see below) |
| ⬜ | `policies/act/` |

`uv run pytest -m "not slow"` runs the core suite without a simulator.

### Recording a dataset

```- 
MUJOCO_GL=glfw uv run python -m dream_robot.sims.robosuite.tasks.pick_place_cube.record \
    --episodes 25 --verify
```

25 successful episodes ≈ 5.3k frames ≈ 176 s of demonstration, 8.9 MB on disk, about a minute to
record. `--verify` reopens the dataset, decodes an episode out of it, and writes a video **from the
bytes on disk** — the seam is only real once you have looked through it. Provenance (every seed,
its outcome, its failure mode, the expert's success rate) lands in `recording_summary.json` beside
the data.

### Training and evaluating

```- 
uv run python -m dream_robot.policies.bc.train --epochs 60

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

`robosuite/pick_place_cube`, 20 episodes from seed 1000 — disjoint from the 25 seeds the
demonstrations were recorded on.

| policy | success | median cycle | failures |
|---|---|---|---|
| expert (scripted) | **100%** (20/20) | 6.9 s | — |
| bc (20 demos, 0.75M params) | **25%** (5/20) | 7.6 s | 14 × `no_grasp`, 1 × `wrong_target` |

**The failure histogram is the finding.** BC transports and places correctly whenever it gets hold
of the cube — its successful episodes have expert-like cycle times — and misses the grasp in 93% of
its failures. Held-out action error is 9.6 mrad, *below* the controller's own 18 mrad tracking lag,
so the policy predicts the expert's actions accurately on the expert's own states.

Re-evaluated on the seeds it trained on, it scores 50%. That splits the loss in two: half is
compounding error — it cannot hold together a trajectory it has seen the demonstration for — and
half is generalisation to unseen cube positions. The first half is exactly what
[ROADMAP](ROADMAP.md) predicts for a **deterministic** expert: the demonstrations cover a ribbon of
state space one trajectory wide, and a policy that drifts off it has never seen how to get back.
The fix is prescribed there too, and it is the next box: a noisy expert with jittered waypoints,
wider start poses, and recorded recoveries from perturbed states.

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
