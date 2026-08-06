# dream-robot

An LLM will handle planning. It will not handle the physical part. **This repo is the physical part.**

A modular bench for manipulation policies: one folder per policy, one folder per simulator, tasks under
each simulator, and **any policy x any task** runnable from one command. Every run emits a success rate,
a failure-mode histogram, and a video.

**Read [ROADMAP.md](ROADMAP.md) first** — the thesis, the skill ladder, the interface contract, and the
commercial filter that decides which rung comes next.

## Status

Working toward the first cell of the matrix: record → dataset → train BC → eval.

| | |
|---|---|
| ✅ | `spec.md` — obs/action contract |
| ✅ | `core/schema.py` — canonical types, allowlist enforced by construction |
| ✅ | `core/env_api.py` — `Env` protocol, `FailureMode`, `check_env_conformance` |
| ✅ | `sims/robosuite/tasks/pick_place_cube/` — Panda, absolute joint control, cube → bin |
| ⬜ | scripted expert (predicate-driven) |
| ⬜ | `core/record.py` → LeRobotDataset — **the seam** |
| ⬜ | `policies/bc/` |
| ⬜ | `core/eval.py` — success · cycle time · failure histogram · video |

`uv run pytest -m "not slow"` runs the core suite without a simulator.

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
