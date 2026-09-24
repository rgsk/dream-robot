# T2 — flow matching (`policies/flow_matching`)

Question: does a sampled policy represent both bins, where BC/ACT only commit to one?
Background: `experiments/t2/notes.md`.

## Step 1 — BC/ACT baseline without shake (2026-09-24)

- T2-fixed checkpoints, eval seeds 1000–1019, `--noise-sigma 0`. Script: `experiments/t2/scripts/rollout_paths.py`.
  Output: `paths/{bc,act}_noshake.json`.

  | | left | right | between |
  |---|---|---|---|
  | BC | 18 | 2 | 0 |
  | ACT (z = 0) | 7 | 12 | 1 (seed 1010) |

- Not one bin every time, as predicted: **T2-fixed pins only the cube.** robosuite's reset still adds
  joint-init noise (start state differs by up to 0.036 rad across seeds), and that picks the bin.
- Rerun of the same seeds: identical bin on 20/20 for both. BC/ACT are deterministic given the start.
- **Consequence for the sampling test:** "identical start" means *one env seed, repeated*, with only the
  policy's noise seed varied. BC/ACT give one bin per env seed by construction.

## Step 2 — Train flow matching on T2-fixed (2026-09-24)

- RunPod RTX 2000 Ada, 100 epochs, 22.5 min (~13 s/epoch; GPU ~56%, bound by video decode on 5 cores).
  Script `scripts/pod_fixed_fm.sh`, log `pod_fixed_fm.log`. → `experiments/t2_fixed_fm_half100/`.
- Final: train MSE 0.051, val 0.065, sampled first-action error 8.1 mrad (BC 6.5, ACT 9.3 on the same data).
- Parked: cache decoded frames on GPU to lift utilisation — worth it only if more runs follow.

## Step 3 — Sampling test + standard eval (2026-09-24)

Sampling test: `rollout_paths.py --repeat-seed S` — env seed S for all 20 episodes, only `policy.seed(i)` varies.
Starts S = 1000, 1010, 1013 (no-shake BC/ACT went L/R, L/between, R/R there).

Predictions:
- Flow matching: **both bins at every start**, 5–15 left of 20; 0 between.
- BC, ACT: one bin 20/20 at every start (deterministic).
- Standard eval (half shake, seeds 1000–1019): FM ≥ 18/20.

Results (pod, log `pod_fixed_fm_eval.log`, plot `paths/sampling_test.png`):

| one start × 20 | start 1000 | start 1010 | start 1013 |
|---|---|---|---|
| **flow matching** | 14 L / 6 R | 10 L / 10 R | 11 L / 9 R |
| BC | 20 L | 20 L | 20 R |
| ACT (z = 0) | 20 R | **20 between** | 20 R |

- **FM represents both bins**: both at every start, all 60 successful. Prediction held.
- BC/ACT one outcome per start, as predicted. ACT at start 1010 averages into the gap every time
  (0/20) — the averaging failure ROADMAP expected, found by testing one start instead of 20.
- Standard eval, half shake, seeds 1000–1019: **FM 20/20**, median cycle 7.2 s (BC 7.1, ACT 7.3).
- Caveat: the shake path run (`flow_matching_shake.json`) went 19 L / 1 R because `rollout_paths.py`
  seeds the policy with 0 on every episode outside `--repeat-seed`. Same noise → mostly the same bin
  despite the shake: FM's choice follows its own noise, not the jitter. The harness eval does not reseed.

Videos (`scripts/film_samples.py` for FM; `core.run --video-seeds` for ACT):
- ACT at start 1010: carries toward the right bin, turns back, drops the cube between the bins, then
  sits until the 600-step timeout. All 20 repeats identical (same path, same cube end point) — ACT
  has no randomness once the start is fixed. → `act_start1010/.../videos/selected.mp4`.
- FM at start 1010, policy seeds 0/1/2: left, right, left, all clean. → `videos/fm_start1010_seeds012.mp4`.
