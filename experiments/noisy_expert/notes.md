# Noisy expert — log

Goal: lift BC above 25% on `robosuite/pick_place_cube` by fixing the demonstrations, not the policy.
ROADMAP "Two demo-data notes": a deterministic expert covers a ribbon of state space one trajectory wide.

Scripts that produced every number below: [scripts/](scripts/). Run from the repo root with `MUJOCO_GL=egl uv run python experiments/noisy_expert/scripts/<name>.py`.
Throwaway visuals: `scratch/noisy_expert/` (gitignored).

---

## Step 1 — Why does BC fail? (2026-09-15)

**Result.** BC doesn't wander; it parks beside the cube and freezes.

Measured gripper-to-cube horizontal error at the moment the gripper reaches grasp height (< 3 cm above cube centre), eval seeds 1000–1019 (`arrival.py`):

| | xy error on arrival | outcome |
|---|---|---|
| demos (25 recording seeds, gripper open) | ≤ 0.25 cm, median 0.15 | — |
| expert on eval seeds | 0.07 – 0.26 cm | 20/20 success |
| BC on eval seeds | 0.87 – 3.86 cm, median 2.18 | 5 success · 11 frozen_open · 3 closed_on_air · 1 closed_then_lost |

- **Every** BC episode, successes included, lands 3.5–15× further off than any demo ever is.
- frozen_open: arm stops moving for good between steps 105–217, gripper fully open, until the 600-step horizon.
- The offset is visible in BC's own wrist camera (`scratch/noisy_expert/wrist_at_arrival.png`). The information is there; no demo ever shows the correction.
- Arrival error only partly predicts outcome (seed 1004 fails at 1.14 cm; seed 1009 succeeds at 2.57 cm).

**Dead end noted.** Joint-space distance to the nearest demo state was tried first and is confounded: the expert itself is 0.04–0.12 rad from any demo state on unseen seeds. Replaced by the task-space arrival error above.

**Decision.** Noise must put the gripper 1–4 cm off *at grasp height* with the expert correcting. That coverage number is step 3's acceptance check.

## Step 2 — Recorder can execute noisy actions, record clean ones (2026-09-15)

**What.** Optional perturbation in `core/record.py`: `env.step` gets `action + noise`, the dataset keeps the expert's clean `action` (DART, Laskey et al. 2017). `JointNoise(sigma)` = Gaussian on the 7 arm joints, gripper untouched. Noise seeded from the episode seed (reproducible). The noise config is stored in `recording_summary.json`.

**Why clean labels.** Recording the noisy action would teach BC to reproduce jitter, not recover.

**Checked** (`tracking.py`, seeds 0–4): the recorder's tracking integrity check (limit 0.25 rad) stays valid on clean labels:

| sigma | expert success | clean-label tracking max | executed-action tracking max |
|---|---|---|---|
| 0 | 5/5 | 0.060 | 0.060 |
| 0.03 | 4/5 | 0.063 | 0.152 |
| 0.05 | 4/5 | 0.069 | 0.250 (at limit) |

Closed-loop expert re-solves from wherever the noise put the arm, so its clean label still matches the next state.

136 fast tests + the real-sim recording test pass. Not committed yet.

## Step 3 — Pick a noise level, record shaky demos (2026-09-15, in progress)

**Robustness probe** (`noise_probe.py`, eval seeds 1000–1009): expert success 10/10 at sigma 0, 9/10 at 0.02 and 0.05, 3/10 at 0.1, 0/10 at 0.2. Failures are all no_grasp.

**Coverage** (`noisy_coverage.py`, recording seeds 0–24, successful episodes only — the recorder discards failures):

| sigma | expert success | demos with a > 1 cm off-centre moment at grasp height | worst | demos > 2 cm | steps > 1 cm |
|---|---|---|---|---|---|
| 0 | 25/25 | 0/25 | 0.25 cm | 0 | 0 |
| 0.03 | 23/25 | 16/23 | 2.19 cm | 3 | 114 |
| 0.05 | 21/25 | 19/21 | 2.70 cm | 7 | 221 |

**Decision.** sigma = 0.05: widest coverage of BC's 1.1–3.9 cm failure range; the lost attempts are simply not kept.
Coverage still tops out at 2.7 cm vs BC's 3.9 cm worst — if the retrained BC still fails at > 3 cm, add start-of-grasp pushes.
Caveat: > 1 cm steps are ~221 of ~6k frames (~4%). Real, but sparse.

**Recording** (sigma 0.05, 25 kept episodes, `record.py --noise-sigma 0.05`):

| dataset | kept | frames | attempts | expert success | failures | tracking max (limit 0.25) |
|---|---|---|---|---|---|---|
| `data/robosuite/pick_place_cube` (clean) | 25 | 5288 | — | — | — | — |
| `data/robosuite/pick_place_cube_noise005` | 25 | 5837 (+10%) | 30 | 83% | 4 no_grasp, 1 dropped | 0.073 rad |

Kept seeds: 1, 2, 4–14, 17–27, 29 (seeds 0, 3, 15, 16, 28 failed). Clean dataset left untouched, so BC can be trained on both.
Round-trip video (episode 0, decoded from disk): `experiments/noisy_expert/videos/dataset_roundtrip.mp4`.

**Acceptance check — does the shaky data contain off-centre moments?** (`coverage_and_recovery.py`, replays the exact kept seeds of each dataset; 0 replay mismatches vs the recordings)

| dataset | grasp-height steps, gripper open | > 1 cm off | > 2 cm off | max |
|---|---|---|---|---|
| clean | 160 | 0 | 0 | 0.25 cm |
| shaky (sigma 0.05) | 356 | 236 | 41 | 2.70 cm (seed 14) |

**Passed.** Clean demos contain zero off-centre examples; shaky demos contain 236 steps > 1 cm, covering most of BC's failure range (1.1–3.9 cm). Gap: nothing above 2.7 cm, where 6 of BC's 15 failures sit.
Visuals: `scratch/noisy_expert/demo_coverage_clean_vs_shaky.png`, `scratch/noisy_expert/shaky_expert_recovery_seed14.mp4`.

**Next.** Retrain BC on the shaky dataset and evaluate on seeds 1000–1019 (3 training seeds), before widening start poses — the retrain answers whether this is enough.

## Step 5 — Retrain BC on shaky demos, 1 run first (2026-09-15, in progress)

Skipped step 4 (wider start poses) for now: a single retrain answers whether the noise alone helps.
Started with 1 training run (user call) as a directional look; 3 runs each on clean and shaky data needed for a trustworthy comparison, since the 25% baseline is itself a single run.

**Training** (`python -m dream_robot.policies.bc.train --root data/robosuite/pick_place_cube_noise005 --out experiments/bc_noise005`, default config, seed 0):

| model | train / val frames | wall clock | best held-out action error |
|---|---|---|---|
| BC, clean demos (baseline) | 4218 / 1070 | 132 s | 9.6 mrad |
| BC, shaky demos | 4671 / 1166 | 144 s | 18.65 mrad |

Higher held-out error is expected, not a regression: corrective actions depend on noise the policy cannot see, so they are less predictable. Held-out error is not the metric — rollout success is (random shift in the BC walkthrough made val error worse while success went 2% → 28%).

Eval: `python -m dream_robot.core.run --env robosuite/pick_place_cube --policy bc --checkpoint experiments/bc_noise005/checkpoint.pt --experiments experiments/bc_noise005/eval` (written to a separate folder so the baseline's results.json is not overwritten).

**Eval result: worse.** 20 eval seeds 1000–1019, single training run.

| model | success | failures |
|---|---|---|
| BC, clean demos (baseline) | 5/20 (25%) | 14 no_grasp, 1 wrong_target |
| BC, shaky demos (sigma 0.05) | **0/20 (0%)** | 20 no_grasp |

Per seed: it lost all five seeds the baseline solved (1003, 1009, 1010, 1012, 1014) and gained none.
Results: `experiments/bc_noise005/eval/bc_on_robosuite_pick_place_cube/results.json`.

Caveat: one training run. But 0/20 vs 5/20 with no seed gained is unlikely to be run-to-run noise alone.

**Leading hypothesis (unverified at time of writing).** The noise kicks the expert out of its 1 cm "close now" window at grasp height, so shaky demos contain 2.2× as many hover-low-with-gripper-open steps (356 vs 160, step 3). BC may be learning "hover open" as the dominant behaviour there. Diagnosis running: `scripts/diagnose_bc.py`, `scripts/dataset_gripper_stats.py`.

**Hypothesis refuted** (`scripts/dataset_gripper_stats.py`, recorded labels):

| | clean | shaky |
|---|---|---|
| frames with close command | 63.9% | 61.9% |
| frames open & arm still (hovering) | 2.6% | 1.8% |
| steps until first close, median (range) | 68 (59–75) | 77 (64–118) |
| arm action std (rad) | 0.134 | 0.152 |

The shaky labels do not contain more hovering-open behaviour — slightly less. The 356-vs-160 grasp-height count in step 3 was steps spent *low and off-centre while correcting*, not idle hovering. Cause of 0/20 still unknown; waiting on `scripts/diagnose_bc.py` (log: `experiments/bc_noise005/diagnose.log`).

**Diagnosis of the 0/20** (`scripts/diagnose_bc.py`, eval seeds 1000–1019):

| model | arrival xy at grasp height | failure kinds | gripper command after arrival (min) |
|---|---|---|---|
| BC, clean demos | median 2.18, 0.87–3.86 cm (step 1) | 11 frozen_open, 3 closed_on_air, 1 closed_then_lost, 5 success | re-measuring (log: `experiments/bc_pick_place_cube/diagnose.log`) |
| BC, shaky demos | **median 1.45, 0.24–2.55 cm** | **20 frozen_open** | **0.58–0.88 — never below 0.5** |

**The shake fixed aim and broke the grasp decision.** Shaky-data BC lands closer to the cube than clean BC on almost every seed (seed 1012: 0.24 cm, as centred as the expert), but never commands close on any seed, then freezes.

**Current hypothesis.** In clean demos the close happens at a near-identical moment and arm pose (first close at step 59–75), so BC can key the close off proprioception. Noise scatters that (64–118) and removes the pose cue; closing must now be judged from the images, and BC doesn't. Test: gripper prediction on held-out demo frames at the true close moment, both models.

**Hypothesis refuted: shaky BC *can* predict the close** (`scripts/gripper_on_demos.py`, frames first_close ±10 in each model's own dataset):

| model | split | pred. gripper before close (label 1) | after close (label 0) | frames predicted < 0.5 | eps with a close |
|---|---|---|---|---|---|
| clean BC | val | 0.89 | 0.19 | 80% | 5/5 |
| clean BC | train | 0.92 | 0.17 | 79% | 20/20 |
| shaky BC | val | 0.94 | 0.22 | 76% | 5/5 |
| shaky BC | train | 0.95 | 0.24 | 75% | 20/20 |

On demo frames both close at the right moment. The failure is rollout-only: BC must be stopping in a pose that never precedes a close in the demos.

**Old BC, same diagnosis** (`experiments/bc_pick_place_cube/diagnose.log`): when it closes it commands a clean 0.00 (5 success, 3 closed_on_air, 1 closed_then_lost); its 11 frozen_open episodes never go below 0.52. Same failure mode as shaky BC — shaky BC just hits it on 20/20.

Next test: pose at freeze (xy, height) vs the expert's pose at its close moment.

**Pose at freeze vs expert's pose at close** (`scripts/freeze_pose.py`; height = gripper site above cube centre):

| | sideways xy, median | height, median (range) |
|---|---|---|
| shaky expert, at its first close (25 kept seeds) | 0.63 cm (max 2.31) | +1.59 cm (+0.19..+2.51) |
| clean BC, frozen-open episodes (11) | 3.23 cm | −1.08 cm (−1.15..−1.00) |
| shaky BC, frozen-open episodes (20) | 1.98 cm | +1.33 cm (+0.77..+1.74) |

- **Height is not the problem** for shaky BC — it freezes inside the expert's closing height range. (Height suspect refuted.)
- **Sideways is.** It stops ~2 cm off (xy grows from 1.45 at arrival to 1.98 at freeze), where the expert almost never closes (median 0.63). So not closing there is *correct* by the demos — the missing part is the sideways correction. BC learned "don't close yet" but not "move toward the cube", and stops.
- Clean BC is different: it drives fingers down beside the cube (−1.1 cm, 3.2 cm off).

**Working hypothesis for why it doesn't correct:** actions are absolute joint targets and state is an input, so "copy the current pose" is a cheap near-correct answer. The corrective part (target − current pose) is a few tens of mrad, similar to BC's own held-out error (18.7 mrad), so it gets lost and the arm stalls. Not yet tested.

**Copycat hypothesis refuted** (`scripts/copycat_test.py`; held-out demo frames, gripper open, binned by label correction size = |label − current pose| over 7 arm joints; "copy-pose err" = error of just predicting the current pose):

| model | correction size | frames | predicted / label size | direction (cosine) | model err | copy-pose err |
|---|---|---|---|---|---|---|
| clean BC | 10–25 mrad | 31 | 1.13 | 0.85 | 10.1 | 17.1 |
| clean BC | 25–45 mrad | 29 | 1.28 | 0.87 | 23.6 | 34.0 |
| clean BC | 45–80 mrad | 97 | 1.00 | 0.95 | 20.0 | 66.6 |
| shaky BC | 10–25 mrad | 26 | 1.62 | **0.49** | **29.4** | 20.7 |
| shaky BC | 25–45 mrad | 103 | 1.39 | **0.46** | **47.2** | 35.3 |
| shaky BC | 45–80 mrad | 129 | 1.01 | 0.71 | 47.3 | 64.6 |

- BC does **not** shrink corrections toward "stay put" — predicted size is ≥ the label's for both models.
- **Shaky BC points small and medium corrections in the wrong direction** (cosine ~0.47, i.e. ~60° off; clean BC 0.85+). On those frames it is *worse than doing nothing*.
- Consistent with the rollouts: correction-sized pushes in inconsistent directions near the cube → no net progress → stall ~2 cm off.

**Working explanation.** Each noisy correction depends on a random kick; from 20 training demos BC can't learn a consistent correction direction. Candidate fixes: many more shaky demos; lower noise; action chunking (ACT, ROADMAP's next policy), which smooths over per-step noise.

## Step 6 — 100 shaky demos (2026-09-15, running)

User call: option 1 (more data) over lower noise or switching to ACT.
Pipeline: record 100 kept episodes at sigma 0.05 → `data/robosuite/pick_place_cube_noise005_n100`; train BC (default config, seed 0) → `experiments/bc_noise005_n100`; eval on seeds 1000–1019 → `experiments/bc_noise005_n100/eval`. Log: `experiments/bc_noise005_n100/pipeline.log`.
Prediction: if 20 demos were simply too few, small-correction direction (cosine) rises well above 0.47 and success rises above 0/20; if success stays near 0, per-step noise itself is the wrong kind of data for single-step BC.

**Recording (100 kept):** 23,359 frames (4× the 25-demo set), 140 attempts, expert success 71% (35 no_grasp, 3 dropped, 2 wrong_target). Lower than the 83% on 30 attempts — the wider seed range includes more setups the shaky expert can't grasp; failures are discarded.

Training was restarted ~1 min in: the pipeline piped Python through grep into the log, both block-buffered, so the log stayed empty. Relaunched with `PYTHONUNBUFFERED=1` and `grep --line-buffered`.

**Result: 2/20 (10%).** Training 9.3 min (60 epochs), best held-out action error 13.86 mrad (25 shaky demos: 18.65; clean: 9.6).

| model | demos | success | failures | seeds solved |
|---|---|---|---|---|
| BC, clean | 25 | 5/20 | 14 no_grasp, 1 wrong_target | 1003, 1009, 1010, 1012, 1014 |
| BC, shaky | 25 | 0/20 | 20 no_grasp | — |
| BC, shaky | 100 | **2/20** | 18 no_grasp | 1011, 1015 (neither solved by clean BC) |

More data helped (0 → 2) but shaky BC is still below clean BC, and single runs throughout — 2 vs 5 out of 20 is within what run-to-run variation could produce. Results: `experiments/bc_noise005_n100/eval/bc_on_robosuite_pick_place_cube/results.json`.
Checking the step-6 prediction (does correction direction improve with 4× data?): `scripts/copycat_test.py` on this model.

**Correction direction vs data size** (`scripts/copycat_test.py`, held-out frames, cosine between predicted and label correction; "err vs copy" = model error / error of just holding the current pose):

| correction size | clean BC (25) | shaky BC (25) | shaky BC (100) |
|---|---|---|---|
| 10–25 mrad | 0.85 | 0.49 | 0.54 (only 10 frames) |
| 25–45 mrad | 0.87 · err 23.6 vs copy 34.0 | 0.46 · err 47.2 vs copy 35.3 | **0.66 · err 33.3 vs copy 35.4** |
| 45–80 mrad | 0.95 | 0.71 | **0.85** |

- Prediction confirmed in direction: 4× data clearly improves correction aim (0.46 → 0.66 on medium corrections) and error drops from worse-than-standing-still to about equal to it.
- Still well short of clean BC's aim on clean data (0.87). Consistent with 0 → 2/20: moving the right way, not there yet.
- Reading: per-step shake corrections are learnable but data-hungry for single-step BC. Options: more data (~400), lower noise (smaller, more consistent corrections), or action chunking (ACT).

## Step 7 — Half the shake, 100 demos (2026-09-15, running)

User call: option 1. sigma 0.025 (half of 0.05), 100 kept episodes → `data/robosuite/pick_place_cube_noise0025_n100`; BC default config seed 0 → `experiments/bc_noise0025_n100`; eval seeds 1000–1019. Log: `experiments/bc_noise0025_n100/pipeline.log`.
Why: smaller kicks → smaller, more consistent corrections, easier for single-step BC. Trade-off: less off-centre coverage (sigma 0.03 on 25 seeds gave worst 2.19 cm vs 2.70 at 0.05).
Prediction: success above 2/20 and correction aim closer to clean BC's; if it lands near clean BC's 5/20 with no gain, the shake isn't buying anything and the next move is ACT.

**Result: 20/20 (100%).** Recording 100 kept / 103 attempts, expert success 97% (2 no_grasp, 1 dropped), 21,909 frames. Training 8.7 min, best held-out action error **7.93 mrad** (lowest of any run; clean 9.6, sigma 0.05 ×100 13.86). Eval median cycle 7.2 s (expert 6.9 s).

| model | demos | success |
|---|---|---|
| BC, clean | 25 | 5/20 |
| BC, sigma 0.05 | 25 | 0/20 |
| BC, sigma 0.05 | 100 | 2/20 |
| **BC, sigma 0.025** | **100** | **20/20** |

Solves every seed, including all 15 clean BC failed.

**Not yet attributable.** Two things changed at once vs the clean baseline: half-strength shake *and* 4× the demos. Missing control: 100 **clean** demos. Also single training run and 20 eval episodes.
Results: `experiments/bc_noise0025_n100/eval/bc_on_robosuite_pick_place_cube/results.json`.

## Step 8 — Control: half shake, 25 demos (2026-09-15, running)

User call, instead of 100 clean demos: sigma 0.025 with 25 demos isolates the shake at the original data size — compare directly to clean BC 25 (5/20).
→ `data/robosuite/pick_place_cube_noise0025`, `experiments/bc_noise0025`, log `experiments/bc_noise0025/pipeline.log`.
Reading: near 20/20 → the half shake is the fix; near 5/20 → the 4× data did most of the work (then 100 clean demos is the next control).

**Result: 4/20 (20%).** Recording 25 kept / 26 attempts (expert 96%), 5,494 frames. Training 2.3 min, best held-out error 13.96 mrad. Eval median cycle 11.1 s, 16 no_grasp.

| demos ↓ · shake → | none | half (0.025) | full (0.05) |
|---|---|---|---|
| 25 | 5/20 | **4/20** | 0/20 |
| 100 | *not run* | **20/20** | 2/20 |

**Reading.** At 25 demos the half shake buys nothing (4/20 ≈ 5/20). The 20/20 needs 100 demos. Whether clean data alone at 100 gets there, or data and shake together, is the one cell left: **100 clean demos**.
Full shake is harmful at both sizes — its corrections are too large/inconsistent for single-step BC to learn.
Results: `experiments/bc_noise0025/eval/bc_on_robosuite_pick_place_cube/results.json`.

## Step 9 — Control: 100 clean demos (2026-09-15, running)

User call. Fills the last cell of the grid. → `data/robosuite/pick_place_cube_n100`, `experiments/bc_clean_n100`, log `experiments/bc_clean_n100/pipeline.log`.
Reading: ~20/20 → more demos alone is the fix and the shake is not needed at this size; well below 20/20 → half shake and data work together.

**Result: 11/20 (55%).** Recording 100/100 (expert 100%), 21,295 frames. Training 8.5 min, best held-out error **4.52 mrad** (lowest of all runs). Eval median cycle 7.0 s, 9 no_grasp.

### The grid (single training run each, eval seeds 1000–1019)

| demos ↓ · shake → | none | half (0.025) | full (0.05) |
|---|---|---|---|
| 25 | 5/20 | 4/20 | 0/20 |
| 100 | **11/20** | **20/20** | 2/20 |

**Conclusion.**
- **Data and half shake work together.** More clean data helps (5 → 11). At 100 demos, adding half shake takes it to 20/20; half shake at 25 demos does nothing (4/20). Half×100 solves all 9 seeds clean×100 fails, and loses none.
- **Full shake hurts** at both sizes: corrections too large/inconsistent for single-step BC.
- **Held-out error is again the wrong metric.** Clean×100 has the best error of any run (4.52 mrad) and scores 11/20; half×100 has 7.93 and scores 20/20.
- Caveats: one training run per cell; 20 eval episodes. The 20-vs-11 gap is large (Fisher exact p ≈ 0.001 at these counts), but the 3-seed rule hasn't been applied.

Results: `experiments/bc_clean_n100/eval/bc_on_robosuite_pick_place_cube/results.json`.

## Step 10 — Confirm with 3 training runs (2026-09-15, running)

Training seeds 1 and 2 (seed 0 above) for clean×100 and half×100, same datasets and config, run as two parallel streams.
Outputs `experiments/bc_{clean,noise0025}_n100_s{1,2}/`, logs `experiments/confirm_n100/{clean,half_shake}.log`.

## PENDING — come back after ACT (user call, 2026-09-15)

- ~~Not committed~~ Committed 2026-09-15 (code, notes, scripts, per-run summaries and results; run logs deliberately not kept).
- **README results matrix** not updated with the grid.
- Decide which dataset becomes the default for later policies (half shake ×100 is the current best).

**Result: confirmed across 3 training runs** (eval seeds 1000–1019 each):

| model | seed 0 | seed 1 | seed 2 | total | range |
|---|---|---|---|---|---|
| BC, 100 clean demos | 11/20 | 15/20 | 6/20 | **32/60 (53%)** | 30–75% |
| BC, 100 half-shake demos | 20/20 | 18/20 | 20/20 | **58/60 (97%)** | 90–100% |

- Half shake is reliably better at 100 demos: its **worst** run (90%) beats clean's **best** (75%).
- Clean data is also much less stable run to run (6 to 15 of 20) — the single clean×100 run (11/20) happened to sit near its average.
- Held-out error again doesn't track success: clean seeds 1/2 have 4.40 / 5.22 mrad (lower than any half-shake run, 7.69–7.89) and score 75% / 30%.

**Timing note.** Two trainings in parallel took 15.3–15.7 min each vs 8.5–8.7 min alone (CPU-bound image decoding across all 20 cores; GPU mostly idle). Total ~41 min vs ~38 min sequential — **parallel was no faster**. Run trainings one after another on this machine.

Per-run outputs: `experiments/bc_{clean,noise0025}_n100_s{1,2}/`; logs `experiments/confirm_n100/`.
