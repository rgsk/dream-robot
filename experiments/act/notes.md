# ACT — log

Goal: the roadmap's next policy. Does action chunking fix what BC couldn't — the stall ~2 cm beside the cube — without needing shaky demos?
Context: `experiments/noisy_expert/notes.md`. BC reference cells (eval seeds 1000–1019):

| demos | no shake | half shake (0.025) |
|---|---|---|
| 25 | 5/20 | 4/20 |
| 100 | 32/60 (3 runs) | 58/60 (3 runs) |

---

## Step 1 — Build ACT (2026-09-15, in progress)

**Decisions.**
- Written from the ACT paper's structure (Zhao et al. 2023) in the same shape as `policies/bc/`. LeRobot's ACT consulted for default *values* only (chunk 100, latent 32, KL weight 10, 4 encoder / 1 decoder layers, dim 512), per the first-implementation rule.
- **Sized for this data, not the paper's.** The paper uses a pretrained ResNet-18 per camera (~11M each) on 50 demos from real ALOHA. Here: BC's small from-scratch conv stack as the image backbone, transformer dim 256. Keeps ACT vs BC a comparison of architecture, not of pretrained features.
- **Chunk 50 steps** (~1.7 s at 30 Hz; the paper's 100 at 50 Hz is 2 s). Episodes are ~210 steps.
- **Temporal ensembling** at rollout: re-plan every step, average all overlapping predictions for the current step with exponential weights (paper's m = 0.01).
- Reuses BC's episode split, random-shift augmentation, normaliser and L1 loss, so differences in the matrix are the architecture.
- First run: **25 clean demos**, the matrix baseline cell (BC 5/20).

**Size, chosen by measurement** (`scripts/act_profile.py`: one training step, batch 64, 2 cameras at 128×128, chunk 50, RTX 4060):

| variant | params | ms/step | GPU s/epoch, 25 demos | 100 demos |
|---|---|---|---|---|
| 16×16 image grid, 4 encoder / 4 CVAE layers | 7.69M | 385 | 25.4 | 128 |
| 8×8 grid | 7.98M | 100 | 6.6 | 33 |
| 2 / 2 layers | 4.53M | 205 | 13.5 | 68 |
| **8×8 grid + 2 / 2 layers (default)** | **4.83M** | **64** | **4.3** | **21** |

Attention over 2 × 256 image tokens was the cost. 8×8 grid (4 conv blocks, one feature cell ≈ 16 px) + 2 encoder / 2 CVAE layers is ~6× faster, so 100 epochs on 25 demos is ~10 min instead of ~50. Decoder stays 1 layer. Revisit the grid if ACT misses the cube by a cell-sized amount.

Smoke test on the 25 clean demos (original 16×16 size, 2 epochs): runs end to end, 1.1 min, held-out first-action error 117 → 93 mrad. Checks: 17 ACT unit tests (chunk padding never crosses episodes, padded steps invisible to the CVAE, temporal ensembling weights, reset, checkpoint round trip); full fast suite 153 passed.

## Step 2 — ACT on 25 clean demos (2026-09-15, running)

The matrix baseline cell (BC 5/20). Default config: chunk 50, dim 256, 8×8 grid, 2 encoder / 1 decoder / 2 CVAE layers, latent 32, KL weight 10, lr 1e-4, 100 epochs, batch 64, seed 0.
→ `experiments/act_clean25/`, eval `experiments/act_clean25/eval/`, log `experiments/act_clean25/pipeline.log` (not committed).
Prediction: chunking fixes the stall beside the cube → above BC's 5/20. If it's also at ~5/20, the clean-demo data gap (no correction examples) dominates regardless of architecture, and ACT needs the half-shake data too.

**Result: 16/20 (80%)** vs BC 5/20 on identical data, split and eval seeds. Single training run.
Training 8.5 min (100 epochs), 4.83M params, best held-out first-action error 10.48 mrad (BC: 9.6).

| | success | failures | median cycle |
|---|---|---|---|
| BC, 25 clean | 5/20 | 14 no_grasp, 1 wrong_target | 7.6 s |
| **ACT, 25 clean** | **16/20** | **3 dropped, 1 no_grasp** | 7.1 s |

- Solves every seed BC solves, plus 11 more. Fails 1000, 1001, 1008, 1015.
- **The failure moved.** BC stalls at the grasp; ACT mostly gets the cube and drops it in transit. Chunking fixes the stall beside the cube even without correction examples in the data.
- Held-out error is again no guide: ACT's 10.5 mrad is slightly *worse* than BC's 9.6, and it scores 3× higher.
- **CVAE latent collapsed.** KL fell from 0.92 to 0.001 by epoch 80: the decoder ignores the latent. Expected on a unimodal scripted expert (one way to do the task, nothing for the latent to encode); relevant for T2 (two valid bins), not here.

Results: `experiments/act_clean25/eval/act_on_robosuite_pick_place_cube/results.json`.

## Step 3 — Confirm with 3 training runs (2026-09-15, running)

User call. Seeds 1 and 2, sequential (parallel was no faster on this machine). Outputs `experiments/act_clean25_s{1,2}/`, log `experiments/act/confirm_clean25.log`.

**Result: confirmed, 49/60 (82%).**

| run | success | failures | best held-out first-action error |
|---|---|---|---|
| seed 0 | 16/20 | 3 dropped, 1 no_grasp | 10.48 mrad |
| seed 1 | 16/20 | 4 no_grasp | 11.15 mrad |
| seed 2 | 17/20 | 3 no_grasp | 9.24 mrad |

- Very stable run to run (16–17/20), unlike BC on clean data (6–15/20 at 100 demos).
- **Correction to step 2:** "the failure moved from grasp to transport" held for seed 0 only. Across all three runs failures are 8 no_grasp, 3 dropped — still mostly the grasp, just far fewer of them.
- BC on 25 clean demos has one run (5/20); ACT's worst run is still 3× it.

Matrix so far (25 clean demos, eval seeds 1000–1019):

| policy | success |
|---|---|
| expert | 20/20 |
| BC | 5/20 (1 run) |
| **ACT** | **49/60 (3 runs)** |

**Failure-label check** (user watched the failure videos and read all failures as drops in transit; `scripts/failure_trace.py` replays run 2 seeds 1001, 1002 and run 3 seeds 1001, 1008, all labelled no_grasp, same outcomes as eval):

| run · seed | closest gripper–cube | gripper opening after close | cube max rise | cube moved toward bin |
|---|---|---|---|---|
| s1 · 1001 | 3.9 cm | 0.01–0.03 | 0.45 cm | 2.0 cm |
| s1 · 1002 | 3.0 cm | 0.01–0.03 | 0.00 cm | 0 |
| s2 · 1001 | 3.9 cm | 0.01–0.02 | 0.41 cm | 0.6 cm |
| s2 · 1008 | 4.0 cm | 0.01–0.02 | 0.12 cm | 0 |

Sequence in all four: descends beside the cube (3–4 cm off), closes fully at t≈80–120 (opening ~0.01 = nothing between the fingers; holding the cube reads ~0.5–0.7), then performs the whole carry empty — lifts, moves to the bin, opens at t≈240. The cube stays on the table.
By the measurements the no_grasp label is right: **closed on air, then an empty carry**, which can look like a drop from the front camera. Frame check for the user: `scratch/act/run2_seed1001_close_and_carry.png`.
Note vs BC: BC's grasp failure was *freezing* beside the cube with the gripper open; ACT's is *committing* to a grasp-and-carry plan from a 3–4 cm miss. Chunking removed the stall, not the aim error.
Frame check (run 2, seed 1001, front + wrist): at t≈110 the fingers close on the cube's edge (rise +0.4 cm, cube visible off-centre between the fingers in the wrist view); by t=140 the cube is back on the table at its start position while the empty gripper lifts and carries to the bin. So: **slips out at the grasp from an edge pinch**, not a drop in transit. The label no_grasp is fair; "edge pinch, slipped at grasp" is the precise description.

## Step 4 — ACT on 100 half-shake demos (2026-09-15, running)

User call: one training run on `data/robosuite/pick_place_cube_noise0025_n100` (BC: 58/60 over 3 runs).
**50 epochs, not 100**: 4× the frames, so still 2× the gradient steps of the 25-demo runs, ~20 min instead of ~40. Otherwise default config, seed 0. ~24 s/epoch.
→ `experiments/act_half100/`, log `experiments/act_half100/pipeline.log`.
Question: ACT's remaining failures are aim errors (edge pinches 3–4 cm off). Does half-shake data fix them for ACT the way it did for BC?
Not yet tried: ACT on 25 half-shake demos (BC there: 4/20).

**Result: 20/20 (100%).** Single run. Training 19.1 min (50 epochs, ~23 s/epoch), best held-out first-action error 8.80 mrad at epoch 44. Eval median cycle 7.3 s (expert 6.9 s); every episode 210–234 steps — no slow recoveries, no near-timeouts.

| data | BC | ACT |
|---|---|---|
| 25 clean | 5/20 (1 run) | 49/60 (3 runs) |
| 25 half shake | 4/20 (1 run) | *not run* |
| 100 clean | 32/60 (3 runs) | *not run* |
| 100 half shake | 58/60 (3 runs) | **20/20 (1 run)** |

Matches BC's best cell; with BC already at 97% there, a single 20/20 can't separate them. The informative ACT cells now are 25 half shake (does ACT need 100 demos to use the corrections?) and 100 clean.
Results: `experiments/act_half100/eval/act_on_robosuite_pick_place_cube/results.json`.

## Step 5 — ACT on 25 half-shake demos (2026-09-15, running)

User call. `data/robosuite/pick_place_cube_noise0025` (BC: 4/20). Default config, 100 epochs, seed 0, same as the 25-clean runs.
→ `experiments/act_half25/`, log `experiments/act_half25/pipeline.log`.
Reading: well above ACT's 49/60 on clean → the corrections fix ACT's aim even at 25 demos; about the same → ACT needs more demos to use them.

**Result: 14/20 (70%).** Single run. Training 9.1 min, best held-out first-action error 16.07 mrad (clean 25: 9.2–11.2). 6 no_grasp (seeds 1004, 1008, 1011, 1015, 1016, 1017), all timing out at 600 steps; successes 213–231 steps.

| data | BC | ACT |
|---|---|---|
| 25 clean | 5/20 (1 run) | 49/60 (3 runs; 16, 16, 17) |
| 25 half shake | 4/20 (1 run) | **14/20 (1 run)** |
| 100 clean | 32/60 (3 runs) | *not run* |
| 100 half shake | 58/60 (3 runs) | 20/20 (1 run) |

**Reading.** At 25 demos the half shake does not help ACT (14 vs 16–17 per clean run) — same pattern as BC (4 vs 5). The corrections only pay off with ~100 demos, for both policies. Seed 1008 fails for ACT on both 25-demo datasets.
Architecture is what matters at 25 demos (ACT ~3× BC); data (100 half shake) is what takes both to ~100%.
Results: `experiments/act_half25/eval/act_on_robosuite_pick_place_cube/results.json`.

## Step 6 — ACT on 100 clean demos (2026-09-15, running)

User call. Last empty cell. `data/robosuite/pick_place_cube_n100` (BC: 32/60 over 3 runs). 50 epochs (same as the 100 half-shake run), seed 0.
→ `experiments/act_clean100/`, log `experiments/act_clean100/pipeline.log`.
Reading: ~20/20 → at 100 demos ACT doesn't need the shake; clearly below → the half-shake corrections still matter for ACT.

**Result: 20/20 (100%).** Single run. Training 19.0 min (50 epochs), best held-out first-action error 5.38 mrad. Median cycle 7.0 s (expert 6.9 s) — fastest of any learned policy; episodes 198–224 steps.

### The grid (eval seeds 1000–1019)

| data | BC | ACT |
|---|---|---|
| 25 clean | 5/20 (1 run) | 49/60 (3 runs) |
| 25 half shake | 4/20 (1 run) | 14/20 (1 run) |
| 100 clean | 32/60 (3 runs) | **20/20 (1 run)** |
| 100 half shake | 58/60 (3 runs) | 20/20 (1 run) |

**Conclusion.**
- **ACT doesn't need the noisy expert.** With 100 clean demos it reaches 20/20; BC needs the half-shake data to get there (32/60 → 58/60).
- **At 25 demos architecture is the lever:** ACT ~3× BC on the same data, and shake helps neither.
- **At 100 demos:** ACT is at ceiling on both datasets; BC only on half shake. The noisy expert is a fix for single-step BC's stall, which chunking removes by itself.
- Caveats: ACT 100-demo cells are single runs; 20 eval episodes each, so 20/20 means "≳ 85%" at this sample size, not "perfect".
Results: `experiments/act_clean100/eval/act_on_robosuite_pick_place_cube/results.json`.

**Committed** 2026-09-15 as `4ac4372` (code, tests, notes, scripts, per-run summaries and results; logs not kept).

## PENDING

- ~~README results matrix~~ — done, committed with the README update.
- Single-run cells: ACT on 25 half shake, 100 clean, 100 half shake.
