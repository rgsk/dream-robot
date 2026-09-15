# T2 — two bins, coin-flip expert (multimodality)

Goal: ROADMAP task T2. Same scene as T1 (`robosuite/pick_place_cube`) plus a second, mirrored bin;
the scripted expert flips a coin for which bin to use each episode. Both bins count as success.

Questions:
1. Does BC average the two choices and put the cube between the bins? (ROADMAP's prediction)
2. Does ACT's CVAE latent get used (on T1 KL collapsed to ~0.00004), and does ACT pick one bin cleanly?

T1 reference, 100 half-shake demos (σ 0.025), eval seeds 1000–1019: BC 58/60 (3 runs), ACT 20/20.
Context: `experiments/noisy_expert/notes.md`, `experiments/act/notes.md`.

---

## Step 1 — Build the task (2026-09-16)

**What.** New task `robosuite/pick_place_two_bins` in `dream_robot/sims/robosuite/tasks/pick_place_two_bins/`.
T1 is untouched; the one shared-code change is `core/registry.py`, whose `expert` factory now
loads the expert from the env's own task package instead of hardcoding T1's.

**Decisions.**
- **Second bin mirrored across the cube**: bins at y = +0.20 (T1's bin, "right" in the front camera)
  and y = −0.20 ("left"). Same carry length to either; the midpoint of the two modes is the spawn area.
- **Env subclasses T1's.** T2 only tests multimodality if everything else *is* T1, so controller,
  cameras, image flip and observation path are inherited, not copied. Overridden: scene, success
  (either bin), failure diagnosis. Test checks every non-bin config value equals T1's.
- **Expert = T1's phase machine + one coin** (`target_bin`, drawn at reset from NumPy's global RNG,
  which `env.reset(seed)` seeds). The choice is a pure function of the seed and is not in the dataset;
  `record.py` writes it to `bin_choices.json` beside the dataset.
- **"Between the bins" detector**: `env.between_bins(cube)` = cube XY over the strip of table between
  the two bins' outer walls, as wide as a bin. In T2's failure histogram **`wrong_target` means exactly
  this**; any other lifted failure is `dropped`; never lifted is `no_grasp`. XY only, so a cube
  hovering undecided over the strip at timeout also counts.
- **Colour: both bins T1's blue** (discussed with user). Distinct colours belong to T5, which must
  randomise them per episode or "green bin" = "left bin". Randomising here would add a
  colour-invariance problem T1 never posed and confound the T2 result; T5 re-records data anyway
  (its expert obeys a prompt, not a coin). `rgba` is per-bin in task.yaml so T5 only edits values.
- Prompt: "put the cube in a bin".

**Checks.**
- `scripts/scene_check.py`: both bins render in the front, top and wrist cameras; expert with the
  coin forced succeeds into each bin on seeds 0–5 (6/6, 203–224 steps, same as T1's ~210).
- `scripts/coin_check.py`: coin is fair and not a visible cue.

  | seeds | left | corr(coin, cube x) | corr(coin, cube y) |
  |---|---|---|---|
  | record 0–99 | 52/100 | −0.14 | −0.16 |
  | eval 1000–1019 | 12/20 | +0.14 | −0.28 |

  Correlations are within chance (±0.10 is 1σ at n=100, ±0.22 at n=20). Seeds 0–6 all happened to
  flip "left", which is why this was checked.
- Tests: `tests/test_pick_place_two_bins.py` (11 slow: conformance, T1-config parity, both bins
  visible, `between_bins` geometry, coin determinism and both outcomes, expert places in each chosen
  bin, idle → no_grasp, registry). T1 slow tests still pass (27 total); fast suite 153 passed.

## Step 2 — Predictions, written before any T2 training (2026-09-16)

**Data cell: 100 half-shake demos (σ 0.025), recording seeds 0–99.** T1's best cell for both
policies (BC 97%, ACT 100%), so a drop on T2 is attributable to the second mode, not data coverage.
Same training configs as T1's cells: BC 60 epochs; ACT 50 epochs, chunk 50, KL weight 10. Eval
seeds 1000–1019.

| | success /20 | failures | bins used (successes) | ACT train KL, final |
|---|---|---|---|---|
| **BC** | **6** (range 3–10) | ≥ 6 `wrong_target` (between bins) | both, roughly even | — |
| **ACT** (z = 0 at rollout) | **≥ 16** | ≤ 2 `wrong_target` | skewed: ≥ 75% into one bin | **≥ 0.1** (T1: 0.00004) |

Reasoning:
- **BC.** Up to the lift, both modes are identical; at the start of the carry the same state has
  two labels. BC's loss is L1, whose minimiser is the *median*, not the mean — for a 50/50 split any
  value between the modes is optimal, so the net lands somewhere in the middle and carries toward
  neither bin. **Risk to the prediction:** closed loop can break the tie. A small drift toward one
  side makes the next frame look like that mode's demos, and BC commits. If BC scores ≥ 15/20, the
  finding is "closed-loop feedback breaks symmetry", not "BC averages".
- **ACT.** The CVAE encoder sees the true chunk, so the bin choice (one bit ≈ 0.69 nats) is exactly
  what it can encode that the observation can't predict: KL should rise from ~0 to around that.
  At rollout the latent is fixed at z = 0, a single point, so the decoder should produce *one* mode
  consistently — clean commitment, but not a 50/50 split. Chunking also commits 50 steps at a time.
  **Direct test of question 2:** roll ACT out with z sampled from N(0, I) instead of 0; if the
  latent carries the choice, bins split roughly evenly.

Diagnostic beyond the harness: `scripts/rollout_paths.py` rolls a policy out on seeds 1000–1019 and
records the gripper's top-down path, where the cube ended (right / left / between / other; resting or
still held), and the **commit step** (first step the gripper is > 0.10 m to one side of the table
centre; bins are at ±0.20). `--sample-z` fixes ACT's latent to a fresh N(0, I) draw per episode
instead of 0. `plot` draws runs side by side.

Expert reference (`experiments/t2/paths/expert.json`): 20/20, **12 left / 8 right**, 0 held,
commit step median 145 (range 138–169) of ~208. So all the ambiguity sits in the last third of
the episode: approach, grasp and lift look identical in both modes.

## Step 3 — Record 100 half-shake demos + expert ceiling (2026-09-16)

`record.py --episodes 100 --noise-sigma 0.025 --root data/robosuite/pick_place_two_bins_noise0025_n100 --verify`
→ log `experiments/t2/record_half100.log`, round-trip video `experiments/t2/videos/dataset_roundtrip.mp4`.

**Result.** 100 kept of 103 attempts (97%), 21,797 frames (T1's same cell: 20,847 + 1,062 = 21,909).
Failed attempts: seed 6 (left), 42 (right), 95 (right); 2 no_grasp, 1 dropped — shake-induced, both bins.
T1's half-shake recording also took exactly 103 attempts with the same histogram (2 no_grasp,
1 dropped): the noise RNG is seeded per episode, so up to the lift the two recordings are the same
shaky trajectories.
**Demo balance: 51 left / 49 right.** Round-trip video decoded from disk looks right.

**Expert through the harness** (`experiments/t2/expert_on_robosuite_pick_place_two_bins/results.json`):
20/20, median cycle 6.9 s (T1: 20/20, 6.9 s). 12 left / 8 right on these seeds. The two-bin scene
costs the expert nothing.

Pipeline (`scripts/pipeline.sh`, log `experiments/t2/pipeline.log`): BC training started 00:16.

## Step 4 — BC on 100 half-shake two-bin demos (2026-09-16)

Training 8.8 min, 60 epochs, best held-out first-action error 7.86 mrad (T1 same cell: 7.93).
→ `experiments/t2_bc_half100/`, eval `experiments/t2_bc_half100/eval/bc_on_robosuite_pick_place_two_bins/results.json`.

**Result: 20/20. Prediction (6/20, many between-bins) was wrong.** Median cycle 7.2 s (expert 6.9 s,
BC on T1 same cell ~7.0–7.8 s). No failures at all, so no averaging failure on these seeds.
Slowest episodes 246 and 268 steps (seeds 1002, 1013), vs expert 208 and 233 — possibly hesitation.

**Paths** (`paths/bc.json`, `scratch/t2/paths_expert_bc.png`): BC uses **both bins, 6 left / 14 right**,
0 between, 0 held. Commit step median 148 (expert 145); the two slow seeds commit late (183, 203) —
visible hesitation, then a clean commit.

**What decides BC's side** (`scripts/side_vs_spawn.py`): the cube's spawn x (distance from robot).
corr(BC goes right, cube x) = +0.48; **all 7 eval seeds with cube x > 0 went right**; x < 0 split
6 left / 7 right. Agreement with the hidden coin 8/20 (chance, as it must be).

**The demos already lean that way** (`scripts/demo_side_vs_spawn.py`): cube x > 0 → 28 right / 24
left; x < 0 → 21 right / 27 left (corr +0.12). A sampling accident of 100 fair coins, which BC
sharpened into a deterministic rule.

**Why no averaging: the loss.** BC trains with **L1** (deliberately, `policies/bc/train.py:191`).
L1's minimiser is the median, and for a two-mode split the median is the *majority* mode, not the
midpoint. So wherever the demos locally lean even 54/46, L1-BC commits to the majority bin.
Averaging into the gap is the **MSE** behaviour; ROADMAP's prediction implicitly assumed MSE.
Closed loop then keeps the commitment: once the arm drifts to a side, the frames look like that
side's demos.

**Consequence for the plan** (user, 2026-09-16): ACT succeeding on T2 no longer shows anything
about multimodality by itself — a deterministic mode-picker already solves the task. Question 2 is
answered only by ACT's final train KL and the sampled-z rollout, not by success rate. ACT run
continues for that. Proposed next cell: **BC with MSE loss** on the same data, to test whether
averaging (between-bins failures) appears when the loss is a mean.

## Step 5 — ACT training, early KL read (2026-09-16, epoch 11 of 50)

Train KL, T2 vs T1's same cell (`experiments/act_half100/training_summary.json`):

| epoch | 8 | 9 | 10 | 11 |
|---|---|---|---|---|
| T1 (one bin) | 0.0033 | — | 0.0022 | 0.0019 |
| **T2 (two bins)** | **0.003** | **0.003** | **0.002** | **0.002** |

**Indistinguishable from T1 so far.** The prediction (final KL ≥ 0.1, about one bit ≈ 0.69 nats at
ambiguous frames) is on track to fail; final value at epoch 50 decides it.

Why the latent may have nothing to carry, even with two modes in the data:
- **The bin is mostly predictable from the observation.** The ambiguity only exists in the ~50
  frames before the commit step (~145); after that the arm's own position says which side. And
  before it, the cube-x lean BC exploited (step 4) is available to ACT's encoder too.
- **The KL penalty (weight 10) prices the latent against that.** If the observation already
  predicts the side most of the time, writing the bit into z costs more KL than it saves in L1.
- **L1 again.** ACT's chunk loss is L1 too, so like BC it can commit to the locally-majority
  mode without any latent at all.

Update epoch 20: KL **0.000** (T1 epoch 20: 0.0004). Collapsed like T1.

**Compute note** (snapshot during ACT training): GPU 99%, CPU 8% — ACT is **GPU-bound** on this
machine (RTX 4060), unlike BC, which is CPU-bound on frame decoding. A bigger rented GPU would speed
ACT; it would not speed BC unless the pod has many CPU cores.

## Step 6 — ACT result, sampled-z test, answers (2026-09-16)

ACT: 50 epochs, 19.4 min, best held-out first-action error 10.1 mrad.
→ `experiments/t2_act_half100/`, eval `experiments/t2_act_half100/eval/act_on_robosuite_pick_place_two_bins/results.json`.

**Train KL, final: 0.00007** (epoch 1: 0.35 → 10: 0.0024 → 20: 0.0005 → 50: 0.00007). T1 same cell
ended at 0.00004. **Collapsed exactly as on T1.** Prediction (≥ 0.1) wrong.

| run (seeds 1000–1019) | success | left / right | between | median commit step | corr(right, cube x) | cube x > 0 → right |
|---|---|---|---|---|---|---|
| expert (hidden coin) | 20/20 | 12 / 8 | 0 | 145 | −0.14 | 3 of 7 |
| BC | 20/20 | 6 / 14 | 0 | 148 | +0.48 | **7 of 7** |
| ACT, z = 0 | 20/20 | 7 / 13 | 0 | 156.5 | +0.47 | **7 of 7** |
| ACT, z ~ N(0, I) per episode | 20/20 | 7 / 13 | 0 | 156 | +0.47 | **7 of 7** |

(`scripts/rollout_paths.py`, `scripts/compare_runs.py`; plot `scratch/t2/paths_all.png`.)

- **Sampled z changes nothing: same bin on 20/20 seeds as z = 0.** The decoder ignores the latent
  entirely. Direct confirmation of the KL number.
- **ACT and BC learned the same rule**: cube spawned farther from the robot (x > 0) → right bin,
  7/7 for both. Otherwise they agree with each other on only 11/20 seeds, so the x < 0 half is
  decided by finer cues that differ between the two nets.
- ACT commits ~10 steps later than BC and the expert (median 156 vs 145–148).
- Open: the ACT panel shows a few left-bin paths arcing over from the right side; the reversal
  check in `compare_runs.py` (gripper > 0.05 m to the other side, then the other bin) flags none.
  Not resolved — possibly the approach, not the carry. Check in the video before relying on it.

**Answers.**
1. **Does BC average and put the cube between the bins? No.** 20/20, zero between-bins. L1 BC
   commits to the locally-majority mode of the demos (step 4); spawn x carries a weak 54/46 lean
   in 100 fair coin flips, and BC turns it into a deterministic rule.
2. **Does ACT's latent get used? No.** KL 0.00007, and sampling z does not change a single choice.
   ACT does pick one bin cleanly — but by the same spawn cue as BC, not via the latent.

**What T2-as-built measured.** With 100 scripted fair-coin demos, both L1 policies turn a
50/50 choice into a spawn-position rule. So this T2 tests "does the policy invent a tie-breaker",
not "can it represent two modes". Two things make it a real multimodality test:
- **Remove the cue**: stratify the coin so every spawn region is exactly balanced (or more demos).
- **Remove the median**: MSE-trained BC, which should average into the gap (ROADMAP's prediction).

Final-KL notes from step 5 retained below for the reasoning.

## Step 7 — RunPod setup + smoke test (2026-09-16, no real training)

User call: prepare the pod for the next T2 runs. Pod: RTX 4000 Ada 20 GB, 48 CPUs, driver 595,
network volume at `/workspace`. Script `scripts/pod_smoke.sh`, log `experiments/t2/pod_smoke.log`.

- Synced: code (no .git/.venv/checkpoints/videos/logs) → `/workspace/dream-robot`, and the T2 dataset.
- System packages the pod image lacks: **`libegl1 libgl1 libglvnd0 ffmpeg`** (it has NVIDIA's EGL
  driver but not the loader MuJoCo opens). Not persistent: reinstall on every new pod.
- `uv sync` 2 m 29 s; torch 2.10.0+cu128, CUDA OK.
- **T2 sim tests with `MUJOCO_GL=egl`: 11/11 passed in 26 s.** Headless rendering works.
- BC 1 epoch: 1.3 min training, 3 m 48 s wall. ACT 1 epoch: 2.3 min training, 2 m 45 s wall.
  Eval 2 episodes (600 steps each) with video: 22 s, 0/2 no_grasp (expected after 1 epoch).
- **Speed not established.** One epoch is dominated by setup (normaliser stats, first video decode
  from the network volume). Locally a 100-demo epoch is ~9 s (BC) / ~23 s (ACT); per-epoch pod time
  needs a 3–5 epoch run, or copy the dataset to pod-local disk first if the volume is the bottleneck.

Pod left running idle.

## Step 8 — T2-fixed: fixed cube pose, exactly 50/50 coin, BC on RunPod (2026-09-16, running)

**Why.** Steps 4–6: with a random spawn, 100 fair coins lean by spawn position, and both BC and ACT
turned that lean into a "far cube → right" rule. Fixing the cube pose removes every scene cue; the
demos at the moment of choosing differ only by the (coin-independent) shake.

**Setup.**
- New registered task `robosuite/pick_place_two_bins_fixed`: T2 with the cube at table centre, zero
  yaw, every episode (checked: identical position and orientation across seeds). T2 itself unchanged.
- Recording: 100 kept half-shake (σ 0.025) demos, **balanced coin** — each episode goes to whichever
  bin has fewer kept demos so far, fair coin on ties — so kept demos are exactly 50/50. Local 6-demo
  check: 3/3, 100% expert success. The resulting order nearly alternates by attempt; nothing the
  policy observes carries the attempt index.
- **Eval with half shake (σ 0.025, seeded per episode, as in recording).** With a fixed pose and a
  deterministic policy, 20 eval seeds would otherwise be 20 copies of one episode. The shake makes
  them differ, and it is exactly the kind of within-episode variation that could tip BC between bins.
  Harness gained an optional `--noise-sigma`; default 0, so T1/T2 numbers are unaffected. Tests: fast
  suite 153 passed, T2 + eval tests 26 passed.
- BC: L1, 60 epochs (as every 100-demo BC cell), trained from pod-local disk.
- Pod script `scripts/pod_fixed_bc.sh`; log mirrored to `experiments/t2/pod_fixed_bc.log`.
- **Restarted once.** First attempt recorded straight onto the network volume: 11 episodes in 4 min
  (~21 s/episode, ~35 min projected; locally ~2.5 s/episode). The recorder sat at ~7% CPU, waiting
  on I/O — it writes a PNG per frame per camera, and the volume is slow on many small files
  (runpod.md measured the same for virtualenvs). Also: the container's CPU quota is **7.65 cores**
  (cfs 765000/100000), not the 48 `nproc` reports. Killed; now records to pod-local disk and
  copies to the volume afterwards.

**Progress (second attempt).**
- Recording: **100/100 attempts kept, exactly 50 right / 50 left**, 6 m 20 s including the verify
  video (19:59:01 → 20:05:21).
- Expert on T2-fixed with half-shake eval: **20/20**, median cycle 7.2 s. 45 s for the 20 episodes.
- BC training started 20:06:07; epoch 13 at 20:11:49.

**BC result (pod): 20/20** with half-shake eval, median cycle 7.1 s. Trained 60 epochs in 11.1 min,
best held-out action error 6.47 mrad. **Prediction (≤ 12/20, ≥ 4 between-bins) wrong again.**
Episode lengths 204–232 steps except seed 1010 (245) and **seed 1012 (315)** — the expert's range on
the same seeds is 207–228, so those two look like hesitation before committing. Which bins, and
whether BC used both, waits on the path diagnostic (still running on the pod). Expert paths on
T2-fixed with shake: 20/20, 11 right / 9 left, commit step median 153.

**BC paths** (`paths/fixed_bc.json`, plot `paths/fixed_paths.png`): **both bins, 14 left / 6 right**,
0 between, 0 held, commit step median 149 (expert 153). Seed 1012 committed at step **247** (normal
140–165) and still placed cleanly in the right bin; seed 1010 at 172. No seed ended undecided.

**Reading.** With no scene cue left, BC still does not average. The eval shake — different per
seed — decides the bin, and BC commits cleanly once it has. This is the closed-loop symmetry
breaking named as the risk in step 2: a small drift toward one side makes the next frames and joint
state look like that side's demos, and L1 goes with the local majority, so the drift is amplified
into a commitment rather than averaged away. The left skew (14/6) is BC's own mild preference at
the tie, not the data's (exactly 50/50). The one visible cost of the tie is seed 1012's ~100-step
hesitation.

So on this task **L1 BC handles two valid goals by picking one per episode**, with the choice set by
noise. What it cannot do is *represent* both — nothing lets you sample or steer the choice — but T2
success rate cannot see that. Averaging into the gap likely needs an MSE loss or a policy that
cannot use closed-loop feedback (open-loop chunks), not just a removed cue.

**ACT on the same demos, local machine** (user call, while BC trains on the pod). Script
`scripts/fixed_act.sh`, log `experiments/t2/fixed_act.log`. 50 epochs, default config, eval with
half shake, paths with z = 0 and z sampled per episode.

Predictions (written before the run):

| | success /20 | final train KL | bins, z = 0 | bins, z sampled |
|---|---|---|---|---|
| **ACT** | **≥ 16** | **≥ 0.05** (T2-random: 0.00007) | ≥ 15 of 20 into one bin | **split: 5–15 right** |

Reasoning: before the commit step nothing in the observation predicts the bin, so the CVAE encoder
(which sees the true chunk) is the only source of that bit — the latent should stop collapsing. At
z = 0 the decoder sees one fixed code, so mostly one bin; a sampled z should pick different bins in
different episodes. If KL still collapses and sampled z changes nothing, ACT breaks the tie
internally exactly like L1 BC, and the latent is not doing multimodality even when it is the only
place the choice can live.

ACT progress read at epoch 40 of 50: train KL printed **0.000** at epochs 30, 35, 40 (log rounds to
3 decimals; exact final value from `training_summary.json` when done). Held-out first-action error
~10 mrad. The KL prediction (≥ 0.05) is on track to fail even with the cue removed.

**ACT result (local): 20/20**, median cycle 7.3 s, 19.0 min training, best held-out first-action
error 9.3 mrad. → `experiments/t2_fixed_act_half100/`.

- **Final train KL 0.000076** (epoch 1: 0.35 → 10: 0.0025 → 50: 0.000076). T2-random: 0.00007;
  T1: 0.00004. **Collapsed again. Prediction (≥ 0.05) wrong.**
- **z = 0: 9 left / 11 right**, 0 between, 0 held. So ACT does *not* pick one bin at z = 0 — the
  prediction "≥ 15 into one bin" was also wrong. Like BC, the shake picks the bin.
- **z sampled per episode: 10 left / 10 right — but the same bin as z = 0 on 19/20 seeds.** Only seed
  1000 changed (right → left, 215 → 250 steps). The split looks even only because the z = 0 split
  already was. The latent carries essentially nothing.
- Commit steps: median ~152, with three late commits (182–186) at z = 0 — mild hesitation, all recovered.
- Plot `scratch/t2/paths_fixed_act.png`.

| T2-fixed, 100 demos exactly 50/50, eval with half shake | success | left / right | between | KL final | z sampled changes bin |
|---|---|---|---|---|---|
| expert (hidden coin) | 20/20 | 9 / 11 | 0 | — | — |
| BC (L1) | 20/20 | 14 / 6 | 0 | — | — |
| ACT | 20/20 | 9 / 11 | 0 | 0.000076 | 1 / 20 |

**Answers on T2-fixed.**
1. **BC does not average**, even with no scene cue and perfectly balanced labels. The per-episode
   shake breaks the tie and closed-loop feedback turns it into a commitment.
2. **ACT's latent is still not used.** It does not need it: the same closed-loop tie-breaking that
   works for BC works for ACT, so encoding the bit in z only costs KL.

**What this says about T2 as a test.** With a scripted expert and a closed-loop policy, "two valid
goals" is solved by *committing to whichever one noise points at*. Success rate cannot separate that
from representing both modes. To see the difference, the test has to ask for something only a
multimodal policy can do — e.g. **sample** the policy repeatedly from one identical start state
(no shake) and check it produces both bins, or measure the action distribution at the decision
point instead of the outcome. Deterministic BC and ACT at z = 0 would give one bin every time;
Diffusion Policy should give both.

**Pod vs local machine, measured:**

| step | local (20 cores, RTX 4060) | pod (7.65-core quota, RTX 4000 Ada) |
|---|---|---|
| record 100 half-shake demos | ~4 min | 6.3 min (on local disk) |
| expert eval, 20 episodes | ~1 min | 45 s |
| BC epoch, 100 demos | ~8.8 s | ~26 s |

**The pod is slower for this workload** — ~1.5× for recording, ~3× for BC. Both are CPU-bound, and the
pod's container gets 7.65 cores against 20 locally (BC also asked for 16 loader workers on 7.65
cores). The pod would only win on GPU-bound ACT/DP, and even then an RTX 4000 Ada is roughly
4060-class; a faster GPU and more CPU quota would be needed for a real speed-up.

**Predictions (written before the run).**

| | success /20 | failures | bins |
|---|---|---|---|
| expert, half shake | ≥ 19 | — | ~50/50 by its hidden coin |
| **BC** | **≤ 12** | **≥ 4 between-bins or hovering** | if it commits, not a fixed rule |

Alternative outcome worth naming in advance: BC puts **all 20 into one bin**. That would mean L1
broke the exact tie by an arbitrary internal preference (initialisation / SGD noise) that the shake
never overturns — no averaging, but also no ability to represent two options.

If final KL stays ~0: T2 with 100 half-shake scripted demos is **not a multimodality test for
L1-trained policies**. The mode choice is either predictable (spawn lean, closed-loop state) or
resolved by the median. Options then: stratify the coin so every spawn region is exactly 50/50
(removes the cue), MSE BC (tests averaging), and/or a smaller KL weight.
