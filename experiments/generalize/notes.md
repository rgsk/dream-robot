# Generalize -- does one executor transfer to a task it never saw?

Question: train ONE executor on several pick-place tasks, hold one out. How well does it do on
the held-out task with 0 / 10 / 50 of its demos, and how fast does it pick?
Background: README "Our own executor"; experiments/{cosmos_policy,uwm,vpp,idm,noisy_expert}/notes.md.
Shared failure so far: accurate on the demo path, hovers and never retries off it.

**PARKED (2026-10-22, user call), with Cosmos.** First build the three levers (force-torque
inputs, DAgger, RL) on cheap local policies (BC / ACT / flow matching); bring the ones that
help to Cosmos afterwards. The plan below stands for when this resumes.

## Step 0 · plan, for review (2026-10-21) -- no spend yet

**Experts still run** (`core.run --policy expert`, seeds 1000-1002, `scratch/generalize/expert_check`):

| task | expert | note |
|---|---|---|
| T1 pick_place_cube | 3/3 | |
| T2 pick_place_two_bins | 3/3 | |
| T5 instructed_pick_place | 3/3 | 224 px demos exist (150) |
| T6 two_arm_lift | 3/3 | 16-D action, 4 cameras |
| T7 assorted_pick_place | **none** | needs one (below) |

**1 · Tasks.**
- Train: T1, T2 (random spawn), T5. Hold out: **T7** (same verb, new shapes, sizes, meshes).
- Drop T6: 16-D action and 4 cameras vs Cosmos' 8-D + top/wrist. A second embodiment is its own
  question, not this one.
- Caveat: T1/T2 are near-subsets of T5 -- all cubes. So the seen set is one verb on one shape, and
  T7 zero-shot tests *object* generalization only. Expect near 0 at 0 demos; the 10/50 curve is
  the real number.
- T7 expert to build (local, free): sim-state target + the arm skills already in
  `sims/robosuite/skills.py` (they pick T7 objects: `tests/test_arm_skills.py`). Eval only on
  seeds whose named object is graspable (not `too_wide`). Acceptance: >= 18/20.

**2 · Data** (all 224 px, local, CPU).

| task | demos | why |
|---|---|---|
| T5 | 150 | have them; add shake + retries |
| T1, T2 | 50 each | near-subsets of T5; more is training cost, not new skill |
| T7 | 10, 50 (+20 eval-only seeds apart) | few-shot sets |

- **Half-shake** (sigma 0.025 on half the demos): BC 53% -> 97% on T1 (noisy_expert step 10).
- **Forced-miss retries** (new, ~20% of demos): expert closes 2-3 cm off on purpose, sees the empty
  hand, opens, lifts, re-aims. Targets exactly the hover failure; nothing in the data shows one now.
- DAgger relabelling (policy rollouts, oracle expert via `SimSnapshot`/`ExpertFuture` labels each
  visited state) needs pod rollouts, so it moves to lever (b) below, not the base data.

**3 · Metric.**
- T7 success at 0 / 10 / 50 demos, 30 eval seeds each, failure kinds (no_grasp, dropped, wrong object).
- Seen success: T5 seen + held-out pairs (30 each, compare to 27/30), T1, T2 (20 each).
- Picks/hour = successes / (sim time + call time, whole eval) x 3600. Bar: 1,500/h (2.4 s per pick).
  Today, per success: ~7.5 s sim + ~15 calls x 1.7 s = ~33 s -> ~110/h. Gap ~14x.

**4 · Runs.**
- Baseline: ONE multi-task Cosmos Policy fine-tune on T1+T2+T5, from the LIBERO checkpoint, as
  stage B. **Checkpoint to HF from the pod** (private repo). Few-shot = short continuations from
  it with T7 10 / 50 demos mixed with seen-task replay, not 3 fresh trainings.
- Speed in the same eval: 5 vs 2 vs 1 denoising steps on T5 seen (10 eps each) + call time
  without decoding the future frames.
- Lever (a): wrist force-torque (+ finger contact) as inputs. state[7] is already the
  gripper opening; the new signal is contact. Needs an optional `observation.ee_wrench` in spec.md
  and a re-record (cheap, local). To check first: robosuite's eef force/torque sensor reads
  sensibly on a grasp vs a miss (5 eps, local).
- Lever (b): recovery from its own mistakes. Cheapest first: one DAgger round (Cosmos rollouts,
  oracle labels, short continuation). RL on the 2B model is ~60 episodes/hour at 1.7 s/call:
  500 episodes ~ 8 h ~ $15. If RL, on a small grasp head over frozen features, not the whole model.

**5 · Costs and ladder** (A100 secure $1.79/h, as stage B; base frames ~54k = 1.6x stage B).

| rung | what | where | est. | gate to next |
|---|---|---|---|---|
| R0 | T7 expert, recordings (shake + retries), F/T sensor check | local | $0, ~1 day | T7 expert >= 18/20; retries visible in 3 videos |
| R1 | base fine-tune (~1 pass, ~3 h) + HF upload + eval seen + T7 zero-shot + denoising-step speed | A100 | ~$8 | T5 >= 24/30 (multi-task didn't hurt) |
| R2 | T7 10 / 50 continuations (~0.5 h each) + T7 evals | A100 | ~$4 | curve exists |
| R3 | lever (a) F/T: retrain base + T7-50 | A100 | ~$9 | fewer no_grasp than R2 |
| R4 | lever (b) DAgger round (RL only if it fails) | A100 | ~$5 | fewer hovers than R2 |

- Balance $9.35 covers R1 only, tightly. R1-R2 need ~$12 -> top up ~$5-10 before R1, or cut R1's
  eval to T5 + T7 zero-shot (~$6) and run R2 after a top-up.
- Evals could move to a cheaper L40S pod (stage A: 0.6 s/call there) -- price checked at pod time.

**Decisions for you.**
1. Hold out T7 (and drop T6)?
2. T1/T2 at 50 demos each, not 100?
3. Forced-miss retries in the base data, or keep the base plain and make retries a lever?
4. Budget: top up before R1, or R1 alone on the $9.35?

Next if approved: R0 step 1 -- the T7 expert, 20 seeds, local.
