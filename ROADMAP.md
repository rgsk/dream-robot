# Dream Robot — Roadmap

> The charter. Thesis, ladder, repo contract, and the commercial filter.
> Absorbs the durable parts of the old `pytorch-practice/startup-roadmap.md` and supersedes
> `pytorch-practice/five-weeks-roadmap.md` entirely (both archived there).

## The thesis

An LLM will handle planning. It will not handle the physical part. **I am building the physical part.**

```- 
  LLM planner        "unload the rack, then fold the towels"
       │ skill call: pick(tube_3) / fold(towel)
       ▼
  MY executor        30Hz closed-loop policy, contact-aware
       │ return: success | failure(no_grasp | dropped | timeout) + progress
       ▲───────────── replan on failure
```

Four things are mine to own, because no language model will do them:

1. **Contact and force.** LLMs have no representation of it. Deepest moat.
2. **The fast closed loop.** 30-50 Hz reactivity. The planner runs at ~0.5 Hz and can't.
3. **Geometric grounding.** "The red one" → an actual grasp pose on an actual surface.
4. **Knowing it failed, and saying so.**

**#4 is the layer boundary.** Without the return channel there is no hierarchy — just an open-loop
script. Design that interface early; it is a bigger decision than which policy class fills the box.

### Brain vs body

The policy ladder below is a complete curriculum for the **brain**, and only the brain. A robot that is
actually good at manipulation also needs a **body** stack this repo barely touches:

```- 
BRAIN (this repo)
  BC · ACT · Diffusion Policy · RL fine-tune · reward model · VLA · world model

BODY (deferred to arm-day / production-arm-day)
  inverse kinematics · compliant & force control · 3D perception · teleop + data collection ·
  tactile sensing · deformable handling · safety rating
```

The biggest body-side gap is **compliant / force control** — nothing soft or contact-rich works with a
stiff position controller. That plus teleop/data-collection is where the company actually lives, and it
is the part that cannot be learned from notebooks. Budget a light robotics track once real hardware lands.

### Why the architecture is not the moat

The demos that make companies notable run the same ACT / Diffusion Policy / VLA stack that's in this repo.
What makes them *work* is three things self-learners under-invest in:

1. **Data-collection infrastructure** — teleoperate cheaply, harvest hundreds of clean demos per task.
   The policy is nearly a commodity; the data pipeline is the product.
2. **Reliability engineering** — 6/10 is a demo, 99/100 on a narrow task is a business. That gap is ~90%
   of the work and almost none of the papers.
3. **Task selection** — pick tasks that are valuable, repetitive, and narrow enough to actually nail.

Building the brain from scratch still matters: when a policy fails at 3am I will know *why*. That depth is
the co-founder asset. But keep an eye on the last mile.

## The skill ladder

Ordered by *capability*, not by *method*. A customer cannot buy ACT. They can buy insertion.

| # | skill (sim) | why it's hard | likely method |
|---|---|---|---|
| 1 | reach + grasp rigid, known pose | none — the spine | BC / ACT |
| 2 | place with tolerance | precision | ACT |
| 3 | **grasp from clutter** | perception, not control | 3D perception + grasp sampling |
| 4 | insertion / fitting | contact, force | RL fine-tune |
| 5 | articulated objects (drawer, handle, door) | constrained motion | ACT / DP |
| 6 | scoop / pour granular | forgiving but long-horizon | DP |
| 7 | bimanual hold-and-act | coordination | ACT (its origin) |
| 8 | deformable: flatten, fold | start-state entropy | DP + bimanual |
| 9 | **LLM-planned multi-skill sequence with recovery** | the whole thesis | planner + RM |

Two edits from the older method-ordered plan:

- **Reward model / failure detection moves early.** 85% → 99% is almost entirely detect-and-retry, not
  a better policy. It is simultaneously the commercial lever and the planner interface. Not a capstone.
- **Rung 3 is new.** Clutter grasping was missing entirely, and every real task needs it.

### Build rung 9 first, badly

Top-down, as always. Early — with only skills 1 and 2 working — wire the entire loop:
sentence → LLM emits `pick(cube) ; place(box)` → executor runs → returns success/failure → planner retries.

It will be embarrassing. It also means every later skill drops into a spine that already exists, and
there is a demo video from week 4 onward that improves every rung instead of appearing at the end.

Deliberately inject a failure (yank the cube mid-grasp) and let the planner recover. Those thirty
seconds are more convincing than any success-rate table.

## The sim spine — freeze early

Built once, reused by every rung. Changing any of it later destroys cross-rung comparability, which is
the entire payoff of running a ladder instead of six disconnected projects.

- **Interface contract** (below) — observation and action semantics.
- **Task suite**, 4-6 variants, **language-annotated from day one**. VLA has nothing to condition on
  with a single task, and retrofitting variety in month 3 means re-recording everything.
- **Eval harness**: success rate, cycle time, demos used, wall-clock to train — plus a **failure-mode
  histogram** (missed grasp / dropped in transit / wrong target / timeout / knocked over). The histogram
  is where "what does the advanced method buy" actually lives, and it is the training signal the reward
  model rung consumes later.
- **One results matrix.** Motivation instrument on bad weeks; best portfolio artifact this project produces.

### Interface contract

Cheap to honor, expensive to retrofit. Keep it even though hardware transfer is not the near-term priority.

| | real arm | sim must expose |
|---|---|---|
| action | joint position targets | joint-position control, not EE-delta / OSC |
| `observation.state` | joint positions | same, same order |
| `observation.images.*` | 2 cams (static + wrist) | 2 cams, same names, same resolution |
| rate | 30 fps | 30 fps |
| object pose | **none** | **banned from the dataset** |

The last row is the one that's easy to violate and fatal. The scripted expert may *read* ground-truth
pose to compute actions — it is a data generator. Ground truth must never enter `observation.*`.
Enforce structurally: the recorder only ever writes state + images.

Corollary: state-based BC is a one-day sanity check, not a deliverable. Image-only after that.

## The task matrix

Design tasks to **isolate one failure axis each**, so every cell has a thesis written down in advance.
Tasks that just get vaguely harder produce a grid of numbers with no story.

| # | task | failure axis isolated | expected winner |
|---|---|---|---|
| T1 | cube → box, randomized poses | baseline / comparability spine | everything runs here |
| T2 | cube → **either of two** boxes, both valid | **multimodality** | DP >> ACT ≈ BC |
| T3 | peg into bottle neck, tight tolerance | **precision + sustained contact** | ACT chunking; RL fine-tune biggest delta |
| T4 | pour N balls, cup → container | **long horizon + orientation** | ACT/DP; world model later |
| T5 | "put the **red** cube in the **left** box" | **instruction ambiguity** | VLA only |
| T6 | bimanual: one holds, other inserts | **coordination** | ACT |

**T2 is the highest insight-per-hour item in the whole plan.** Physically identical to T1 — same scene,
same expert code — except the scripted expert flips a coin on which box. BC averages the two modes and
drives the cube into the wall between the boxes. Visually unmistakable, and DP fixes it cleanly.
Costs an afternoon.

**Two demo-data notes.**

- The scripted expert must be **noisy**: randomize start poses wider than eval, inject Gaussian noise on
  commands, jitter waypoints, record some recoveries from perturbed mid-trajectory states. A
  deterministic expert covers a ribbon of state space one trajectory wide, BC falls off it at step 20,
  and the result looks like "BC is broken" when it's actually the data.
- **Diffusion Policy will not beat ACT on scripted demos.** Its edge is multimodality and a hardwired
  expert is unimodal by construction. Human teleop demos supply the multimodality — the same piece of
  work that rehearses the sim→real recording path. It pays twice.

### Controlling combinatorics

6 methods x 6 tasks = 36 runs. That is the version that burns out. Staircase instead:

- **Every method runs T1** — one column where all numbers are directly comparable.
- **Each method also runs the task designed to expose it** (DP→T2, ACT→T3/T6, RL→T3, VLA→T5, WM→T4).
- Fill other cells only when a result surprises.

~12-15 runs. Loses almost nothing. **Write the predicted winner for each cell before running it** — ten
minutes, and it turns a method that doesn't help into a finding rather than a demoralizing week.

## Sim backends, and where sim lies

No single simulator does rigid + granular + cloth well. **Multiple backends coexist in this repo** — not
a migration to plan for, a folder per sim. A new backend can be tried at any time.

| rungs | backend | caveat |
|---|---|---|
| 1-7 | MuJoCo / LeRobot / ManiSkill | fast, LeRobot-adjacent, fine for rigid + granular |
| 6, 8 (granular, cloth) | Isaac (or Genesis) | MuJoCo cloth is not the tool |

Two places sim lies, to be stated in the walkthroughs rather than papered over:

- **Contact stiffness (rung 4).** Insertion in sim teaches the algorithm, not the difficulty.
- **Liquids.** No free-surface fluid sim worth using here. "Fill bottle" becomes: pour ~20 rigid spheres
  (looks like filling, simulates fine) or insert a peg through the neck (the precision version).

## Repo structure — modular, any permutation

One folder per policy, one folder per sim, tasks under each sim. Any policy x any task must be runnable:
ACT on Isaac pick-and-place, Diffusion Policy on LeRobot bimanual handoff, any combination.

**The seam is the dataset, not an env API.** Policies never import sim code.

- **Offline (training):** every task writes a **LeRobotDataset-format** dataset. Every policy reads one.
  `act` trains on a directory and has no idea which simulator produced it.
- **Online (eval / RL only):** a thin `Env` protocol — `reset / step / render / success` over a canonical
  obs-action dict.

This matters practically: **Isaac Sim will not share a Python environment with LeRobot/MuJoCo** (pinned
runtime and Python version). With a dataset-on-disk seam, recording happens inside Isaac's venv and
training inside the policy venv, and nothing conflicts. Only eval needs both in one process — which works
so long as policies stay thin.

```- 
dream-robot/               # repo (hyphen)
  ROADMAP.md               # this file — the charter
  README.md                # the map: what's here, how to run a permutation
  spec.md                  # THE CONTRACT: obs/action schema, dataset format, eval protocol
  dream_robot/             # package (underscore)
    core/
      schema.py            # canonical obs/action dicts + validation
      dataset.py           # LeRobotDataset read/write
      env_api.py           # Env protocol (reset/step/render/success)
      record.py            # scripted-expert + teleop recorders
      eval.py              # rollout loop -> success rate, cycle time, failure histogram
      registry.py          # "isaac/fold_napkin" -> constructor
      run.py               # the permutation entrypoint
    sims/
      isaac/
        README.md          # install, launch, quirks, which venv
        tasks/
          pick_place_cube/ # env.py + expert.py + task.yaml
          unload_beads/
          fold_napkin/
      lerobot/
        README.md
        tasks/
          pick_place_cube/ # SAME task, different sim — the control
          bimanual_handoff/
    policies/
      bc/ act/ diffusion_policy/ rl_finetune/ vla/
        model.py train.py README.md walkthrough.py   # jupytext-paired
  experiments/
    <exp_id>/              # config.yaml · results.json · videos/ · notes.md
  results.md               # the matrix, regenerated from results.json
```

Permutation surface is one command, two flags:

```- 
python -m dream_robot.core.run --env isaac/fold_napkin --policy act
python -m dream_robot.core.run --env lerobot/bimanual_handoff --policy diffusion_policy
```

Scripted experts live **with the task**, never with the policy — they need privileged state, and the task
folder is the one place that's allowed.

### Five rules that keep permutations honest

1. **Policies depend on torch + core only.** Never on a sim package. This is what survives the Isaac venv split.
2. **Read dimensions from dataset metadata, never hardcode.** Bimanual is just a larger action dim — that
   is the entire reason ACT-on-bimanual works with no policy changes.
3. **Normalize camera names in the adapter** to canonical `observation.images.top` / `.wrist`. Every sim
   names them differently; this is the silent breakage across backends.
4. **`pick_place_cube` exists in both sims on purpose.** It is the control: when a number drops it tells
   you whether the *policy* got worse or the *sim* got harder. Without a shared task, cross-sim numbers
   are not comparable and the matrix is decorative.
5. **Every run writes `results.json` + a video.** `results.md` is regenerated, never hand-edited.
   Visibility made structural rather than a habit to maintain.

### Setup vs algorithm, kept separate

- `sims/<name>/README.md` — **setup only**: install, launch, quirks, what's weird about this backend.
- `policies/<name>/walkthrough.py` — **algorithm only**, jupytext-paired, ending in the run-and-see payoff.

When something breaks it is unambiguous which document to open. This deliberately keeps walkthroughs
inside the policy folder rather than in a separate `nb/` tree — "everything about ACT in one folder" wins here.

## Hardware

- **SO-101 (leader + follower) is an instrument, not the product.** Order early — ~14 day lead times and
  variants go out of stock. It sits in a box until needed; nothing on the ladder blocks on it.
  **Buy the pair** — a follower alone cannot record demonstrations.
- **Arm arrival is an interrupt, not a milestone.** When it lands: pause the ladder ~1 week, run
  record → train → eval on real hardware with whatever the best policy is at that moment, resume. The
  interface contract is what makes that a one-week interrupt instead of a restart.
- **Production arm, month 1-2.** Two capabilities the SO-101 lacks and every contact-rich commercial task
  needs: **force/torque sensing**, and a **safety rating** that gets past a customer's floor manager.
  UR5e/UR3e + Robotiq is the integrator-standard pragmatic pick; xArm 6 the budget option; an ALOHA-style
  bimanual setup if deformables become the committed direction.
- Do not over-fit task design to the SO-101's 5-DoF kinematics. It is the first arm, not the last.

## Deployability filter

Every manipulation task that reached paid deployment shares six properties. Impressiveness is not one.

1. **Fixed base, one station.** Mobile manipulation multiplies the failure surface.
2. **Semi-structured start state.** *The* lever — objects arrive as a stack, rack, tray, conveyor.
3. **Low failure cost.** A bad fold gets redone. A dropped plate is breakage, hazard, liability.
4. **High repetition, narrow SKU set.**
5. **Labor pain that isn't purely wage** — turnover, night shift, hygiene, consistency, traceability.
6. **Countable units.** If you can't count it, you can't price it.

Property 2 explains napkin folding: commercial napkins arrive from a laundry service in **flat stacks**.
Laundry from a crumpled pile is a different problem — the crumpled grasp-and-flatten step is unsolved for
reliability. That is the gap between ~80% research folding results and 99+% deployed folding. Not model
quality. Start-state structure.

### Landscape

| task | where the art is | hardest sub-problem | verdict |
|---|---|---|---|
| flat-textile folding (napkins, towels) | deployed, 99+% | consistent flat-edge grasp | best template; well-funded incumbent |
| garment folding, flat start | ~80% research demos | garment variety | plausible, not reliable yet |
| **laundry from a pile** | research | crumpled grasp | **avoid as first task** |
| dishwasher unloading, home | concept demos only | clutter + fragility + safety | avoid; home is the worst quadrant |
| commercial dish room | partial | wet, hot, high failure cost | maybe |
| **machine tending (CNC / molding)** | mature | fixturing, not perception | most bankable, weakest moat |
| **lab / diagnostics sample handling** | mature-ish, fragmented | traceability, sterility | strong solo fit |
| **food portioning / bowl assembly** | deployed at scale | granular, hygiene compliance | strong; top-3 vertical by units |
| each-picking / kitting | mature | clutter grasping | crowded |
| apparel-manufacturing fabric handling | research | limp fabric between operations | second-market relevant, brutal |

Logistics, food service, and semiconductor are ~64% of commercial robot deployments by unit volume.
Not a coincidence — they fit the six properties best.

### Where to aim

1. **Lab / diagnostics sample handling.** Rigid, extremely structured, high repetition. Customers buy on
   consistency and traceability, not wage. Small footprint, reachable reliability bar.
2. **Flat-textile folding for commercial laundry / hospitality.** Not beating anyone to the frontier —
   applying a demonstrated capability where they aren't selling. The easy end of deformable.
3. **Machine tending, small-mid manufacturing.** Boring, fastest to a signed pilot, real cash, thin moat.
   Puts me inside customer sites, and task selection cannot be done from a laptop.

**Geography — build from SF, operate in both.**

- **Primary market: US.** Labor costs there make a straight wage-replacement payback story close, which
  is the simplest sale there is. Aim the first pilots at US customers.
- **India as second market and deployment site.** Real customers, later. The payback story there needs a
  **non-wage** pain — hygiene, consistency, traceability, night shift, attrition, hazard, audit — because
  wage arbitrage alone will not close. Worth knowing before pitching, not a reason to skip the market.
- **India as cheap in-house operations.** Teleop and demo collection run far cheaper from India, and
  per the moat argument above, **data collection is the product**. Running the data pipeline in-house at
  a fraction of US cost is genuine structural leverage, not a side hustle.

Note the distinction: **in-house teleop for my own data is core; selling teleop as a service to other
robotics companies is the thing explicitly rejected below.** Same activity, opposite consequence — one
feeds the dream, the other becomes the company.

## Customers now, revenue later

**Start the conversations now, in sim, with videos.** Function is *task selection*, not selling. Walk
into a commercial laundry or a diagnostics lab, show a 30-second clip, ask "where in your process would
this sit, and what would it have to do?" Free, honest about the stage, and it picks the next rung.

**Explicitly rejected: teleop / data-collection services for other robotics companies.** It would eat the
calendar and quietly become the company. Same for any revenue that routes around the dream.

When revenue does come, the shapes that fit:

- **Paid pilots / PoCs** — a 4-8 week engagement at a customer site, invoiced. Real revenue, and more
  importantly real failure modes. The only honest answer to a "few months" timeline.
- **Sell the outcome, not the robot** — per unit, per hour, per shift. De-risks the customer's decision,
  recurring, and it forces ownership of reliability, which is the actual product.

Autonomous-product revenue in a few months is not realistic for manipulation. Say so out loud.

## Operating rules

Written against a specific failure: three weeks of tabular RL with no visible payoff drained the tank.

1. **No rung runs longer than ~1.5 weeks without a video.** Overrunning → cut that rung's scope, ship the
   video, move on.
2. **Every walkthrough ends with "what this buys / where it fails."** Accumulated across rungs, that
   section *is* the founder asset — calibrated judgment about which capability is reliable enough to
   sell. Write it as if a customer is asking.
3. **One results matrix, identical columns, every rung.**
4. Honest calendar: the full ladder is **~10-12 weeks of sim**. Say the number now so week 7 doesn't
   feel like failure.

## Related repos

- **`../pytorch-practice`** — the learning repo. From-scratch pedagogical implementations of general
  techniques (CNN, LLM/GPT, diffusion, tabular RL). Boundary rule: *from-scratch implementation of a
  general technique lives there; anything that runs against a sim task and emits a `results.json` lives
  here.* Its `archive/` holds the superseded startup and five-week roadmaps.
- **`../world_model`** (branch `learn`) — Genie-style three-stage video world model (video tokenizer →
  latent action model → MaskGIT dynamics), no action labels required. The eventual imagination engine for
  rung 9's planning variant. Parked until the ladder reaches the world-model rung.

## Standing debts

- **Diffusion track** (`../pytorch-practice/new/diffusion/`) is parked, not abandoned. Diffusion Policy
  collects the debt — the denoiser gets rebuilt in a day because it is needed.
- **Tabular RL** (`../pytorch-practice/nb/rl/handwritten/`) is done and not wasted, but the piece that
  matters downstream is policy-gradient / actor-critic on top of a pretrained policy. **RL is fine-tuning,
  never from scratch** — RL from scratch on pixel manipulation is days of wall-clock for a policy worse
  than a 30-line scripted expert, and it is the most motivation-destroying thing in the stack.
