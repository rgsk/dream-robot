# T6 — bimanual coordination

Goal: ROADMAP task T6, "one holds, other inserts / coordination", expected winner ACT.
The real question underneath it is not ACT's success rate. It is **whether the spine survives a
second arm** — ROADMAP rule 2 says "bimanual is just a larger action dim", and that claim has never
been executed. T6 is the test of the architecture, and the video is the payoff.

Context: T1 `experiments/noisy_expert/notes.md`, ACT `experiments/act/notes.md`, T2 `experiments/t2/notes.md`.
Reference numbers, 100 half-shake demos, eval seeds 1000–1019: BC 58/60 (3 runs), ACT 20/20.

---

## Step 0 — scoping: which two-arm task, and what it costs (2026-09-19)

**What is free.** Checked, not assumed:

- **Policies need no change.** `bc/model.py` and `act/model.py` both build one vision encoder per
  name in a `cameras` tuple and take `state_dim` / `action_dim` from config, which is read off
  dataset metadata. Nothing hardcodes 7 joints, 1 gripper or 2 cameras.
- **The schema already says bimanual.** `Embodiment(arm_joints=14, grippers=2)` → `dim == 16` is the
  literal example in `core/schema.py`'s docstring.
- **The scene is free.** robosuite 1.5.1 ships `TwoArmLift`, `TwoArmHandover`, `TwoArmPegInHole`,
  `TwoArmTransport`. No MJCF authoring needed, unlike T1's bin.

**`TwoArmPegInHole` is rejected, despite being the literal T6 row.** It asserts
`gripper_types is None` — the peg is welded to one end-effector and the plate to the other, and
there are no grippers at all. Consequences: `Embodiment` raises on `grippers=0`, and spec.md's
action layout ends in a gripper entry in [0, 1]. Using it would mean either two permanently-constant
fake gripper columns in the dataset, or widening the frozen action contract to accommodate a task
that has no grasp in it. Neither is worth it for a task whose "insertion" is two welded rigid bodies
aligned in free space — which is not contact-rich insertion anyway. It also uses `EmptyArena`, so
it shares no scene with T1.

**Chosen: `TwoArmLift`.** Panda + Panda on a `TableArena`, both with default grippers. Each arm must
grasp its own handle of a pot and the two must lift together; success is
`pot_bottom > table_top + 0.10`. Fits the contract exactly at `dim == 16`, keeps a table like T1, and
its failure mode is coordination made visible: lift out of sync and the pot tips.

It is not "one holds, other inserts". It is the **coordination axis of T6 without T3's precision and
contact axis bundled in**, which is what the task matrix asks of every task — isolate one axis.
Hold-and-insert stays available later as `TwoArmHandover`/a custom scene, once insertion exists.

**Cost, honestly.** The scripted expert is the whole job: two phase machines plus synchronisation
predicates ("neither arm lifts until *both* grippers have settled closed"). Roughly 2× T1's expert.
Everything else is adapter work.

### Open contract question: cameras

`CANONICAL_CAMERAS = ("top", "wrist")`. A two-arm scene has `robot0_eye_in_hand` and
`robot1_eye_in_hand`, and the allowlist has one slot for them. robosuite offers `agentview`,
`frontview`, `birdview`, `sideview`, `robot0_eye_in_hand`, `robot1_eye_in_hand`.

Two ways, decided before recording anything because it is in every frame of the dataset:

- **A. `top` + one wrist.** No spec change. The other arm is blind at the wrist, asymmetric, and any
  coordination failure is then partly an observability failure — the wrong reason to lose.
- **B. Add `wrist_left` / `wrist_right` to the allowlist.** Purely additive; T1 and T2 datasets are
  untouched and still validate. Bimanual records 3 cameras. Matches ACT's origin setup (ALOHA ran 4).
  Costs ~1.5× image bytes and ~1.5× vision compute. Also tests a second unexercised claim: that the
  policies really are generic over camera count.

### Predictions (written before any of it runs)

| | success /20 | dominant failure |
|---|---|---|
| scripted expert | ≥ 18 | — |
| **ACT**, 100 half-shake demos | **≥ 16** | tipped pot on the lift |
| **BC**, 100 half-shake demos | **≤ 8** | one arm stalls, pot tips |

Reasoning for the BC/ACT gap being *larger* here than on T1 (where 100 half-shake demos put BC at
97%): BC's characteristic T1 failure was freezing 1–4 cm off-target with the gripper open. With one
arm that is a stalled episode. With two it is an actively destructive one — the other arm keeps
lifting and tips the pot — so the same per-arm error rate produces a much worse task success rate.
Chunking should help ACT disproportionately, because a chunk commits both arms to a synchronised
plan instead of re-deciding each arm independently every frame. If that reasoning is right, this is
the first task in the repo where ACT's margin over BC *grows* with good data instead of closing.

Also expected: the failure vocabulary needs a bucket. `FailureMode` is a closed set by design, with
"a task that genuinely needs a new bucket adds it here, once, for everyone" written into it.
"Grasped one handle, lifted anyway" is not `NO_GRASP` and not `DROPPED`. Candidate: `DESYNCHRONISED`.
Decide it from what the expert actually does, not now.

---

## Step 1 — the env adapter (2026-09-19)

Decision on the open camera question: **B, additively.** `CANONICAL_CAMERAS` is now
`("top", "wrist", "wrist_left", "wrist_right")`. `wrist` still means what it meant, so every T1 and
T2 dataset validates unchanged; a task uses a subset. Full suite: **189 passed**.

**What was built.**

- `core/schema.py` — the camera widening, plus `BIMANUAL_PANDA = Embodiment(arm_joints=14, grippers=2)`,
  `dim == 16`. This constant was already the worked example in the module docstring; it is now used.
- `core/env_api.py` — `FailureMode.DESYNCHRONISED`.
- `sims/robosuite/tasks/two_arm_lift/` — `env.py` + `task.yaml`, no MJCF (robosuite ships the scene).
- `core/registry.py` — `robosuite/two_arm_lift`.
- `tests/test_two_arm_lift.py` — 8 tests.

**What the contract already handled with no change:** `check_env_conformance` passes at dim 16 and
three cameras without an edit. The 16-dim action and the N-camera observation were designed for and
never exercised; both worked first time.

**The closed-set test earned its keep.** Adding `DESYNCHRONISED` broke
`test_failure_modes_are_a_closed_set` immediately, which is exactly the intent — a new histogram
bucket is a decision about every task's numbers, and it should cost a deliberate edit.

**The one test worth having written.** `test_action_halves_drive_their_own_arm`: hold arm 0 at its
current joint angles, command arm 1 alone, assert only arm 1 moved. A swapped or transposed action
vector is the bimanual wiring bug, both halves are 7 long so no shape check sees it, and a policy
trained through a swap still converges — to a robot that moves the wrong arm. Measured: arm 0 moved
< 0.02 rad, arm 1 moved > 0.10 rad.

**Scene, three seeds:** `scratch/t6/scene.png`. Two Pandas side by side (`parallel`: robot0 at
y = −0.25 is the left arm, robot1 at y = +0.25 the right), red pot with a green and a blue handle,
pot yaw randomised per seed. Handle orientation is the scene's random variable, and it is what
decides where each gripper must go.

### Open: which camera is `top`

`scratch/t6/cameras.png` compares all six at 128 px. The T1 mapping `agentview -> top` is **wrong
here**: agentview frames the pot nicely and puts *both arms out of frame*, so a coordination policy
would see the object and neither gripper. Candidates:

- `frontview` — both arms and the pot; pot is small, handles ~10 px.
- `birdview` — top-down; both arms plus the clearest read of pot yaw, which is the variable.
- `sideview` — occludes one arm. Rejected: asymmetric between the two arms.

**Settled in step 1b: `birdview -> top`.** Adding the marbles decided it. Re-rendered with marbles
in the pot, `agentview` hides them behind the pot wall until they spill, and `frontview` renders the
pot ~20 px wide. `birdview` is the only camera carrying all three things the task depends on: both
grippers, the pot's yaw (the randomised variable), and the marbles. `sideview` stays rejected for
occluding one arm.

Still open: how much the **wrist** cameras carry. Both still point at the wall at reset, and the
hand-rolled joint nudge used for that render never aimed them at the handles. Measure at step 2,
from a frame taken as the expert closes on a handle.

---

## Step 1b — marbles in the pot (2026-09-19)

Tilt was already *measured* (`_pot_tilt_deg`). What it lacked was a **cost**: success was a height
threshold, so a policy could hoist the pot at 45° and score. 8 marbles now sit in the pot; success is
`lift > 0.10 m` **and** ≥ 6 of 8 still inside. A spill reports `KNOCKED_OVER`.

Bonus: this is T4's trick (pour N balls) rehearsed early — rigid spheres standing in for a fluid.

**Three things that bit, all worth remembering:**

1. **Group-1 geoms build massless.** `mass=` / `density=` on the geom is ignored by the compiled
   model, and MuJoCo refuses the model outright: *"mass and inertia of moving bodies must be larger
   than mjMINVAL"*. Fix: an explicit `<inertial>` element, `I = 2/5·m·r²` on all three axes.
2. **`site_xpos` is stale inside `_reset_internal`.** The placement sampler writes the pot's qpos but
   kinematics are not recomputed until `sim.forward()`. Reading the pot centre before that forward
   placed every marble where the pot was *last* episode — which presents as "the marbles fall out at
   reset" and looks like a physics problem rather than an ordering one.
3. **Counting in the pot's bounding box is too generous.** It counts a marble lying on the table
   beside a pot that has rolled onto its side — the exact case the number exists to catch. The check
   is now the inner cavity (inset by wall thickness, floored at the inner base), in the pot's frame
   so it rotates with it.

**Verified:** 8/8 in the pot at reset and after 1.5 s idle across three seeds; a pot rolled ~120°
spills to 4/8 and reports `KNOCKED_OVER`. Full suite **192 passed**. Picture (upright ×2, spilled ×1):
`scratch/t6/marbles.png`.

**Note for the camera decision:** marbles are invisible from `agentview` while they are *in* the pot
— the side view sees only walls. They are obvious the moment they spill. `birdview` would show them
throughout, which is one more point for it over `agentview` as `top`.

---

## Step 1c — at what angle do the marbles actually spill? (2026-09-19)

The point of marbles is that tilting should *cost* something, so the angle they leave at should be the
angle we call a failure. Target: spill at 45°. Rig: `scripts/t6_spill_curve.py`.

**Answer: they cannot. Not at 45°, not at any fill, not at any pot depth.**

```-
 pot depth   marbles     20°   30°   45°   60°   70°   90°  105°
   0.135 m         8       8     8     8     8     8     5     0
   0.135 m       100     100   100   100   100   100    15     0
   0.065 m        75      75    75    75    75    75     -     -
```

**Why, and it is not a simulator artefact.** Rotate the pot by θ about x. Gravity in the pot's own
frame is

```-
  g_pot = Rx(θ)ᵀ · (0, 0, −g) = (0, −g·sin θ, −g·cos θ)
```

The z component stays **negative for every θ < 90°** — gravity still points *into* the pot. A marble
at the rim would have to travel in +z to leave, and nothing pushes it there. The contents simply pile
against the down-slope wall. They only come out when the opening turns past horizontal, which is why
the measured curve is flat to 75°, half-empty at 90° and empty by 105°.

The fluid model (`θ_spill = atan((d−h)/a)`, derived in the script) says the same thing once you feed
it real numbers: this cavity is 12 cm wide and 13 cm deep, so freeboard exceeds half-width until the
pot is brim-full — and brim-full was measured too (75 marbles in a 6.5 cm pot, no spill at 60°).

**Two measurement bugs found on the way, both worth remembering:**

1. **Holding a free body still by rewriting its pose is not the same as it being still.** The contact
   solver computes impulses assuming the pot can recoil, the next pose write throws the recoil away,
   and marbles creep through the 5 mm walls. It read as a spill at 10° and an untilted pot losing a
   marble. Pinning per physics substep instead of per control step helped and did not fix it. The
   measurement only became trustworthy on a **standalone MuJoCo model with the pot as a static body**.
2. **A lattice sized to the spawn box, not the cavity, silently loses marbles.** Surplus marbles stack
   into layers that start above the rim and are outside before the clock starts — a count short by a
   constant at every angle, which looks like an early spill.

**Decision: `tilt_threshold_deg: 45.0` is the numeric failure line; the marbles are the video signal.**
A dumped pot spills visibly and reads as `knocked_over` either way. A 45–90° botched lift is caught by
the tilt number. Cost: nothing. The alternative — authoring a shallow tray with a low lip to get a
tunable spill angle — changes the graspable object and the grasp itself, and buys a prettier failure
mode for a task whose point is coordination.

**Follow-up: does more marbles or bigger marbles fix it?** Both measured, both no.

- **More** — 100 marbles in the deep pot, and a 3-layer fill in a 6.5 cm one: flat to 70°.
- **Bigger** — r up to 0.020 in a single layer, every pot depth from 13 cm down to 4.5 cm: flat to 80°.
  (r ≥ 0.028 cannot form one layer in this cavity, so those rows lose marbles at placement, not to a
  spill — a placement artefact, not a result.)

**The blocker, derived.** A ball does not float over a wall, it pivots over the wall's top edge. Ball
radius r, wall height L above the floor, contact edge at horizontal offset √(2rL − L²) from the ball
centre, which sits at height r. Gravity's along-slope component tips it over, the normal component
holds it back:

```-
  tan θ_escape = √(2rL − L²) / (r − L)
```

Two things fall out. The denominator says **a ball never escapes a wall taller than itself** (L ≥ r →
no solution) — and robosuite's pot wall is 130 mm against a 12 mm marble, ~11× too tall, which is why
neither count nor size moves the answer. Setting θ = 45° and solving 2u² − 4u + 1 = 0 for u = L/r:

```-
  L = 0.293 · r        (a 3.5 mm lip for a 12 mm marble; 9 mm for a 30 mm ball)
```

So a 45° spill needs a **tray with a lip about 0.3 × the marble radius**, not a pot. That formula is
derived and consistent with the measured "never below 90°", but not yet measured directly: the rig's
inside-test bounds contents below the rim, which is false for a tray where marbles stand proud of the
lip. Measuring it means fixing that test and authoring the tray object.

---

## Step 1d — the tray (2026-09-19)

Replaced robosuite's pot with a purpose-built tray so that a 45° tilt actually costs something.
`dream_robot/sims/robosuite/tasks/two_arm_lift/tray.py`. **Full suite: 193 passed.**

**The object.** `TrayWithHandlesObject` subclasses `PotWithHandlesObject` and re-lays-out its geometry:
a flat base, a 13.7 mm lip, and handle bars carried on **posts** at 5.5 cm. The posts are the point —
the original ties handle height to wall height (`handle_z = body_half_size[2] − handle_radius`), so a
merely-shallower pot has its handles lying on the table where the fingers cannot reach them.

It is still **named `pot`**. Every site, body and joint name downstream (`pot_center`, `pot_handle0`,
`pot_joint0`, `pot_root`), every `_check_grasp` against `handle0_geoms`, and every line of the env and
its tests are unchanged. The shape changed; nothing else did. The scene subclass swaps the object by
rebinding the name robosuite's own `_load_model` looks up, rather than forking that method — forking
would copy robot placement, arena construction, the placement sampler and task assembly into this
repo to rot quietly against the installed robosuite.

**The lip height, and why the derivation was not the answer.** A ball pivots over the lip's top edge:

```-
  tan θ_escape = √(2rL − L²) / (r − L)       →  L = r · (1 − cos θ)
```

which is the circular segment's sagitta, and gives 0.29 r for 45°. **Built to it, the tray spills at
20°.** The derivation balances a ball *sitting still*; a ball on a 12 cm tray starts rolling once the
tilt passes the friction angle (atan 0.4 ≈ 22°) and reaches the lip with momentum, hopping a lip that
would statically hold it. So the coefficient is measured, not derived:

```-
  u = L/r    0.29   0.60   0.76   0.80   0.85   0.93
  onset      20°    30°    45°    50°    55°    70°
```

`lip_over_radius: 0.76` — all 9 marbles aboard at 40°, 6 at 45°, against a `required_inside` of 7.

**What the derivation still buys, and it is the load-bearing part:** `L < r`. A ball cannot escape a
wall taller than itself at any angle below 90°. robosuite's pot is 11 r, so no number of marbles and
no marble size was ever going to work — which is what the three previous steps measured the hard way.
A test now pins it: `test_lip_is_shorter_than_the_marble`.

**Scene:** `scratch/t6/scene.png` — rows 1-2 seeds 1000/1001, row 3 rolled to 60° (1 of 9 aboard).
9 marbles at r = 18 mm in a 3x3 lattice, clearly legible from `birdview`, which is `top`.

**Two visual bugs in the first tray, both worth naming.**

1. **It was placed 5 cm in the air.** `CompositeObject` derives `bottom_offset` from `total_size`,
   which assumes the shape fills its bounding box both ways. The tray does not: handles reach 6.5 cm
   up, the base stops 1.2 cm down. The sampler dutifully lifted it by 6.5 cm to "rest" it on the
   table, and it fell on the first frames. Overridden, along with `top_offset` — and `_pot_lift` now
   reads `bottom_offset`, since robosuite's own success check uses `top_offset` and those are only
   the same number for an object symmetric about its centre.
2. **The handles were joined to nothing.** The uprights sat under the bar, 9 cm outboard of the tray.
   One rigid body, so it simulated correctly and looked like a handle hovering beside a tray. Now:
   uprights stand on the tray's own rim, struts run outboard at handle height, bar across the end —
   a closed loop, legible from `birdview`.

Rendering right after `reset()` is what surfaced both. Filming a settled frame would have hidden the
first one entirely.

---

## Step 2 — the two-arm expert, and what the barrier is worth (2026-09-19)

The expert was scoped in step 0 as "roughly 2x T1's expert: two phase machines plus
synchronisation predicates". That is what it is. `two_arm_lift/expert.py`, plus `demo.py`,
9 unit tests and 6 rollout tests. **Full suite: 209 passed.**

**Headline, eval seeds 1000–1019:**

```-
  barrier          success      tilt at the end of the lift     marbles aboard
  on               20 / 20      4–11 deg  (median 9)            9/9 every seed
  off              11 / 20      9–48 deg  (median 32)           7–9
```

Step 0 predicted the scripted expert at ≥ 18/20. It is 20/20 first try, no tuning: the
numbers in `task.yaml` were written before any of it ran and none of them moved.

The ablation is the point. Same expert, same seeds, `sync_barrier: false`, so each arm
advances on its own predicates — and *nine of the twenty episodes end as `DESYNCHRONISED`*,
with the other failures being `dropped`. The eleven that still "succeed" do it at a median
tilt of 32 deg, one of them at 48. Videos: `experiments/t6/videos/expert_two_arm_lift.mp4`
and `..._no_barrier.mp4`.

So the barrier is not a tidiness measure. Without it, **half the recorded demonstrations
would be the failure the task exists to measure**, and the other half would be a tray
hoisted at 30–48 deg — i.e. BC and ACT would be scored on how well they imitate a
desynchronised expert, and the T6 result would say nothing about coordination.

### Design, and the three decisions inside it

**Two phase indices, not one.** With the barrier on, the arms are in lockstep by
construction, so a single shared index would be simpler and would make desynchronisation
*unrepresentable*. That sounds like a safety property and is actually a measurement loss:
the ablation above is only possible because divergence is expressible. Lockstep is instead
derived and tested — `test_the_arms_stay_in_lockstep_whatever_order_they_arrive_in` drives
400 steps of random arrival order and asserts the two indices never differ, and in physics
`test_the_arms_never_leave_each_other_behind` asserts it per frame of a real episode.

**Waiting is not failing.** T1's phase timeout counts steps in a phase and gives up at 5 s.
Under a barrier an arm that arrived first sits on its waypoint for as long as its partner
needs, which would trip that timer and report the *fast* arm as the stuck one. So an arm
whose own predicate is satisfied stops its clock: `waiting` and `timed_out` are separate
flags, and `demo.py` can still say which arm actually could not get there.

**Per-arm settle windows.** `grip_settled` keeps a rolling window of gripper openings. One
shared window would let one arm's fingers vouch for the other's — the single most direct way
to break "neither lifts until *both* have settled closed" while still passing a success test.

### The grasp frame — the part T1 did not need

T1 held the gripper's reset rotation for the whole episode, and got away with it because a
cube is symmetric about z: any yaw grasps it. A handle bar is not, and the tray's yaw is the
scene's randomised variable (measured over 12 seeds: 120 deg of spread, which is exactly
robosuite's sampler range of π ± π/3).

Measured first, in `experiments/t6/scripts/grasp_geometry.py`: the vector between the two
finger bodies, expressed in the grip site's own frame, is (1, 0, 0) for both arms — **the
fingers separate along the site's x axis**. The bar runs along the tray's local x and the
handle hangs off its local ±y, so with d̂ the horizontal unit vector from tray centre to
handle,

```-
  x̂ = d̂                 fingers close across the bar
  ẑ = (0, 0, −1)        approach from above
  ŷ = ẑ × x̂             and the frame closes:  x̂ × ŷ = ẑ
```

R = [x̂ ŷ ẑ] as columns. Both ±d̂ grasp the same bar, so the sign is free — take the one
nearer the wrist's current x axis, which halves worst-case yaw travel from 180 to 90 deg.

Also measured there, and worth having checked rather than assumed: handle0 lands on arm 0's
side of the table for every seed. The sampler's ±60 deg of yaw is not enough to swap them,
so the fixed `gripper0 → handle0` pairing robosuite uses in its own grasp checks is safe.

### What bit

1. **`env._env.sim` is a different object after every reset.** robosuite runs `hard_reset=True`,
   so caching the sim handle gives you the construction-time scene forever. The first probe
   read handle sites through a fresh handle and the tray's body pose through a cached one,
   which reads as "the tray never moves while its handles do" and looks like a broken seeding
   path rather than a stale pointer. `env.py`'s own properties were already right about this
   (`mujoco` says so in its docstring); the probe script was not.
2. **Three cameras do not divide 256.** `observation_panel` scales the camera column to share
   the wide view's height at integer scale and refuses to fake it, so the render height had to
   go to 384 = 3 × 128. Video only; nothing in the dataset changed.

### Two things found that are not the expert's to fix

- **Success does not check tilt.** `_diagnose` runs only on truncation, so the tilt threshold
  never applies to an episode that succeeds — which is how the ablation's seed 1000 scored
  SUCCESS at 48 deg with 8 of 9 marbles still aboard. The marbles are the only thing guarding
  a successful lift, and at 45 deg they are only just starting to leave. A policy that learns
  to hoist the tray at 40 deg would be scored a clean success. Worth deciding before any T6
  numbers are recorded: either `success` also requires `tilt < tilt_threshold_deg`, or the
  threshold is documented as a failure-*diagnosis* number only.
- **`handles_held` flickers under load.** Mid-lift, `_check_grasp` goes False on an arm that is
  visibly carrying the tray — the load spreads the fingers ~2 mm and the contact set changes.
  `_ever_grasped` is latched so `DESYNCHRONISED` is unaffected, but the `DROPPED` branch reads
  the instantaneous value and could mislabel a successful carry that times out for another
  reason.

---

## Step 2b — "make it hold the tray" (2026-09-19)

Question from the T6 review of step 2: instead of gating success on the tilt *angle*, keep the
episode running a few seconds after the height is reached — a tray held at 45 deg spills, and
then the marbles decide it. Rig: `experiments/t6/scripts/hold_after_lift.py`, which samples
tilt / marbles aboard / tray height at 0, 1, 2 and 3 seconds past the moment success fires.

**The clean expert is untouched by a hold. 20/20, every sample:**

```-
  held for        0s      1s      2s      3s
  tray height     0.10    0.24    0.24    0.24   m
  tilt (median)   9       4       3       2      deg
  marbles         9/9     9/9     9/9     9/9
  still above 0.10 m   20/20   20/20   20/20   20/20
```

Tilt *falls* during the hold: the arms pull the tray level as they finish the lift to +0.25 m.
So a hold requirement costs the expert nothing, and at 1 s it costs no static frames either —
the tray is still rising at that point. That last part matters for BC, whose T1 failure mode was
freezing; ending every demonstration with two seconds of stillness is how you teach that.

**The unsynchronised expert's eleven "successes", held for 3 s:**

```-
  dropped the tray back onto the table       1000  1012  1015        3
  still up, but hanging at 66-71 deg         1005  1017              2
  still up, tilt <= 20 deg                   1004 1010 1011 1013 1016 1019   6
```

So the hold works, **but through the height, not the marbles**. A one-armed lift is unstable and
falls within 1-2 s; that is what the extra seconds expose.

**The marbles flagged none of the eleven. Not one, at any hold length.** Seed 1000 ends with the
tray flat on the table and a marble on the floor, and `marbles_inside` still reports 7 of 9 —
a pass. Two reasons, and both are worth keeping in mind:

1. **They arch.** Seed 1005 hangs at 71 deg for three seconds with all 9 aboard
   (`scratch/t6/hold_1005.png`). Nine 18 mm balls in a 12 cm cavity wedge into the low corner and
   jam against each other, the way grain jams in a hopper. The step 1c/1d spill curve was measured
   on a tray tilted *empty-ish and quasi-statically*; a packed tray behaves differently.
2. **The counter is generous by construction.** It counts in the tray's own frame, so once the
   tray is lying on the table the marbles that rolled out beside it are still inside its footprint.
   Step 1b chose that frame to avoid the opposite error (world bounds counting a marble beside an
   upended tray) and this is the mirror-image cost.

**Decision proposed, not yet taken:** success = tray above `lift_height` **sustained for ~1 s**
*and* tilt below `tilt_threshold_deg` at that moment. The hold catches the drops, the tilt catches
the dangle — neither catches both, and the marbles catch neither. Effect on the numbers: expert
with barrier 20/20 (unchanged), without barrier 11/20 -> 6/20, i.e. the ablation gets sharper
rather than noisier. The marbles stay as the video signal they were always best at.

---

## Step 2c — the lip was calibrated against dynamics the task never produces (2026-09-19)

Review question after step 2b: the marbles do not spill because they are too big — shrink them
and they will. Measured, and the answer is that size is not the lever, but chasing it found the
real fault. Rigs: `scripts/t6_spill_curve.py scale`, `experiments/t6/scripts/marble_size.py`.

**Size, in the standalone rig: the hypothesis looks right.** Lip tied to the marble radius, so
the static escape condition is scale free in theory. It is not in practice — roughly 10 deg
earlier per halving:

```-
  r (mm)    n   lip (mm) |  20°  30°  35°  40°  45°  50°  60°  70°   % aboard
      18    9      13.7  | 100  100  100  100   66   66   66    0
      12   25       9.1  | 100  100   60   40   40   40    0    0
       9   36       6.8  | 100   83   50   50    0    0    0    0
       6  100       4.6  |  50   40   30   10    0    0    0    0
```

**Size, in the actual scene: no effect at all.** 12 mm marbles, lip re-tuned to 0.85 r, mass
scaled by r³ so the payload does not change (16 x 5.9 g = 95 g; keeping 20 g each would have
been 320 g against today's 180 g, and that confound was in the first run). Unsynchronised expert:
9 of 11 still pass, both failures on height — and trays hanging at **61, 65 and 72 deg kept all
sixteen marbles**.

**Where the marbles actually are, at 72 deg:** all sixteen flat on the tray floor, pressed against
the down-slope lip (max offset 0.052 m against a 0.060 m inner wall), one layer, at rest. Not
arched, not piled. Simply held.

**The lip, evaluated:**

```-
  tan θ_escape = √(2rL − L²) / (r − L)

  r = 18 mm, L = 0.76 r   ->  76 deg      <- today
  r = 12 mm, L = 0.85 r   ->  81 deg      <- the "fix"
  r = 18 mm, L = 0.293 r  ->  45 deg      <- the derivation, unmodified
```

Both trays were built to hold their contents past 70 deg. Nobody computed the angle the measured
ratio implied. And the measured ratio came from the standalone rig, where marbles are **dropped at
a fixed angle and roll the length of the tray with momentum**, hopping a lip that would statically
hold them. That is a real effect and it is why 0.293 spilled at 20 deg *there*. It is not what the
scene does: the tray tilts over about a second while the arms hold it, the marbles creep to the low
wall and settle. Quasi-static, so the static formula is the right one — step 1d corrected a number
that was never wrong for this scene, using a rig that measures a different regime.

**Built to the derivation, `lip_over_radius: 0.293`, 18 mm marbles, 1 s hold:**

```-
  expert, barrier on    20/20 pass, 9/9 marbles every seed, tilt 0-10 deg
  expert, barrier off    6/11 pass (was 9/11 at lip 0.85 r)
     newly caught BY THE MARBLES:  1005 (69 deg, 6/9)  1017 (64 deg, 6/9)  1016 (21 deg, 5/9)
     still missed:                 1015 (57 deg, 9/9 aboard)
```

So the marbles now do real work, and they cost the clean expert nothing — not even on the jolt
when the grippers close, which was the risk of a 5 mm lip. Seed 1015 is the case for keeping the
tilt gate as well: a fast tilt that has not had time to shed anything by the time the height fires.

**Recommended package, one decision:** `lip_over_radius: 0.293`; success = height sustained ~1 s
AND tilt under `tilt_threshold_deg` AND `keep_fraction` of the marbles aboard. Ablation lands at
5/20 against the expert's 20/20. Also worth doing at the same time: rename `spill_angle_deg` from
documentation to the number the lip is actually derived from, since it now is one.

---

## Step 2d — spill vs tilt vs time, on the proposed tray (2026-09-19)

Review question: is spilling a threshold in tilt, or does a shallow tilt spill too if held long
enough? Rig: `experiments/t6/scripts/spill_vs_time.py`, lip 0.293 r, 9 marbles at 18 mm.

**The rig had to be rebuilt to answer it.** `scripts/t6_spill_curve.py` builds the tray already at
the target angle and drops the marbles in, so they slide its whole length and arrive at the lip with
speed — the error behind the 0.76 r calibration (step 2c). Here the tray stays level and **gravity is
rotated over a 1 s ramp**, which is the motion the arms actually produce, then held.

```-
  tilt |  marbles aboard after holding for 1s  2s  5s  10s  30s
    5° |                                    9   9   9   9   9
   10° |                                    9   9   9   9   9
   15° |                                    9   9   9   9   9
   20° |                                    9   9   0   0   0
   25° |                                    0   0   0   0   0
   45° |                                    0   0   0   0   0
```

**Both matter, but the time-dependent band is narrow.** Below ~15 deg nothing leaves however long you
wait — the marbles sit against the lip in equilibrium. At 20 deg it takes a couple of seconds. From
25 deg up it is gone inside one second.

Consistent with the in-scene numbers from step 2c: clean lifts run at 0–10 deg and keep 9/9; the
unsynchronised seed at 21 deg lost 4 of 9 over the lift; seeds at 64 and 69 deg lost 3 of 9 in the
1 s hold. And seed 1015 at **57 deg kept all nine**, because it had only just reached that angle when
the height fired — the case for keeping the tilt gate as well as the marbles. The marbles catch a
sustained tilt; the angle catches an instantaneous one.

Also worth noting for T4 later (pour N balls): a sphere rolls at any slope — the friction angle
argument that applies to a sliding block does not apply here, which is why the "never" boundary sits
at 15 deg rather than at atan(0.4) = 22 deg.

---

## Step 2e — a 2 s hold makes the angle rule unnecessary (2026-09-19)

Review call: rather than bolting a tilt gate onto success, hold longer and let the marbles report
what a bad angle does. Measured on the proposed tray (lip 0.293 r), hold raised from 1 s to 2 s:

```-
                          hold 1 s              hold 2 s
  expert, barrier on      20/20, 9/9 aboard     20/20, 9/9 aboard, tilt 0-7 deg
  expert, barrier off      6/11                  5/11
  seed 1015 (57 deg)      PASS, 9/9 aboard      FAIL  -- tray back on the table
  seed 1005 (70 deg)      FAIL, 6/9             FAIL, 1/9
  seed 1017 (64 deg)      FAIL, 6/9             FAIL, 5/9
```

Every remaining failure is caught by height or by marbles, so **the tilt gate is not needed in the
success test** — 2 s is long enough that a tray tilted at the moment the height fires has either
shed its load or fallen. `tilt_threshold_deg` stays as a diagnosis number, labelling the failure
`KNOCKED_OVER` in the histogram; it stops being a third success condition.

The five that still pass are genuinely acceptable lifts: tilt 10-21 deg, tray up, marbles aboard.

**Cost:** episodes grow from ~160 to ~220 steps, of which roughly 30 frames are the arms holding
still after the lift completes. That is ~14% of a demonstration spent stationary, and BC's
characteristic T1 failure was freezing, so it is worth watching in the first BC run rather than
dismissing. Against it: holding what you have lifted is a real part of the skill, and the 2 s hold
is what makes the success criterion a single physical statement instead of three thresholds.

**Proposed, for approval:** `lip_over_radius: 0.293`; success = tray above `lift_height`, still above
it 2 s later, with `keep_fraction` of the marbles aboard. Expert 20/20, ablation 5/20.
