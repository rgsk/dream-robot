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
