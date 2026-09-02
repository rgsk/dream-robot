# spec.md — the contract

What a policy is allowed to see, and what it is required to emit. Every task in every
simulator honors this, or its numbers are not comparable to anything else in the matrix.

**Scope of this document today:** observation, action, and the dataset layout on disk. The eval
protocol is marked OPEN at the bottom and gets written when it is built, not before.

---

## Action

```- 
action : float32[8]   absolute target joint positions
```

Same eight degrees of freedom as `observation.state`, same order, same units. **Absolute, not
delta** — `action[t]` and `state[t+1]` live in the same space and can be diffed directly.

Why absolute joint targets and not end-effector deltas:

- It is what a real arm's control interface accepts. Nothing has to be re-derived at transfer.
- It is sim-agnostic. An EE-delta is defined relative to a controller's frame and gain, so the
  same numbers mean different motions in robosuite and Isaac. Joint targets mean one thing.
- It makes the action self-supervising: because `action[t] ≈ state[t+1]`, a policy that has
  learned nothing still produces a physically legal command, and a broken recorder is visible
  as a mismatch between the two columns rather than as a mysteriously bad success rate.

**The scripted expert does not have to think in joint space.** It may plan in end-effector space
exactly as it likes; the recorder writes the joint targets that resulted. Generating the action
and recording the action are different jobs.

## Observation

```- 
observation.state         : float32[8]              joint positions
observation.images.top    : uint8[H, W, 3]          static third-person camera
observation.images.wrist  : uint8[H, W, 3]          wrist-mounted camera
```

`observation.state` layout, and this order is frozen:

| index | meaning | units |
|---|---|---|
| 0-6 | arm joint positions, base to wrist | radians |
| 7 | gripper opening, normalized | 0.0 closed → 1.0 open |

The gripper is **one** scalar even on hardware with two mirrored finger joints. Real arms have one
gripper command; exposing two makes the action dimension a property of the simulator's URDF.

Camera names are canonical and normalized **in the sim adapter**, never downstream. Each backend
names its cameras differently (`agentview`, `robot0_eye_in_hand`, `top`, `cam_high`, ...) and this
is the single quietest way to break cross-sim comparability.

## Banned from `observation.*`

| banned | why |
|---|---|
| object pose, any form | The policy must find the object in the image. Ground truth deletes the task. |
| goal / target pose | Same. |
| derived object relations (`cube_to_target`) | Same, laundered through subtraction. |
| step index, episode fraction, phase id, phase progress | See below. |
| contact flags, grasp booleans, privileged sim state | Not available on hardware. |

The scripted expert **may** read any of these — it is a data generator with privileged access, and
that is exactly what makes it cheap. The prohibition is on `observation.*`, not on the expert.

**Enforced structurally:** the recorder builds observations from an explicit allowlist of keys and
cannot pass anything else through. Not a convention, not a review checklist.

### Why time and phase features are banned

This is the failure mode that motivated the rule, observed in `../rl-prac`. A scripted expert with
fixed per-phase step budgets produces fixed-length episodes with phase boundaries at fixed offsets.
"Where am I in the episode" then genuinely predicts the action — so a policy given `step_fraction`
and a phase one-hot scores well by learning the clock, and the success rate stops measuring
manipulation.

The structural fix is upstream of the observation: **the expert advances phases on a predicate**
(within ε of the waypoint; gripper closed *and* object lifted), never on a step count, with a
timeout as the failure path. Episodes then vary in length and the clock carries no information —
which is what makes the ban cheap to honor rather than a discipline to maintain.

Corollary: state-only BC is a one-day sanity check to prove the pipeline moves, never a reported
result. Reported cells are image-conditioned.

## Rate

30 Hz. One recorded frame per control step, no subsampling, no frame skip.

Note this differs from `../rl-prac`, which ran at 20 Hz. 30 Hz is chosen to match the real-arm
target rather than the prior project, and changing it later invalidates every cycle-time number in
the matrix.

## Allowed to vary between tasks

- Action dimension. Bimanual is 16, not 8. Policies read the dimension from dataset metadata and
  are otherwise unchanged — this is the entire reason a single ACT implementation covers both.
- Number of cameras, as long as names come from the canonical set.
- Episode length, which is variable by design (see above).

## Frozen numbers

| what | value | cost of changing it later |
|---|---|---|
| control rate | 30 Hz | every cycle-time number |
| camera resolution | 128 x 128, both cameras | re-record every dataset |
| state layout | 7 arm + 1 gripper | every checkpoint |

---

## Dataset layout on disk

**LeRobotDataset**, written by `core/record.py`, described by `core/dataset.py`. One directory per
task; policies read the directory and never import sim code.

| feature | dtype | shape |
|---|---|---|
| `observation.state` | `float32` | `(8,)` |
| `action` | `float32` | `(8,)` |
| `observation.images.top` | `video` | `(3, 128, 128)` |
| `observation.images.wrist` | `video` | `(3, 128, 128)` |

Plus LeRobot's own bookkeeping columns — `timestamp`, `frame_index`, `episode_index`, `index`,
`task_index` — and a `task` string on every frame carrying the natural-language instruction.
**That set is the whole file.** A recorded dataset containing any other key is a contract violation,
and it is the checkable form of the ban above: `tests/test_record.py` asserts the written feature
set equals this table.

- **State and action share a shape and a set of element names.** They are the same quantity in the
  same units one step apart, and declaring them identically is what lets a misaligned recorder show
  up as a diff between two columns.
- **Images are stored as video, not PNG frames.** A 215-step episode is ~21 MB raw across two
  cameras and a few hundred kB encoded. The decoder returns `(3, H, W)` float32 in [0, 1]; use
  `core.dataset.frame_to_uint8` to get back to the `(H, W, 3)` uint8 the rest of the repo speaks.
  Encoding is lossy, so nothing may assert pixel equality across the round trip.
- **`fps` equals the task's `control_hz`.** One frame per control step, no subsampling.
- **Episode length varies**, by design (see above). Nothing may assume a fixed length.
- **The terminal observation is not in the dataset.** N steps yield N `(observation, action)` pairs;
  the final observation — the one showing the task already completed — has no action to pair with,
  and inventing one would put a command in the action column that no policy ever issued.
- **Only successful episodes are recorded**, unless a recorder is explicitly asked otherwise.
  Behaviour cloning imitates what it is shown. The noisy-expert work is a different thing that is
  easy to confuse with this one: recoveries from perturbed states are *successes* that start
  somewhere unusual.
- **Provenance travels with the data.** `recording_summary.json` sits in the dataset root with every
  seed attempted, its outcome, its failure mode, and the expert's success rate.

### Recorder integrity check

Because `action[t]` and `state[t+1]` are the same quantity, `|action[t] - state[t+1]|` over the arm
joints is checked on every episode before it is written. Over 25 recorded episodes of robosuite
`pick_place_cube` it runs mean 0.018–0.020 rad with a peak of 0.062 — sitting just above the IK's
`max_joint_step` of 0.060, which is the largest move the expert may command in one step and the
bound the lag is measured against. This is a **mode** check, not a precision one: a controller
quietly running in delta mode, or a recorder writing the wrong column, puts it in the order of
radians. The threshold is 0.25 rad.

---

## OPEN

- **Eval protocol.** Episode count, seed policy, success predicate, video capture rules. Written
  when `core/eval.py` is built, not before. The failure-mode taxonomy it consumes is already closed
  — `core.env_api.FailureMode`.
