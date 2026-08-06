# spec.md — the contract

What a policy is allowed to see, and what it is required to emit. Every task in every
simulator honors this, or its numbers are not comparable to anything else in the matrix.

**Scope of this document today:** observation and action only. Dataset layout on disk and the
eval protocol are marked OPEN at the bottom and get written when they are built, not before.

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
| camera resolution | OPEN — see below | re-record every dataset |
| state layout | 7 arm + 1 gripper | every checkpoint |

---

## OPEN

- **Camera resolution.** Proposal: `128 x 128`, both cameras, chosen to fit BC and ACT training on
  8 GB of VRAM. Raising it later means re-recording, so it is worth one deliberate decision now.
- **Dataset layout on disk.** LeRobotDataset, version and exact feature keys to be pinned when the
  recorder is written.
- **Eval protocol.** Episode count, seed policy, success predicate, failure-mode taxonomy for the
  histogram, video capture rules.
