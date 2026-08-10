# env.py — forward-build ticket list

Rebuilding `sims/robosuite/tasks/pick_place_cube/env.py` from a dumb v0, one felt requirement at a
time. **Do not open `env.py` until Ticket 9.** Working file: `rough/env_v0.py`.

Method: [[rebuild-forward-under-pressure]] — every ticket is a requirement someone would actually
want, not "now add an abstraction". The written answer is the deliverable; the code is the excuse.

---

## ✅ T1 — Look at what robosuite actually hands you

`suite.make("Lift", robots="Panda", has_renderer=False, has_offscreen_renderer=False,
use_camera_obs=False)`, print all 14 keys with shapes.

### T1 — Result

*Code: `t1()` in `rough/env_v0.py`.*

**Done.** Proved by assertion that `object-state == cube_pos ++ cube_quat ++ gripper_to_cube_pos`
and `robot0_proprio-state` is the concatenation of the nine `robot0_*` keys. `action_dim == 7`.

**What it bought:** `object-state` survives any filter keyed on `"cube"` — the argument for
`build_state` taking named arguments instead of a dict. And the contract is an *allowlist*:
`robot0_eef_pos` is excluded because it was never named, not because it was banned.

Still owed: walk all 14 keys against `spec.md`'s ban table in a comment at the top of the file.

---

## ✅ T2 — Make `action_dim` say 8

**Requirement.** The env accepts 7 absolute joint targets + 1 gripper command. The default is
OSC_POSE — end-effector deltas, which `spec.md` bans.

**Acceptance.**
1. `env.action_dim == 8`.
2. Step with `np.concatenate([obs["robot0_joint_pos"], [0.0]])` a few times and the arm stays roughly
   where it is instead of flinging. **This is the real check** — it proves the numbers mean what you
   think they mean.
3. Print `|action - next_state|` over the arm joints. Expect small and non-zero (the PD loop lags);
   this is `joint_tracking_error` in `core/schema.py`, arrived at yourself.

**Traps** (documented in `sims/robosuite/README.md`): `load_composite_controller_config` flattens
`body_parts.arms.{right,left}` → `body_parts.{right,left}`, so indexing the on-disk shape raises
`KeyError: 'arms'`. And `impedance_mode` must be `"fixed"` for `input_type="absolute"` to work.

**Non-goals.** Cameras, the bin, any class.

### T2 — Result

*Code: `t2()` in `rough/env_v0.py`.*

**Done.** `load_composite_controller_config(robot="Panda")`, then two assignments on
`cfg["body_parts"]["right"]`: `type = "JOINT_POSITION"`, `input_type = "absolute"`. Passed back
through `suite.make(controller_configs=cfg)`. `action_dim == 8`.

**What it bought.**

*The static hold test proves nothing on its own.* Stepping
`concatenate([obs["robot0_joint_pos"], [0.0]])` once gives a tracking error of ~1e-11 — machine
noise on numbers of order 1, i.e. the arm did not move. But "absolute targets work" and "the
controller is ignoring my action entirely" predict that identically. The discriminating test is to
command a target the arm is *not* already at: `action[3] += 0.1`, step, and the error comes back at
8.69e-2 — the joint covered ~0.013 rad of the 0.1 in one 50 ms control step. Only the second test
distinguishes the hypotheses. A passing check that every hypothesis passes is not a check.

*Tracking error is lag, and it has a per-joint floor.* Holding the offset target for 20 steps,
joint 3 decays 8.69e-2 → 1.4e-4 and is still falling; joint 5 stops falling around 4.8e-4. The
floor is not noise — it is the steady-state residual on the joint carrying the arm's weight, which
fixed gains hold against but do not erase. So `joint_tracking_error` in `core/schema.py` cannot
have one "converged" threshold shared across joints, and it is a *lag* signal, not an offset one:
under delta control the same numbers would have flung the arm on step one.

*The joints are not independent.* Driving joint 3 alone lifted the other six from ~1e-11 to 1e-8 —
1e-4. Their controllers are correcting a perturbation nobody commanded, transmitted through the
arm's dynamics. Any per-joint threshold has to tolerate the neighbours moving.

**Traps, as actually encountered.** The `arms` flattening is real — `body_parts` has exactly one
key, `right`. `impedance_mode` was **already** `"fixed"` in `default_panda.json`, so it never had
to be set; the trap bites only when starting from a config where it isn't. Also worth noting the
default that was replaced: `output_max = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]` — six entries, visibly
Cartesian, which is the OSC action shape sitting in plain sight in the config.

**Bugs found in the writing, all one shape.** Three times the comparison was taken across a
mutation the simulator had not yet seen: `env.step(action)` with the return dropped (printing the
reset obs twice); `action[3] += 0.1` printed without an intervening step (the 1.0e-01 was just the
`+=` read back); and `+= 0.1` left *inside* the 20-step loop, ratcheting the setpoint 2 rad and
producing a diverging error curve. Each looked like a physics result and was arithmetic. When a
number looks like a finding, check that a step happened between the cause and the measurement.

---

## T3 — I want images

**Requirement.** Get camera frames out of the env.

**Acceptance.** `obs` now contains image arrays; print their keys, shapes and dtype. Save one to
PNG and look at it.

**What you'll discover.** The keys are `agentview_image` and `robot0_eye_in_hand_image` — nothing
downstream should ever see those names. Also: from here on you need `MUJOCO_GL=glfw`, and
`use_camera_obs=True` requires `has_offscreen_renderer=True`.

**Non-goals.** Renaming them yet. Just look.

---

## T4 — The images are upside down

**Requirement.** Frames that match what a real camera would produce.

**Acceptance.** Save before/after PNGs. The fix is `[::-1]`, and the reason is OpenGL row order.

**Why this ticket exists at all:** unflipped, a policy trains happily on inverted frames and nothing
complains until they meet a second simulator or real hardware. Visible bugs are the cheap ones.

**Non-goals.** Canonical names. Still just looking.

---

## T5 — There is no bin in `Lift`

**Requirement.** A static open-top box on the table for the cube to go into.

**Acceptance.** Subclass `Lift`, override `_load_model()`, append MJCF geoms via `new_body` /
`new_geom` / `array_to_string`. The bin appears **in the camera images**, not just in physics.

**The trap, and do this one deliberately.** Build the walls with `new_geom`'s default `group=0`
first. robosuite treats group 0 as collision-only and its renderers do not draw it — you get a bin
the cube bounces off that **no camera can see**. Screenshot that state before fixing it with
`group=1`. A success predicate passing over an image of an empty table is the most instructive bug
in this repo.

**Also:** name it `goal_bin`. robosuite already ships `left_eef_target_box` / `right_eef_target_box`
indicator geoms parked at `z=-1`, and substring searches match them.

---

## T6 — The gripper should be one number in [0, 1]

**Requirement.** Contract units: `observation.state[7]` is 0.0 closed → 1.0 open. And the action's
gripper entry is the same scale.

**Acceptance.**
1. `robot0_gripper_qpos` is `(2,)` — two mirrored finger joints — and becomes one scalar.
2. The max span is **read from the model** (`jnt_range` via `mj_name2id`), not hardcoded.
3. Sending `1.0` opens and `0.0` closes. robosuite's GRIP convention is `-1` open / `+1` close, so
   the adapter converts: `1.0 - 2.0 * opening`.

**Why one scalar:** real arms have one gripper command. Exposing two makes the action dimension a
property of the simulator's URDF rather than of the robot.

---

## T7 — Did it succeed, and if not, why?

**Requirement.** After each step: is the cube in the bin? If the episode ends without that, which
failure was it?

**Acceptance.** A success predicate (inside the footprint **and** resting low — not merely held
above the bin by the gripper) and a diagnosis returning one of `core.env_api.FailureMode`.

**The trap.** "Was it ever lifted" must be measured against **the cube's own starting height**, not
the table surface. robosuite's placement sampler spawns with a `z_offset` already applied, so an
idle episode measured against the table reads as a successful grasp.

**Read `FailureMode` before writing your own** — the closed set is the point. A free-text reason
string produces a different vocabulary per task and a histogram that cannot be summed.

---

## T8 — Same seed, same episode

**Requirement.** `reset(seed=0)` twice gives the identical cube placement.

**Acceptance.** Assert `cube_pos` matches across two seeded resets, and differs across two seeds.

**The quirk.** robosuite's `UniformRandomSampler` calls `np.random.uniform` directly, so the
**global** numpy RNG is the only seeding hook that exists. That is why `PickPlaceCube.reset(seed=...)`
looks the way it does.

---

## T9 — Promote to `env_practice.py`

**Requirement.** Make it a real adapter that satisfies the contract.

**Acceptance.**
1. Numbers move to `task_practice.yaml`, read by a `TaskConfig` you write *now* — after knowing
   which values there actually are. (The current stub was transcribed before that; expect yours to
   differ.)
2. Observations are built with `core.schema.build_observation`; steps return
   `core.env_api.StepResult`; cameras are renamed to `top` / `wrist` **in the adapter**.
3. `render()` is separate from `observation.images` — wide view for humans, contract resolution for
   the policy.
4. **`check_env_conformance(env)` passes.** That is what makes "honours the contract" a checked
   property rather than a claim.
5. `demo_practice.py` imports `PickPlaceCube` / `TaskConfig` from `env_practice` and still produces
   the video.

**Then diff against `env.py`** and write down every disagreement.

---

## Then

`expert_practice.py` — the phase machine (predicates, never step budgets) plus `ik.py`. The pure
machine is testable with no simulator running, which is the design property worth rediscovering.
