# robosuite — setup and quirks

**Setup only.** Algorithms live in `policies/<name>/walkthrough.py`. When something
breaks, this is the file to open if the answer is "install", "launch", or "why does this
backend behave like that".

## Which venv

The root one. robosuite shares a venv with lerobot and torch — no split needed here.

```bash
uv sync
```

Isaac will need its own venv when it arrives. robosuite does not, and the version pins
below are what make that true.

## Version pins, and why they are not superstition

Both ceilings in [`pyproject.toml`](../../../pyproject.toml) are load-bearing. Neither
package declares the bound itself, so without them uv floats straight into a break.

| pin | reason | symptom if you lift it |
|---|---|---|
| `robosuite>=1.5.0,<1.5.2` | 1.5.2 pins `mink==0.0.5`, which requires `numpy<2.0`; lerobot requires `numpy>=2.0` | `uv sync` fails to resolve. 1.5.1 leaves mink unpinned and gets 1.2.0, which is numpy-2 clean |
| `mujoco>=3.9,<3.10` | robosuite calls `mj_fullM(model, dst, data.qM)`; 3.10 reordered it to `(model, data, dst)` and 3.11 removed `MjData.qM` | `TypeError: mj_fullM(): incompatible function arguments` on env construction, or `AttributeError: 'MjData' object has no attribute 'qM'` |

Floor is 1.5.0 rather than 1.4: `load_composite_controller_config()` is the 1.5 API.
1.4 used a flat controller dict and will not run this code.

## Rendering

```bash
MUJOCO_GL=glfw
```

Measured on this machine:

| backend | result |
|---|---|
| `glfw` | works |
| `egl` | `EGLError` — unresolved |
| `osmesa` | not installed |

**`glfw` works because there is a display (`DISPLAY=:1`).** It is a hidden window on a
real X server, not headless rendering. That is fine locally and blocks nothing today,
but **it will fail on RunPod**, which has no display. EGL is the headless path and is
currently broken here — likely PyOpenGL selecting Mesa over `libEGL_nvidia.so.0`, which
is present. Fix it before the first remote run, not before the next local one.

## Quirks that have already cost time

**Geom group 0 is invisible to cameras.** robosuite treats group 0 as collision-only and
its renderers do not draw it. `new_geom()` defaults to `group=0`. Build scene geometry
with `group=1` or you get objects the physics respects and no camera can see — the
success predicate passes while the images show an empty table, and a dataset recorded
from that teaches a policy to place into thin air. Collision is unaffected by the group.

**Images render bottom-up.** OpenGL row order, passed straight through. The adapter
applies `[::-1]`. Unflipped, a policy trains happily on upside-down frames and nothing
complains until they meet a real camera or a second simulator.

**`env.sim.data._data` goes stale across `reset()`.** Capture model/data *after* the
final reset. A reference taken earlier silently keeps returning the old world — the arm
steers on kinematics that never update and no exception is raised.

**Placement sampling uses the global numpy RNG.** `UniformRandomSampler` calls
`np.random.uniform` directly, so `np.random.seed(seed)` before `reset()` is the only
seeding hook available. That is why `PickPlaceCube.reset(seed=...)` looks the way it does.

**The default controller is OSC, which spec.md bans.** `load_composite_controller_config(
controller="BASIC")` gives `OSC_POSE` — end-effector deltas, `action_dim=7`. The contract
needs absolute joint targets, so the arm's part config is replaced with
`JOINT_POSITION` + `input_type="absolute"`, giving `action_dim=8`. In that mode
`set_goal` assigns `goal_qpos = action` verbatim.

**Nothing clips actions.** `control_limits` reports `[-1, 1]`, so `env.action_spec`
advertises bounds that no code path enforces. Absolute joint targets outside that range
pass through untouched — which is what makes the above work, and also means a genuinely
malformed action fails silently rather than loudly. `core.schema.validate_action` is the
check that actually runs.

**The config loader flattens the arms.** `body_parts.arms.{right,left}` in the JSON
becomes `body_parts.{right,left}` after loading. Indexing the on-disk shape raises
`KeyError: 'arms'`.

**Five harmless warnings per env construction.** `The config has defined for the
controller "left"/"torso"/"head"/"base"/"legs", but the robot does not have this
component.` The BASIC config covers humanoids; a Panda has none of those parts.

**Watch out for `target_box` as a name.** robosuite already has `left_eef_target_box` and
`right_eef_target_box` indicator geoms parked at `z=-1`. Substring searches match them.
This task's bin is named `goal_bin`.

## Running it

```python
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube

env = PickPlaceCube()
obs = env.reset(seed=0)          # obs.state float32[8], obs.images {top, wrist}
result = env.step(action)        # absolute joint targets, dim 8
```

Tests:

```bash
MUJOCO_GL=glfw uv run pytest tests/test_pick_place_cube.py    # slow: real physics
uv run pytest -m "not slow"                                   # core only, instant
```

`tests/test_pick_place_cube.py` runs `core.env_api.check_env_conformance`, which is what
makes "honours the contract" a checked property of this backend rather than a claim in
this README.

## If ROS is installed on the machine

Sourcing ROS puts its python3.12 `site-packages` on `PYTHONPATH`, and Python honours
`PYTHONPATH` regardless of the interpreter's own version — so those packages get injected
into this 3.14 venv. pytest then autoloads ROS's `launch_testing` plugin and dies before
collecting a test. Load ROS on demand from a shell function instead of at shell start.
