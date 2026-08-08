# improvements_practice.md

Scratch list for the `practice` branch. Things noticed while reverse-engineering the
stack from the demo video backwards. **Not a commitment** — take them once the flow is
clear, drop the ones that turn out to be wrong.

Each entry: what, why it matters, sketch of the fix, rough size. Ordered by
size-to-value, not by severity.

---

## A. Defects — small, real, worth fixing

### A1. The demo always exits 0, even when a seed fails

`demo.py` `main()` returns `0` unconditionally. It prints `FAILED` for a seed that
missed the bin and then reports success to the shell anyway, so `&&` chains, `make`,
and CI cannot tell a good run from a bad one. The `raise SystemExit(main())` machinery
is wired up but carries no signal.

**Fix.** Track whether any episode failed; return `1` if so.

**The interesting part is the commit message,** not the diff: `demo.py`'s docstring
insists it is *not* the eval harness and refuses to report success rates. So argue
whether a demo script should fail the process on one bad seed at all — a demo that
exits non-zero is arguably reporting a number. Possible answer: exit code says "the
script ran and the expert worked", not "here is the success rate". Decide and write it
down.

**Size.** Two lines plus a paragraph of reasoning.

### A2. The terminal frame is never captured

`demo.py` `rollout_frames()` appends the panel *before* `env.step()`. On the iteration
where success registers, the frame written is the one from *before* the successful
step — so the moment the cube is actually in the bin never appears, and
`frames.extend([frames[-1]] * hold)` holds the state one step earlier.

33 ms at 30 Hz, so visually almost nothing. But it quietly contradicts the comment
directly above it ("hold the final frame so the placement is readable").

**Fix.** One `frames.append(observation_panel(env.render(), obs.images))` after the
loop, before the hold. Watch out that `obs` has already been rebound to
`result.observation` by then — which is exactly what you want.

**Size.** One line.

---

## B. Additions — build these to understand the stack

### B1. `--annotate`: draw the phase on the video

The single highest-value thing for reverse-engineering. Draw phase name, step-in-phase,
and gripper command onto the wide panel. Then scrubbing to any frame tells you which of
the eight phases is running and why it has not advanced — the state machine stops being
a table in a file and becomes something watchable.

**Design question to answer first, in the commit message:** `spec.md` bans phase id and
step index from `observation.*`. Is drawing them on this frame legal? Work it out from
where `render()` output goes versus where `obs.images` goes. There is a right answer,
and the fact that it is *cheap* to be sure is precisely what the `render` /
`observation.images` split bought.

**Where.** Drawing helper belongs in `core/video.py` next to `observation_panel` —
sim-agnostic. No new dependency needed if drawn with numpy slicing or a small bitmap
font.

**Size.** An afternoon, and it pays for itself immediately in section C.

### B2. Exit-code / failure plumbing has no test

Whatever A1 decides, `tests/` has nothing asserting the demo's exit behaviour. A test
that runs `main(["--seeds", "0"])` and checks the return value would pin it. Note this
is a `slow` test — it builds a MuJoCo scene.

**Size.** Small, but only worth doing after A1's decision.

---

## C. Experiments — no code, just knobs and predictions

`task.yaml` is the control panel. Change **one** number, write the predicted visible
effect down *first*, re-run, watch. ~2 minutes per iteration, ends in a video every
time. A knob whose effect you called correctly is a layer you understand; one that
surprises you is the next thing to read.

| knob | from → to | predict |
|---|---|---|
| `ik.max_joint_step` | 0.06 → 0.15 | smoothness vs tracking error — does `test_actions_are_absolute_joint_targets_that_track` still pass? |
| `ik.damping` | 0.15 → 0.005 | the singularity behaviour λ exists to prevent |
| `grasp_dz` | 0.005 → 0.03 | which `FailureMode` comes back, and does `_diagnose` name it correctly? |
| `drop_height` | 0.085 → 0.03 | bin walls are 0.05 tall — what happens on release? |
| `grip_settle_epsilon` | 2.5e-3 → 2e-4 | the documented past bug; reproduce it, then re-read the yaml comment |
| `cameras.mapping` | swap top/wrist | right panel swaps and *nothing else in the repo changes* — the adapter earning its keep |

Then: break one layer, look only at the video, name the layer. Good candidates —
delete the `[::-1]` in `_observe` (which panel flips? why not both?); latch `cube_pos`
per-step instead of at reset; index the Jacobian with `qpos_idx` instead of dof indices
(prediction: nothing visibly breaks, and *that* is the lesson).

---

## D. Not defects — known and deliberate, listed so they are not re-discovered

### D1. `upscale_nearest` is currently a no-op

Panel height is `256 // 2 = 128` and both cameras are already 128, so the scale factor
is 1 and the function returns the frame untouched. The code is correct and general; the
blockiness argument in its docstring just is not being exercised yet. It starts
mattering the moment render height changes or camera resolution drops.

### D2. `experiments/expert_demo/` has no `config.yaml` / `results.json` / `notes.md`

`.gitignore` says those three *are* committed — "they are the matrix" — and only the
heavy artifacts are ignored. They are missing. **This is correct for now:** `demo.py`
deliberately refuses to emit success rate, cycle time, or a failure histogram, because
that is `core/eval.py`'s job and "a demo script drifting into reporting numbers is how a
results matrix stops being trustworthy". ROADMAP rule 5's `results.json` half lands with
`core/eval.py`, not here.

### D3. The video on disk predates the `frontview` switch

Commit `8dde73c` changed `render.camera` from `agentview` to `frontview`, but the file
in `experiments/` is byte-identical to the pre-commit one (397439 bytes, same frame
hashes) — it was regenerated *before* the yaml edit. Just re-run:

```bash
MUJOCO_GL=glfw uv run python -m dream_robot.sims.robosuite.tasks.pick_place_cube.demo
```

Side note worth keeping: that regeneration reproduced the original **bit for bit**,
which is a strong reproducibility signal for the whole pipeline.

### D4. Headless rendering (EGL) is broken — blocks the first remote run

Documented in `sims/robosuite/README.md`. `glfw` works only because there is a display
(`DISPLAY=:1`); it is a hidden window on a real X server, not headless rendering. This
will fail on RunPod. Likely PyOpenGL selecting Mesa over `libEGL_nvidia.so.0`, which is
present. Not urgent locally, blocking before the first remote run.

**This is a genuinely good "unglamorous infra" contribution** — the kind that earns
trust on a real project.
