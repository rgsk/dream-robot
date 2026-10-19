# Agent -- the robot as MCP tools, an LLM (or Claude Code, or you) as the planner

Plan: ROADMAP rung 9, built first and badly. Modular executor (perception -> grasp -> motion)
behind MCP tools; learned policies slot in behind the same `pick` later. Drivers: Claude Code,
MCP Inspector (`npx @modelcontextprotocol/inspector@0.22.0 --config .mcp.json --server dream-robot`),
then an OpenAI planner.

## Step 1 · MCP server: reset, look, status (2026-10-19)
- `dream_robot/agent/server.py`, registered in `.mcp.json`. Commit cef0f71.
- Bug found by the Inspector: MCP runs sync tools on any worker thread; MuJoCo's GL context is
  bound to its creating thread -> `look` returned noise. Fix: all sim calls on one sim thread.
- `look`'s camera is an enum built from the env (`scene` = wide render, then the env's cameras).

## Step 2 · find_by_colour(phrase) -> 3D point (2026-10-19)
- Top camera RGB-D at 256², colour threshold on hue, back-project, centre of the top-face points.
  SAM3 comes as its own tool (`find_sam3`), alongside, never replacing: old tools stay
  callable so the two compare on the same episode. `facebook/sam3` is gated, access requested.
- `experiments/agent/scripts/find_accuracy.py`, T5 seeds 1000-1019, both cubes:
  **40/40 found, xy error median 3.7 mm, max 5.0 mm; z median +0.7 mm.**
- All the xy error is a steady +3-4 mm in x (towards the camera). Not chased: the expert's reach
  tolerance is 15 mm.

## Step 3 · two blue cubes: ambiguity reported, not averaged (2026-10-19)
- `debug_paint` (only on the `dream-robot-debug` server, never for planners) recolours the cubes.
- Seed 7, both blue, old find: `found: true`, 994 px (two cubes' worth), point on the lower cube.
  Predicted "between the cubes" -- wrong: the median is a pixel vote, the nearer cube is bigger.
  Lucky, not right: nothing in the output said two objects matched.
- Fix: split the mask into blobs (scipy `ndimage.label`), one candidate each, largest first.
  `found` + `point` only for exactly one; else `ambiguous: true`, `point: null`, numbered
  candidates on the image. The planner chooses.
- Same scene now: ambiguous, candidates x = +0.072 / -0.074 (the slots are at ±0.07).
  T5 accuracy unchanged: 40/40 exactly one match, xy median 3.7 mm.
- Known limit: two same-coloured objects touching in the image are one blob.
- `find_by_colour` now takes `colour` (enum: red/yellow/green/blue), not a phrase: it reads
  nothing else, so no sentence to misparse. `find` takes a mask; SAM3 will bring its own.

## Step 4 · pick(point) and place(bin): 10/10 on T5 through MCP (2026-10-19)
- `sims/robosuite/skills.py`: the expert's phase machine split in half (`PICK_PHASES`,
  `PLACE_PHASES`), same IK and waypoints. pick aims at find's top-face point minus 2.1 cm.
  Known, not sensed: robot kinematics, bin positions (fixtures). Cube pose: never read.
- Holding = commanded closed **and** gripper opening > 0.15. First version used width only and
  said "holding" at reset (an open gripper is widest) -> every pick refused. Caught on call 1.
- `scripts/scripted_planner.py` (regex instead of an LLM), seeds 1000-1009:
  **10/10 task success**, 209-247 steps (7.0-8.2 s; expert 6.9 s).
  For scale: SmolVLA and our flow matching had colour at chance on T5.
  Not a fair fight: scripted motion, flat-coloured cubes, fixed bins. It shows the tools work.
- Failure paths, real sim: place with nothing held -> no_grasp (0 steps); pick at empty table
  -> no_grasp after lift; pick while holding -> error; green into left on a yellow->right task
  -> place ok, `status.success` false. Skill result != task result; the planner reads both.
- pick takes `point: [x, y, z]`, the shape find returns: one paste in the Inspector, a
  straight pass-through for a planner. Exactly 3 numbers, checked by the schema.

## Step 5 · a live window on the sim: --viewer own | docked | sharp (2026-10-19)
- Driving by tool calls showed nothing between `look`s; the window shows the server's own sim,
  paced to real time (else a pick flashes past in <1 s). Never for evals.
- Wayland: no app may place its window; MuJoCo's viewer cannot be sized (opens at 2/3 of
  *physical* px = bigger than a 2x-scaled screen -> GNOME maximises). So three modes:
  own (ours: GLFW + mjr_render, 1/3 wide, sharp, kept across resets; GNOME places it -- default),
  docked (MuJoCo viewer on XWayland, EWMH-moved to the right third, soft at 2x),
  sharp (MuJoCo viewer, Wayland, maximised). Sharp + auto-placed needs GNOME 47 (have 46).
- Bugs found on the way: Inspector strips DISPLAY (and ignores `-e` with `--config`) -> the
  server finds the Wayland socket itself; glfw picks its X11/Wayland lib at import, so that runs
  before the sim build; C prints (GLFW, GL warnings) on stdout broke stdio -> fd-level guard;
  robosuite's collision shapes (geom group 0) drew a green arm -> hidden.
- Once: own window went grey mid-session; not reproduced (7 read-backs via `snapshot()` clean).

## Step 6 · T7 scene: assorted objects on T5's table (2026-10-19)
- `sims/robosuite/tasks/assorted_pick_place/` (registry: `robosuite/assorted_pick_place`).
  2-3 objects per episode, each a random kind: box, bar, cylinder, ball (random size, palette
  colour) or a robosuite mesh (can, bottle, milk carton, cereal box, bread). Random yaw.
  Instruction: "put the {object} in the {side} bin" ("bin", since a box is now an object).
  Names unique per scene. Success: the named object resting in the named bin.
- The MJCF is redrawn every reset (`_load_model`); placement is ours: rejection sampling with
  4 cm of clear table between footprints (open gripper 8 cm + fingers).
- Bug found: robosuite's mesh metadata is wrong. Stated bottoms put meshes 2-3 cm in the air
  (they fell 19-27 mm at spawn); the bottle's footprint is 4.0 cm, not the listed 2.5.
  Fix: compile each object alone, measure footprint and resting height from its geoms.
- `scripts/t7_scenes.py`, seeds 1000-1029: all 9 kinds drawn (5-10 each); max movement over 1 s
  of holding still **2.4 mm** (= the 2 mm spawn gap). Nothing rolls or tips.

## Step 7 · find_objects: everything on the table, from depth (2026-10-19)
- `agent/tabletop.py` + MCP tool `find_objects()`. Top-camera depth -> world points; table
  height = most common z (sensed); foreground above it; clusters on a 4 mm xy grid (so a tall
  object hiding its neighbour in the image still separates); per cluster: min-area rectangle,
  height, round/rectangular, flat/domed, colour, and kind.
- Bins are perceived, not known (user decision): a container's points fill a ring of its
  footprint. Fill on 30 seeds: objects 0.96-1.27, bins 0.32-0.36 -> threshold 0.5.
  Containers return `drop_point` + `opening`; objects return `grasp` (point, closing yaw
  across the narrow side, width, `too_wide` > 7.5 cm). `place(bin)` will retire for `place_at`.
- Given, as on a real cell: the table's extent, and a robot self-mask (segmentation render of the
  arm's own geoms). Without the mask the gripper fingers came back as 2 objects: 7/30 -> 30/30.
- `scripts/find_objects_accuracy.py`, seeds 1000-1029: **30/30 scenes counted right** (every
  object + both bins, right kind). Primitives, median/max:

  | kind | xy mm | height mm | width mm | axis deg |
  |---|---|---|---|---|
  | box (6) | 0.6/0.7 | 2.0/2.1 | 0.0/0.2 | 0.0/0.0 |
  | bar (9) | 0.6/0.7 | 2.0/2.1 | 0.0/0.1 | 0.0/0.0 |
  | cylinder (8) | 0.5/1.0 | 2.0/2.1 | 2.1/3.1 | - |
  | ball (7) | 4.8/7.0 | 1.6/1.7 | 11.1/14.2 | - |

  Height +2 mm = the spawn gap (scored before anything settles). Colour 30/30 on primitives.
  Bins: centre and size within 0.5 mm, opening 11.8-12.1 cm (true 12.0).
- Known limits: a ball's far half is hidden from the oblique camera -> width 1.1 cm short,
  centre 5 mm towards the camera (within pick's 15 mm reach tolerance). Touching things merge.
  Shape words for meshes: can round/flat, bottle round/domed, milk mostly rectangular/domed,
  cereal and bread rectangular/flat.
- Overlay: outlines, not tints (the planner must see true colours); ids in boxed labels;
  yellow bar = grasp closing direction.

## Step 8 · pick(grasp) + place_at(point); place(bin) retired (2026-10-19)
- `skills.py`: pick turns the wrist about vertical so the fingers close along the grasp's
  `yaw_deg` (short way, <= 90 deg) and lands the pads on the grasp point; place_at lowers until
  the held object's bottom (from the grasp's `above_table`) is 2 cm over the point, releases.
  Carry keeps the pick's wrist. `place(bin)` retired (special case of place_at; user rule).
- find_by_colour now also returns a `grasp` (its old cube assumption: top - 2.1 cm, wrist as
  reset), so T5 still runs: find_by_colour -> pick -> find_objects (bin) -> place_at.
- Wrist check, the 5 bar targets in seeds 1000-1029: fingers close within **2.0 deg** of the
  grasp yaw; all 5 lifted ~24 cm.
- Found: the pads cannot go below **2.0 cm** over the table (fingertips touch; descend stalls
  at z 0.820). A 3 cm box's mid-height asks for 1.5 cm -> pick raises any grasp to 2.5 cm.
- Smoke: `scripts/t7_pick_place.py` (tools only; target matched by a referee env, not an LLM)
  seeds 1000-1002: 3/3 (box, can, bread). Tests: `tests/test_arm_skills.py` 3/3 (T5, T7,
  failure paths); `tests/test_agent_server.py` 14/14.
- `.mcp.json`: new `dream-robot-t7` server (T7, live viewer).

## Step 9 · Claude drives T7 by hand through MCP (2026-10-19)
- Server `dream-robot-t7` (live viewer). Per episode: reset -> find_objects -> choose ids from the
  numbered image + fields -> pick(grasp) -> place_at(container drop_point) -> status.
- **5/7 task success.** Naming was right in all 7, including look-alikes (two yellows:
  bar 11.4 cm vs box 7.4 cm; three reds: can / cereal / box; milk vs bottle).

  | seed | target | result |
  |---|---|---|
  | 1006 | yellow bar | fail: carried at ~20 deg to the walls = 12.1 cm across a 12.0 cm opening -> rests on the rims |
  | 1009 | blue cylinder | ok |
  | 1000 | red box | ok, but the arm knocked the cereal box over on the way (motion avoids nothing) |
  | 1004 | bottle (neck grasp) | ok, stands in the bin |
  | 1013 | red ball (4.2 cm) | fail: pick raises the grasp to 2.5 cm, above its 2.1 cm equator -> slipped while lowering, wedged on the bin's inner edge; touching the bin it merged into it -> no retry possible |
  | 1025 | cereal box | ok, upright |
  | 1017 | milk carton | ok, upright |
- Open: place_at ignores fit (rotate held object to the opening, e.g. its diagonal); small
  balls below the pad floor; collision-blind approach; touching objects merge (segmenter).

## Step 10 · Audit: where driving still cheats (2026-10-19)
User caught it: a real robot has cameras only. Rule from now: the driver (me, an LLM, a person)
gets cameras + the robot's own sensors; ground truth goes to a scorer outside the tools.
- In the loop: `status.success` read from the sim; episode ends on success and `place_at`
  reports `episode_over`; scripts pick bins by "left = -y"; t7_pick_place matches the target
  with a referee env.
- In debugging: step 9's bar-on-rims and knocked-cereal causes came from sim poses / a
  before-after guess, not frames; seeds chosen by reading targets ahead.
- Not cheats (corrected after review): place_at's release height (rim and object height both
  come from find_objects, i.e. the camera) and the 2.0 cm pad floor (gripper geometry, a spec
  or one-time calibration). Robustness gap only: release assumes no slip since pick.
- Sim-perfect sensing (accepted, stated): noiseless depth, exact calibration, robot self-mask
  from the sim's segmentation render.
- Plan: drop success signals from tools, verify by looking, record clips per skill (frame grid
  + review_clip + depth-diff "something else moved"), camera-chosen bins in scripts.

## Step 11 · No sim knowledge in the loop (2026-10-19)
- Tools no longer report success: `status` = instruction, steps, steps_left, time_up; skills
  stop only when time is up (never on success); the episode no longer ends on success. The
  driver judges from images (place_at already returns one).
- Scoring: `build_scored_server()` -> (server, score); `score()` is the referee (success,
  failure_mode, the target's spawn xy for checking the naming), never a tool.
- Scripts are camera-only: bins by image order (find_objects lists left to right as the top
  camera sees them); T7's target chosen by a rule-based namer over find_objects' fields
  (colour, footprint, top, size), standing in for the LLM. Smoke: T7 3/3 named right and
  succeeded; T5 planner 2/2.
- Bug found and fixed: under pytest, a second find_objects returned nothing (NaN depth, empty
  robot mask). Each hard reset leaves the old sim's GL context to the garbage collector; its
  `__del__` frees GPU buffers by id in the *current* (live) context and detaches it. robosuite
  never re-attaches. Fix (pick_place_cube/env.py): wrap MjRenderContext.render and __del__ to
  make their own context current first. Plain scripts were unaffected (old sims freed at once),
  so past evals stand; pytest and any long-lived process with delayed GC were exposed.
- Per user: no new tests while tools churn; existing ones edited minimally (14/14, 3/3).
- Deleted `scripts/t7_pick_place.py` (rule-based namer): T7 results come only from an LLM
  driver (Claude in chat, referee log), never a hand-written matcher.

## Step 12 · Claude drives T7, camera-only, referee blind (2026-10-19)
- Fresh seeds 2000-2009 (never peeked). Tools only; my verdict written per episode
  (`scratch/agent/driver_t7.jsonl`) before opening the referee log (`--score-log`).
- **Referee: 8/10 success. Named the right object 10/10. My own verdict matched 9/10.**

  | seed | target | referee | me |
  |---|---|---|---|
  | 2000, 2005 | bread | ok | ok |
  | 2001 | yellow bar 14.8 cm | fail (on the rims) | fail |
  | 2002, 2003 | cereal box | ok | ok |
  | 2004 | can | ok | ok |
  | 2006 | blue bar 12.8 cm | fail (propped, above the rim) | **ok, medium confidence** -- wrong |
  | 2007 | yellow ball 5.5 cm | ok | ok |
  | 2008 | bottle | ok | ok |
  | 2009 | yellow box | ok | ok |
- Both failures: bars longer than the 12 cm opening, released at whatever angle pick left them.
  Same as seed 1006. Fix = place_at turns the held object to fit (e.g. the opening's diagonal).
- My one wrong verdict: a bar along the diagonal looked "inside" from the wrist camera; one
  end was propped on a wall. Images at 128 px don't show a 1-2 cm lift -> verification needs a
  better check than a glance (side view, or depth: is the object's top below the rim?).

## Step 13 · Recorded clips: judge with eyes (2026-10-19)
- Rule (user): perception tools aim the arm; judging (did it work, did anything else move,
  why it failed) is done by looking at frames. No depth-diff oracle: if a judgement is weak,
  get better frames.
- `agent/recorder.py`: pick and place_at record scene/top/wrist at 512 px, 10 fps (JPEG in
  memory for the episode; MP4s with `--clips-dir`, on in `dream-robot-t7`). Skill results add
  the clip id and return a grid of 8 timestamped scene frames (last = end state).
  New tool `review_clip(clip, camera, start, end, frames, crop)`: more frames, any window,
  a region enlarged. A recorded pick takes 1.4 s.
- First use, seed 1000 (red box target): the grid shows the cereal box tipping at 0.8-2.4 s;
  `review_clip` top camera, cropped, shows the hand's body pressing on it at 1.2 s. The step-9
  guess ("arm swept it") is now seen, not inferred: the straight approach to the red box passes
  through the cereal box.
- review_clip's `clip` is optional (default: latest) and takes a number (1, 2, ... this
  episode) or an id; `status` lists the episode's clips. A per-episode dropdown would need the
  tool schema rewritten after every skill (tools/list_changed), uneven across clients -- skipped.
  Clip prefixes: `s<seed>`, `unseeded`, `warmup` (before any reset).

## Step 14 · Debug drive: tall objects, clear the table, judged from clips (2026-10-19)
- Driver's choice, not the instruction (debugging). Seeds 2010-2015, fresh; 2013 skipped (no tall object).

  | seed | done | seen in clips |
  |---|---|---|
  | 2010 | yellow box -> L, cereal -> R | cereal untouched by the first pick; upright in bin; its pick slow (7.3 s vs ~5) |
  | 2011 | cereal -> L (reaching past a cylinder), cylinder -> R | cereal looked tipped in the grid; review_clip top crop: upright |
  | 2012 | milk carton -> R, bread -> L | milk upright in bin; bread next to it unmoved (re-find: 0.2 mm) |
  | 2014 | bottle -> R (carried past a can), can -> L | hand passed the bottle in the bin, no touch |
  | 2015 | blue cylinder 6.3 cm wide | **pick timeout**: wrist clip shows a finger landing on the cylinder's rim, pushing it into a tilt; descend stalls |
- Tall objects (cereal x2, milk, bottle): 4/4 picked, placed upright, nothing knocked over.
- New gaps: (1) wide objects -- 6.3 cm leaves ~0.9 cm per side in the 8 cm span; `too_wide`
  at 7.5 cm is too generous for that margin; (2) after a failed pick the arm stays pressed on the
  object, and there is no tool to lift away / go home -> no recovery possible.
- Fix: `home` tool (lift straight up, then the start pose). An automatic back-off reflex inside
  pick was discussed and deferred: in sim time freezes between calls, so `home` alone recovers;
  on a real arm a stalled hand keeps pressing while the LLM thinks -> add the reflex then.
- Recovery test, seed 2015 driven through MCP: cylinder pick fails (timeout, hand pressed on it)
  -> `home` (3.2 s, arm clear; the cylinder had tipped onto its side) -> re-find: now 5.7 cm
  across -> bread to the right bin -> retry the cylinder: picked and placed in the left bin.
  Table cleared. The failure itself made the retry easier: lying down, it fits the gripper better.

## Step 15 · place_at turns the held object to fit (2026-10-19)
- `place_at(point, yaw_deg=None)`: turns the held object's long side to `yaw_deg` while carrying
  (short way, <= 90 deg); pick records the long side (closing direction + 90). Containers now
  report `opening.axis_yaw_deg` and `opening.diagonal` so the driver can choose side or diagonal.
- The three bars that failed, placed along the bin diagonal (in-process, referee-scored):

  | seed | bar (find) | slack along diagonal | result |
  |---|---|---|---|
  | 1006 | 11.4 x 4.1 cm | ~0.9 cm | **success** (failed before) |
  | 2006 | 12.8 x 3.8 cm | ~0.3 cm | fail: wrist view shows it exactly on the diagonal; drops in, one end rests on a wall at the corner |
  | 2001 | 14.8 x 2.7 cm | -0.4 cm | fail: cannot lie flat in a 12 cm square at any angle |

  Slack = side - (L + W)/sqrt(2). No wrist-limit trouble (turns stay within +-90 per skill).
- Beyond this: a bar that does not fit flat needs a different placement (tilted / one end
  first, or standing), i.e. a new skill, not a parameter.
- Driven through MCP (after reconnect): seed 1006, yellow bar 11.4 cm; long side ~-20 deg, bin side
  -90 -> chose the -45 diagonal (25 deg turn); review_clip top crop: flat along the diagonal,
  inside the walls. The same bar rested on the rims in step 9.

## Step 16 · The driver's base prompt (2026-10-19)
- `agent/driver_prompt.md`, sent as the MCP server's instructions -> every driver (LLM, Claude
  Code, Inspector) gets it. Covers: camera-only rules, frame conventions, the gripper's limits
  (8 cm span, ~6 cm reliable width with evidence, 2.5 cm pad floor, small balls slip), the
  loop, editing a grasp (yaw = other side, z = thinner part), fitting into a container
  (side / diagonal / will not fit), judging by frames + review_clip, recovering with home,
  known limits (touching things merge, collision-blind motion).
- Only measured facts; each number is from steps 6-15.

## Step 17 · Primitives: move_to, move_by, gripper; hand state (2026-10-19)
- User: generality means the LLM drives with granular commands, not only fixed pick/place.
  Agreed plan: add primitives, rebuild pick/place on them later, then measure primitives-only
  vs macros on 10 seeds; retire the macros if they add nothing.
- `move_to(point, yaw_deg)` (stall reflex -> "blocked": < 2 mm in 0.5 s short of target),
  `move_by(dx, dy, dz)`, `gripper(open|close)` (waits to settle), `hand` in status and every
  result (pads xyz, closing yaw, opening m, holding). All recorded as clips.
- Smoke, seed 2009, yellow box by primitives only: 8 calls, 226 steps, referee success
  (macros: ~250 steps). move_to into the table: stops "blocked" at z 0.8095 (1 cm above).
- driver_prompt.md gained a "Primitives" section with a by-hand pick recipe.

## Step 18 · Driving with primitives only (2026-10-19, in progress)
- Seeds 2020-2024, instruction target only, no pick/place_at. Verdicts in
  `scratch/agent/driver_prim.jsonl` (referee log unread until the end).
  2020 cereal, 2021 bread, 2022 ball 6.5 cm (slid in the grip, opening 5.5 -> 4.8 cm, held),
  2023 bottle carried over a milk carton, 2024 yellow box 7 x 6.2 cm: all my verdict success.
  8 calls per episode (11 on 2024), same as the by-hand recipe.
- Bugs found on 2024, fixed: (1) move_to said "blocked" in free air while the wrist turned
  (stall check watched position only) -> now position AND rotation; (2) a yaw after a cut-short
  move was computed from the measured wrist but applied to the commanded one, compounding
  (-25.4 asked -> -9.5) -> now relative to the commanded wrist. Check: -25.4/60/-80 asked ->
  -27.9/57.4/-82.8; into the table still "blocked".
- **Result, seeds 2020-2029, primitives only (no pick/place_at): referee 10/10, my verdict
  agreed 10/10.** 83 tool calls (8 per episode, 11 once) vs ~3 with macros.
  Kinds: cereal, bread, ball x3 (5.7-6.5 cm), bottle x2 (carried over a milk carton), box
  7 x 6.2 cm, can, cylinder. Nothing knocked over in any clip.
  Not a like-for-like vs the macros' 8/10 (2000-2009): those failures were bars > 12 cm, and no
  long bar came up here. What primitives bought: grasp height chosen per object (ball middles),
  high carries over tall objects, and they surfaced 2 motion bugs.

## Step 19 · pick, place_at, home retired: primitives only (2026-10-20)
- User: move_to / move_by / gripper driven by an LLM can do anything; the macros are special
  cases -> retired (special-case rule). Tools left: reset, look, status, find_objects,
  find_by_colour, move_to, move_by, gripper, review_clip.
- Their know-how moved into driver_prompt.md as measured recipes: pick (4 calls), place in a
  container (4 calls, carry height = tallest in path + hang + 3 cm, release 2 cm over rim),
  recover (open, straight up, re-find), turning to fit (carry move_to yaw), ball grasp at the
  middle. Kept in code: the two reflexes (blocked stop, gripper settle).
- skills.py: pick/place_at/home/_run and the expert phase import removed.
- Tests rewritten to the recipes: arm 2/2 (T7 seed 2009 succeeds; close-on-nothing and
  into-the-table reported, not raised), server 14/14. T5 scripted planner on primitives: 3/3.
- find_by_colour retired too: find_objects already reports colour and measures the grasp
  (find_by_colour assumed a 4 cm cube). Its one edge, separating touching objects of different
  colours, is SAM3's job. T5 scripted planner now: find_objects -> the named colour: 3/3.
  Library colour_mask stays (find_objects names colours with it).
- Also retired (user OK'd): `debug_paint` + `--debug` + T5 env.paint; the colour-blob `find()`
  and its overlay (perception.py keeps `colour_mask` for naming colours); `find_accuracy.py`
  (measured find_by_colour; in git history). `.mcp.json` listing cut to `dream-robot` (T5) and
  `dream-robot-t7`; viewer modes own/docked/sharp stay in code. `move_by` kept (user).
- Final tool set: reset, look, status, find_objects, move_to, move_by, gripper, review_clip.
  Tests: server+perception 11/11, arm 2/2; T5 planner 2/2.
- `.mcp.json` now has one server, named by task: `t7-pick-place-objects` (was `dream-robot-t7`).
  The T5 listing is gone: T7 covers it for driving. T5 stays as a task (the learned-policy /
  VLA benchmark) and its scripted planner runs in-process, needing no listing.
- "time is up: call reset" in the Inspector: T7's 30 s clock (900 steps) is too short for driving
  by hand. New `--horizon-seconds` (server) -> `horizon_seconds` (T7 env; robosuite's own horizon
  follows). `t7-pick-place-objects` runs with 300 s (9000 steps); evals keep the task's 30 s.
- `move_by(dx, dy, dz, dyaw_deg)`: relative wrist turn (from the commanded wrist). Check from
  -86.4: +30 -> -57.1, +30 with dz -5 cm -> -29.4, dx only -> -26.8 (kept within 3 deg).
- Wrist limit: Panda joint 7 is +-166 deg, starts at +44. Five move_by(dyaw_deg=80) in a row hit
  the stop on the 4th ("blocked", then timeout). Fix: a turn may go the other way (turn -+ 180),
  same closing axis, chosen so the joint stays 25 deg off its stops. Now 8/8 consecutive +80 turns
  ok, joint stays within -120..-7.
- Yaw semantics (user, option A): `move_to(point?, yaw_deg?)` -- both optional; yaw is a world
  DIRECTION (-180..180), not an axis; the arm turns whichever way keeps joint 7 inside its stops,
  else "wrist_limit". `move_by(..., dyaw_deg)` turns exactly that way and stops at the stop
  ("wrist_limit"). The automatic 180 flip is gone. `hand.wrist_room_deg` = {ccw, cw} left.
  Turning runs in <=45 deg chunks (the IK alone takes the short way), each sized from the joint
  read live. First version predicted the joint at 1:1 with hand yaw (it is ~0.7:1) and stopped
  at j7 -104; live re-reading reaches -156 (stop -166, 10 deg margin). Checks from -86 deg:
  +80 x3 ok, 4th -> wrist_limit at 167 deg; yaw 0 -> 1.2; yaw 180 -> -178.4; empty call rejected.
