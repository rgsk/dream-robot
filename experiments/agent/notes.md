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
- Next: the LLM planner on these tools.
