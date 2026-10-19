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
- Next: `pick(point)` and `place(side)`.
