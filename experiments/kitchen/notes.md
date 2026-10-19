# Kitchen -- a sequence of real tasks in RoboCasa, driven through MCP

Plan: a water-free kitchen sequence where nothing is faked (rigid pick/place, doors, drawers,
buttons, knobs, base driving): mug -> coffee machine + start; bowl -> microwave, close, start;
pot -> stove, burner on; spoon -> drawer, close. Drop any step that proves too hard. Demo video.

## Step 1 · RoboCasa in its own venv (2026-10-19)
- `envs/robocasa/setup.sh`: Python 3.11, robosuite master (1.5.2), MuJoCo 3.3.1; our code on the
  path via `.pth`. Pattern for every new sim in `envs/README.md`.
- Assets are **23 GB** (docs say ~10); venv 6.6 GB. `yes |` + pipefail -> exit 141 after a good
  download; fixed.
- `CoffeeSetupMug`, PandaOmron (wheeled, 12-dim action): build + reset 8.0 s; 60 steps with three
  512 px cameras 1.2 s (faster than real time). Instruction: "Pick the mug from the counter and
  place it under the coffee machine dispenser."

## Step 2 · the kitchen behind the MCP server, in our viewer (2026-10-19)
- `dream_robot/sims/robocasa/kitchen.py` (adapter) + `registry.task_factory`: `robocasa/<Task>`
  names any RoboCasa task. Server runs in the 3.11 venv; shared code needed **no changes** on 3.11.
- Ready in 4-5 s; seeds 1, 2 -> different kitchens (layout + style from the env's rng).
- Our viewer showed half grey / half brown: it aimed at the origin (fine for T5's table, inside a
  wall in a kitchen). Env now offers `view()`; the kitchen aims at its robot. Looked right after.
- Viewer: zoom/orbit now redraw while the sim is idle (`mjv_updateCamera`); scroll reversed.
- `scene` camera (robot0_frontview) sits behind the arm -- a better overview camera is still open.
- Next: `step()` + scripted pick / place_at / press -> mug under the coffee machine, start it.
