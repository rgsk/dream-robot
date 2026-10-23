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

## Step 3 · The arm and aiming in the kitchen: primitives + object_at on RoboCasa (2026-10-20)
- Adapter (`sims/robocasa/kitchen.py`): arm on absolute joint targets (RoboCasa's default is OSC
  deltas; swapped in via its config loader), `step()` maps our 8 numbers to its 13 (arm 7, gripper
  1, base 3 held at 0, torso held, 1 more); `arm_qpos_idx` = qpos 4-10; RoboCasa's
  `_check_success` -> `success`, referee only. Server: move_to / move_by / gripper / sequence run
  unchanged; recorder and point_at/object_at take their cameras from the env.
- Three things stopped the arm, found in order:
  1. kp 150 (T1's): this arm sags 0.2-0.35 rad off its targets, more than the IK's 0.06 rad step
     -> kp 500: worst joint 0.09 rad (a 12 Nm wrist motor), rest < 0.02.
  2. Reset pose: almost straight up (hand z 1.55), elbow -0.15 rad vs its -0.07 limit; the IK knows
     no limits, kept asking the elbow past it, the arm stayed put (lag 0.06 rad every step), 29 cm
     short. Fix: ready pose = hand 25 cm straight down from reset (over the robot's own base),
     solved offline with joints clamped, seeded bent: hand z 1.292, gripper down, elbow -2.25.
  3. First ready pose tried ("45 cm ahead at 1.20 m") was inside the coffee machine (joint 2 at its
     80 Nm limit; the wrist camera saw it). Dropped: a pose chosen without looking is not safe.
- Cameras: observation keeps front (centre) + wrist; aiming also gets agentview left / right
  (rendered on demand, not observed: schema allows canonical names only). Front is mostly arm;
  left sees the mug clear.
- Check, seed 1, CoffeeSetupMug: object_at on the mug from the left camera -> found, centre
  (-0.086, -5.030, 1.017), 9.6 cm tall, support z 0.921 (counter) -- **kind "container"** (a mug's
  top is a ring), so no grasp. sequence above it then 4 cm over the rim: reached, 70 sim steps,
  3 mm / 9 mm; wrist camera shows the hand centred over the mug.
- Open: how to grasp a mug (rim wall, handle, or body ~8 cm with the handle).

## Step 4 · Rim grasp (handle measured, grasp next) (2026-10-20)
- User: containers need a rim grasp (bowls have no handle), then a handle grasp for mugs.
- `tabletop._rim_and_handle`, for every container: rim = top band (4 mm) fitted with a
  least-squares circle (works on a partial arc) -> centre, mid-wall radius, wall; handle = points
  > 4 mm outside the rim circle and below the rim (>= 40) -> direction, point, reach. `rim_grasp`:
  on the mid-wall, 2 cm below the rim (<= half the height), closing radially (one finger in, one
  out), on the side away from the handle, else the side the camera sees. object_at and
  find_objects both return `rim`, `rim_grasp`, `handle`.
- Mug, seed 1, from left and right cameras (agree <= 1 mm): rim centre (-0.083, -5.040), radius
  3.7 cm, wall 3.5 mm; handle at 72 deg, sticks out 3 cm; rim_grasp on the far wall.
- Pick by rim_grasp, one sequence: fingers closed to 6.6 mm -> "nothing_held". Contacts: both
  fingertips on the mug; lifted anyway -> mug up 14.2 cm. The holding test was wrong for thin walls:
  shut on air reads 1.4 mm (kitchen) / 2.0 mm (T7), the wall 6.6 mm, the threshold was 12 mm.
  `HOLDING_MIN_OPENING` 0.15 -> 0.05 (4 mm). Rerun: 5/5 steps, holding, mug up 14.2 cm.
  Limit: a wall under ~2 mm cannot be told from air by width.
- T7 bins: no false handle; the circle fit on a square bin says wall 3 cm (corners), not 1.2 cm --
  the rim model assumes round. Centres right.

## Step 5 · First full drive: claude -p on CoffeeSetupMug, rim grasp only (2026-10-20)
- Driver prompt gets an env "cell" section (`Kitchen.cell_prompt`: base still, left/right see past
  the arm, no one table height -> support_z, cups by rim_grasp); the rest is the same everywhere.
  `experiments/kitchen/mcp_kitchen.json`; `.mcp.json` also lists it as `kitchen-coffee-mug` (for the
  Inspector). headless_drive.sh now starts only the one server it drives (`SERVER=`).
- Seed 1, cancelled by the user after 20 calls (~4 min): object_at found the mug and its rim_grasp
  on call 1, but the mug was never picked. ~8 point_at calls measured the coffee machine, then 4
  sequences probed the dispenser recess with an empty hand: blocked / blocked / timeout.
- Why (reading of the calls, geometry not checked): a top-down rim grasp puts the hand right above
  the mug -- where the dispenser head overhangs. Placing under it needs the hand beside the mug:
  the **handle grasp**. Rim grasp stays for bowls / pots.
- Also: a deliberate empty close inside a sequence stops it ("nothing_held") -- the driver closed
  the gripper on purpose to probe.

## Step 6 · Side grasp attempt, then the sink task (2026-10-20)
- Dispenser, scripted (truth allowed in checks): handle pinch *across* the loop swung the mug 40 deg
  on the carry (weight torque about the pads' normal, friction only); pinch *along* the handle
  (one finger through the loop) held it at 5-6 deg. Rays at the spout: drip tray 4.5 cm above the
  counter (reaches ~6 cm out), machine head at z 1.127 -> 16.2 cm gap for a 9.6 cm mug. Lifts of
  1-3 cm hit the tray edge; at 7 cm the mug cleared it, then the hand was pushed up 2.6 cm
  (cause not confirmed). Top-down grasps leave the hand above / beside the mug, in the head's way.
- Tilt added to move_to (`tilt_deg`: about the closing direction, 0 down, 90 horizontal toward
  yaw + 90; hand reports tilt_deg). Orientation right (fingers (-0.99, 0.02, -0.11) at 90) but
  position does not converge at tilted poses: one DLS step per tick from the live pose stalls
  (11 mm and shrinking < 2 mm / 0.5 s -> "blocked"); damping 0.05 made it worse. No contact, no
  motor saturation, no joint limit. -> motion rework (solve to convergence offline, then track),
  separately. Until then tilt stays 0.
- PickPlaceCounterToSink (no overhang), seed 1, claude -p: stopped by the user after ~25 calls,
  hovering. object_at was right; the potato is 0.77 m from the base (edge of reach) and the
  descents / sink moves came back "blocked".
- Found on the way: **a seed did not name a scene.** One env reset to seed 1 put the potato at
  y -3.62 and -4.75 alternately (RoboCasa's reset carries state over). Fix: a seeded reset builds a
  fresh env with RoboCasa's own seeding; check: 1, 1, 2, 1 -> 1's scene each time; reset 7-10 s.
- Seed scan 1-12 (object / sink distance from the base centre): 0.53-0.86 m / 0.46-0.65 m; seed 1
  was the 4th-farthest object (0.77 m). Closest: 10 (dish brush, 0.53 / 0.53).
- **Seed 10, claude -p, PickPlaceCounterToSink: RoboCasa success; the driver's verdict agreed.**
  23 calls (8 point_at, 4 object_at, 5 look, 4 sequence), 174 s (tool 38 / model 122), read 422k,
  output 7.8k. Its own account: first grasp at object_at's centre closed on air (1.6 mm) -- the
  brush's thin neck sits at the 2.5 cm pad floor; re-aimed at the brush head (dome) across its axis
  -> held at 3.9 cm; one "blocked" on arriving over the basin, but it ended where intended, still
  holding; lowered to z 0.87 in the basin, opened, retreated; re-measured it on the sink floor
  (support_z 0.716). The lemon already in the sink and a snack packet were not touched.
- First full kitchen success through the same tools as T7. Costs ~3x a T7 episode (read 422k vs
  ~140k): more measuring (12 point_at/object_at), no find_objects.
