# Dream-1 + our LLM

Goal: Dream-1 (Dream Labs' API policy) executes and our LLM driver plans. The sim is built around
Dream-1's own embodiment so the policy works as well as it can. RoboCasa is parked
(`experiments/kitchen/`).

## Step 1 · The sim and the env (2026-10-20)
- Dream ships its own sim, **Dream YAM Sim**, inside the `dreamlabs` wheel
  (`dream_examples/yam_mujoco`). It has two YAM arms with the station's top + wrist cameras, plus
  an apple, orange, banana, water bottle, bowl and plate. Its own notes: apple/orange into the
  bowl worked once each live; banana onto the plate is inconsistent; nothing scores success.
- `envs/dream_yam/setup.sh`: Python 3.12, dreamlabs 1.0.1 pinned, MCP, `dream_robot.pth`. Offline
  checks pass (sim steps; dry-run request = 9 frames x 3 cameras).
- The docs' route (`ik` arm `yam` on the model's own keys) is refused for dream-1-beta today: the
  server drops the poses the client adds. The sim sends the older station keys
  (`state.arm_r.joint_pos`, `image_window.top`, ...) and the server computes the poses itself.
  `scripts/trial.py` reuses that code.
- `scripts/trial.py`: one task, a cap on paid calls, physics held during each call, an mp4 of the
  three cameras, and a score from sim positions (bowl r 7 cm, plate r 6.5 cm; referee only).
  The key comes from `.env` `DREAM_LABS_API_KEY` and is never printed.
- Every reset puts the objects in the same places: apple (0.15, 0.105), orange (0, 0.13),
  banana (-0.115, 0.085), bowl (0, -0.04), plate (-0.22, -0.045).
- Price (Billing page): dream-1-beta $0.0014, dream-1-beta-flash $0.0009 per get_action call
  (docs: billed per successful call; refusals free). $10 credit ~ 7000 calls.

## Step 2 · First paid trial (2026-10-20)
- "put apple into the bowl", 15 calls (~$0.02), RTT 0.63-0.95 s, no pool warming; 37.6 s sim.
- Not in the bowl: the apple moved 5 mm. Video: still for ~11 s, then the **left arm** (the apple's
  side) went to the apple at ~13 s (apple between the fingers in the left wrist view) and hovered
  or adjusted until the end. The right arm stayed home. No lift.
- Reading: the integration is right (correct arm, object and aim) but slow; Dream's own sim gives
  120 s. Next: log grip + apple z per call, rerun with 50 calls (~2 min sim).

## Step 3 · 50 calls: apple in the bowl (2026-10-20)
- "put apple into the bowl", 50 calls (~$0.07), 121.7 s sim. **Success**: apple ends at
  (-0.019, -0.035, 0.802), in the bowl. Done by ~36 s; the left arm only, again.
- `scripts/grip_events.py` on the trace:
  - 20.3 s first close: one fingertip touched, then none (opening 0.11), so it closed on air beside
    / on top of the apple.
  - ~26 s the empty hand went toward the bowl (lift before a grasp), then came back.
  - 30.1 s both tips on the apple at opening 0.69; lifted at 30.8 s, peak +17.8 cm.
  - ~36 s released over the bowl and the apple settled at z 0.80. Later closes were empty, the
    apple stayed in the bowl.
- The commanded grip never went below ~0.49 (it is not 0/1). The hold comes from the position gap:
  command 0.5 against the apple at 0.69. It held anyway.
- All three of the user's guesses showed up once (close on top/beside, lift before grasp); the
  retry recovered. 15 calls were simply too few.

## Step 4 · The LLM on top: an MCP server over the sim (2026-10-20)
- `dream_robot/sims/dream_yam/station.py`: the sim, Dream's request code, the policy loop (capped
  calls, fresh episode per run), trace, chunks, referee. `trial.py` now uses it.
- `dream_robot/sims/dream_yam/server.py` + `driver_prompt.md`: tools reset / look / policy(command,
  seconds) / home / review_clip / status. Each run of 2.4 s is one paid call; a server-wide cap
  (`--max-calls`); the referee goes to `--score-log` after every motion, never a tool.
- Offline check (`scratch/dream1/offline_server_test.py`): every tool through the MCP layer, policy
  on a fake client replaying the 50-call run's chunks. Open-loop replay put the apple in the
  bowl again. The sim is deterministic enough that a paid run can be replayed free.
- `experiments/dream1/mcp_dream1.json`: server `dream-yam`, task apple + orange -> bowl,
  banana -> plate, cap 100 calls (~$0.14).
- `policy(..., continue_run=true)` (user asked: review mid-run): goes on with the last run, same
  episode and camera windows, the sim held in between. Refused for a different command or after
  home/reset. Offline: 15 + 15 + 30 s pieces = the same continuous replay; apple in the bowl
  again; a refused continue left the run continuable. The prompt now says: run 10-15 s, look,
  continue.
- `--viewer` on the server: Dream's MuJoCo window + their three camera tiles, paced to real time
  (user wants to watch runs live). Replay check opened it fine (one GL 0x502 warning at start).
- First claude -p run stopped by the user after 7-14 calls (apple step, run in pieces as
  prompted): no window to watch.

## Step 5 · claude -p drives Dream-1: 3/3 (2026-10-20)
- Task: apple + orange -> bowl, banana -> plate. Fresh `claude -p`, dream-yam tools only, live
  window. **Referee: apple bowl, orange bowl, banana plate. Driver's verdict agreed on all three.**
- 86 Dream-1 calls (~$0.12), 223.6 s sim, 388 s wall, 20 turns; tokens read 344k cached + 26k.
- Per step (pieces of 15 s, continue_run as prompted, home between steps):
  - apple: 12 calls (in by the first 15 s piece);
  - orange: 28 calls, four pieces, ~35 s hovering before a clean grasp;
  - banana: 4 pieces of "put banana on the plate" (28 calls) hovered without a grasp. The driver
    sent the arms home, re-worded it "pick up the banana and place it on the plate" and started
    fresh: on the plate after 18 calls. Re-wording or the fresh start: not separated.
- The planner's part was real: it judged "stuck" from the frames, changed course, and verified at
  the end with a top look after home.
- Fixed: the referee row's sim_s was overwritten by the run's (now `run_sim_s`).
- Video of the whole run: scratch/dream1/llm_run_3of3.mp4 (16 clips joined).

## Step 6 · Real time: the world does not stop (2026-10-20)
- User: the sim paused during Dream-1 calls and while the LLM thought; a real robot cannot.
- `sims/dream_yam/realtime.py`: one loop thread owns the sim at 20 steps/s of wall clock.
  Between commands the arms hold their last target (grip included) and physics runs. Dream-1
  is called as the docs' own loop does: get_action on its own thread, `client.refill_due`
  pacing (async while round trips < 0.7 s: the next chunk is asked for during the current
  one; sync above it: the arms hold through each call), the row due now by time since the
  observation, past rows skipped. Frames go to the client every 0.2 s while the episode is
  open, pauses included. Tools post work to the loop; images are stamped with episode time.
- Server on it: `policy` reports time since the driver's last motion, the time the arms held
  waiting for Dream-1, round trip, pacing mode. Prompt: "time runs on"; default piece 15 s.
- Offline (fake 0.7 s client with the real Pacer): async, 9 calls per 12 s (vs 5 paused),
  held 0.8 s per run (the first chunk), a 4 s think gap shows up, loop lag 0. A prefetched
  call can land after a run ends: paid, unused. Cap raised to 200 calls.
- Paused mode stays in station.py for trial.py (deterministic, replayable).
- **claude -p, real time: 3/3, verdict agreed.** 94 calls (~$0.13), 220 s episode, 244 s wall,
  14 turns, tokens read 183k cached (paused run: 86 calls, 223.6 s sim, 388 s wall, 20 turns,
  344k).
  - Think gaps (arms holding, world running) 3.8-6.2 s per turn; nothing slipped or rolled in
    them.
  - Round trips 0.67-0.85 s, right at the 0.7 s sync threshold: the apple step ran mostly sync
    (arms held 4-8 s per 15-20 s piece) and took 4 pieces with two missed grasps; orange and
    banana mostly async (held 0.8-2.4 s), one fresh command plus one continue each. The banana
    went on the plate at once this time.
  - Bug: the first policy opened the Dream client on the loop thread and stalled the sim 14 s.
    Fixed (opened on the tool thread); offline re-check lag 0.
- Video: scratch/dream1/clips/261020-115124-rt.mp4 (whole episode, real time).
