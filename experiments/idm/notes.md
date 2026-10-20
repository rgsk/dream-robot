# IDM -- the reader half of our executor (ROADMAP "Next: our own executor")

Question (step 1, oracle IDM): if a perfect video model hands us the future frames, can a small
model read the arm motion off them? On T5, against SmolVLA's 22 / 11 / 10 of 60
(seen / held-out / paraphrase).

## Step 1 · oracle + IDM built (2026-10-20)
- `sims/robosuite/oracle.py`: `SimSnapshot` (save / rewind the whole sim) + `ExpertFuture`
  (shadow expert follows the real episode; each imagination forks it, plays k steps, keeps the
  frame, rewinds). The future comes from the *current* state, not a recorded demo.
- Rewind had to cover 3 things outside `mj_getState`'s integration state: the gripper's
  rate-limited command, robosuite's observable sample timers, and the rest of MjData
  (integration state alone left ~1e-5 rad drift per rewind). Now full `mj_copyData`.
- `scripts/check_rewind.py`, 3 seeds, k=10: imagining every 10 steps leaves the expert's episode
  bit-identical (state diff 0), and every imagined frame equals the real frame k steps later
  (pixel diff 0). The oracle is exact.
- `policies/idm/`: BC's spatial-softmax encoder on both frames per camera (shared), head sees
  now, future, future - now, + joints -> k actions, masked L1. 1.05M params. Same frames shifted
  together in augmentation. Data = the 150 T5 demos, 10 held-out episodes.
- Registry `oracle_idm`; `experiments/t5/scripts/eval_t5.py --policy oracle_idm [--execute N]`.
- Smoke: 1 epoch on 5 episodes trains (0.6 min, 4060), 2-episode eval runs. Plumbing only.
- Caveat: training frames are AV1-decoded, oracle frames are raw renders (same gap BC/ACT had).
- Next: full k=10 train (~30-60 min local), then 60 episodes x 3 splits.
- Eval videos now have a third column: the imagined future frames (top, wrist) the current plan
  aims at (`policy.aim_images`, filmed by core.eval after the call, before the step). Real vs
  imagined side by side says whether a miss is the reader's or the target's. Video/eval tests 30/30.

## Step 2 · k=10 trained + evaluated (2026-10-20)
- Train: 60 epochs, 24.8 min on the 4060 (before the frame cache). Held-out error 21 -> 7.5 mrad,
  first and 10th action alike; still falling slowly at 60.
- Eval, 60 episodes per split, seeds 1000-1059 (SmolVLA's):
| split | oracle IDM k=10 | SmolVLA 20k |
|---|---|---|
| seen pairs | **47/60** | 22/60 |
| held-out pairs | **46/60** | 11/60 |
| paraphrases | **47/60** | 10/60 |
- **0 wrong cube, 0 wrong bin** in 180. All failures are motion: no_grasp 12/11/12, dropped 1/3/1.
- Paraphrase = seen, episode for episode: the IDM never reads words, same seeds -> same scenes.
- **Every failure is slot 0** (seen 26/39, held-out 23/35; slot 1 is 21/21 and 23/25). Failure
  video: the arm hovers above the slot-0 cube (the one nearer the robot) and never descends, while
  the imagined frame keeps showing it a little lower. Guess: the descent there is almost straight
  down the top camera's line of sight, so the frames barely show it. Not checked yet.
- Frame cache: `<root>/decoded/{top,wrist}.npy` (17 s to build, matches LeRobot exactly). First
  timing run OOM'd: workers got a full copy of the memmap each. Fixed (workers open the file);
  untested.
