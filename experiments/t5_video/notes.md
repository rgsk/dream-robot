# T5 video model + IDM -- instruction-following pixel video model as the executor's imagination

Notebook: `rough/t5_video.ipynb` (self-contained; cache `cache/t5_video_nb`). Task T5: two cubes,
"put the <colour> cube in the <side> box", 150 clean demos (`instructed_pick_place_clean_n150`).
Eval: closed loop, seen pairs, seeds 1000-1059 (60), template wording. Instruction = table
embeddings (colour x side), not CLIP. All video models: x-prediction (from T1, `rough/video.ipynb`),
flow matching, 20 Euler steps, 30k steps, ~1.3M params, one training run each.

## Steps 0-12 · recap (2026-11-07..09)
- UNet (instr at bottom) then UNet2 (FiLM every block). Closed loop with our top-only IDM: 0/20, 1/20.
- Diagnosis: at the grasp the gripper hides the named cube in the top view; dreams follow the
  visible one. UNet3 (wrist frame as input) written, never run (superseded by step 15).

## Step 13-14 · oracle check of the IDM (2026-11-09)
- Bug: `oracle.py` `_replan_if_lost` threshold 0.5 -> 0.85 (fingers on a cube read ~0.72, every
  grasp counted as lost). Committed alone (72a976f).
- Oracle futures, 60 seeds, 10-step / whole-clip: our top IDM 24/60 / 37/60, repo IDM (top+wrist)
  47/60 / 59/60 (= October's numbers). -> our IDM was a weak link; use the repo IDM from here.

## Step 15-16 · UNet4: dream top + wrist, repo IDM reads them
- Run on an L40S pod (notebook kernel over SSH, `runpod.md`); 30k steps in 10 min.
- UNet4 val: top 0.0055 (UNet2 0.0039), wrist 0.0084.
- Closed loop **4/60**: 43 no_grasp, 10 wrong cube. Split by named cube: **near 0/39, far 4/21**
  (near = nearer the robot, under the arm's start). Oracle repo IDM: near 26/39, far 21/21.
- Films: wrong cube = arm goes to the far cube while the named one is in view.

## Step 17 · memory, t-60 (UNet5)
- Val top 0.0053 / wrist 0.0081. Closed loop near 0/39, far 3/21: no change (raw repo IDM).
- Offline, first 60 steps of held-out demos: near-cube wrist dream error ~2x far (0.10-0.12 vs
  0.06); top not worse. Data balanced (65 near / 85 far demos).
- Mixed futures, near seeds: oracle top + dream wrist **0/39**, dream top + oracle wrist **5/39**,
  both oracle (same wrapper) 26/39. -> any dreamed camera breaks the grasp; an oracle camera
  removes wrong cube.

## Step 18 · repo IDM fine-tuned on UNet4 dreams
- Dreams (teacher-forced, from real demo frames) for all 31,885 rows; L1 fine-tune 10k steps from
  the checkpoint, shift aug. Val L1 on dreams 0.0569 -> 0.0412, on real 0.0355 -> 0.0373.
- Closed loop **14/60** (near 3/39, far 11/21). Grasps work; near wrong cube 10 -> 19.

## Step 19 · classifier-free guidance (UNet6: instruction blanked 10%)
- Val top 0.0057 / wrist 0.0085. With step 18's IDM: w=1 23/60 (near 13, far 10), w=3 22/60
  (near 14, far 8). w=1 -> 3: near wrong cube 15 -> 5, no_grasp 10 -> 19.

## Step 20 · IDM fine-tuned on UNet6 w=3 dreams -- best so far
- Val L1 on dreams 0.0559 -> 0.0406. Closed loop **29/60** (near 14/39, far 15/21).
- w sweep with this IDM: w=2 27, w=3 29, w=5 26, w=8 10. Near wrong cube 14/11/3/0, near
  no_grasp 12/13/20/33.
- Strip: w barely changes the top dream; the wrist dream gets invented colours (blue blob, the
  distractor turned blue) and broken edges at w >= 5.
- Films: `scratch/best_*.mp4`.

## Step 21 · guide only the top half (wrist w=1)
- top w=3/5/8: 25/23/20 of 60; near wrong cube 15/16/14 (back to unguided). -> the cube choice the
  IDM uses lives in the WRIST dream. Colour-only guidance dropped (unseen blank combo; the damage is
  colour anyway).

## Step 22 · the past, t-10 ... t-60 (UNet8, 48 input channels)
- Val top 0.0050 / wrist 0.0077 (~10% better); IDM val on dreams 0.0479 -> 0.0367 (better than 20).
- Closed loop **28/60** (near 15/39, far 13/21): no gain, no copycat collapse.
- Per seed vs step 20 (near): same outcome on only 19/39 -> single runs can't separate setups
  2-3 apart; noise is bigger than ±4.
- Films (`scratch/cmp_*_seed1003/1006.mp4`): seed 1003 both setups take the far red cube cleanly;
  the dream **paints the red cube yellow** (named colour) -- the instruction is satisfied in pixels
  (recolour the held cube), not in behaviour. Some no_grasp are wrong-cube approaches that never
  closed.

## Findings
- x-prediction (T1) carried; the IDM must be fine-tuned on the dreams it reads (4->14, 22->29).
- Guidance trades wrong cube for no_grasp via the wrist dream; best w=3.
- Past frames: better offline, same closed loop (3rd time offline error didn't predict it).
- Wrong cube = recolouring: a small scratch model binds colour to pixels, not objects. Cosmos
  Policy (2B, pretrained) imagined the right cube with true colours, 0 wrong cube in 60.
- Next candidates: condition on WHERE (a point / heatmap from a grounding tool, e.g. `point_at`)
  instead of the colour word; measure recolouring rate; dreams from own closed-loop states for the
  IDM; 3 seeds before small claims.
- Pod: L40S AU, ~3h20m, ~$3.70.
