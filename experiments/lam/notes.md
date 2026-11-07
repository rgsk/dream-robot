# LAM + WAM (arm C) -- make the 80 video-only demos supervise the action side

Question: can a latent-action model name the motion between frames well enough that a WAM trained
on those names (z) for all 90 demos acts better from 3 labelled demos? Baselines at 3 labelled
(rough/dreamzero.ipynb, one seed): floor 11/40, fine-tune all 17, co-train 13, frozen 6.

Plan (C): LAM on video -> z for every frame pair (8 steps apart) -> WAM predicts frames + z (90
demos) + real actions (3 labelled), one trunk, 32 labelled + 32 video per batch.

## Step 1: LAM + probe (scripts/lam.py)
- z = 8 numbers from (frame t, frame t+8); decoder redraws frame t+8 from frame t + z.
- Check: straight-line fit z -> true motion (state[t+8] - state[t]) on val demos, R^2.
  Answer key only, never trained on. Control = untrained encoder.
- z8 (10k steps, 4.5 min): decoder ignores z (val/copy 0.182 with z = 0.182 without; KL 8 -> 0.9).
  Scripted expert -> frame t alone predicts t+8. Probe on val: arm R2 0.57 (untrained 0.27),
  gripper 0.24 (untrained 0.19) -> some arm info, gripper missed. Fit on 3 demos ~same (0.54/0.23).
- Next: random gap / reversed pairs so frame t alone can't tell the motion.
- z8_rev (--reverse): no better. KL 0.24, z still unused (val/copy 0.483 = with z wiped; worse
  draw because direction now unknown). Probe arm 0.56, gripper 0.21. Not the cheat alone ->
  likely posterior collapse: z starts as pure noise (sigma ~1), decoder learns to ignore it early.
- Next: deterministic z (no noise, no KL), squeeze by 8 dims only (LAPA-like; theirs = VQ codes).
- z8_det_rev (--deterministic --reverse): z now USED (wiped z: val/copy 5.9 vs 0.205 with z;
  reversed pairs drawn as well as forward-only z8). Probe arm 0.61, gripper 0.28 (fit on 3: same).
  Gripper moves in only ~10% of 8-step pairs (|delta| > 0.07), so its R^2 rests on few moments;
  a straight line may be too strict a test there.
- Fair re-check (scripts/lam_eval.py, eval.log): "z=0" was misleading for det runs; now average z
  and another pair's z. z8_det_rev drawing/copy: true 0.205, average 0.540, wrong 0.729 -> z used.
  Nearest-neighbour probe (k=20), val R^2 arm/gripper, fit on 90 | on 3:
    untrained   0.61/0.57 | 0.52/0.39   (control: frames alone tell the task phase)
    z8          0.86/0.60 | 0.77/0.41
    z8_rev      0.82/0.75 | 0.72/0.61
    z8_det_rev  0.90/0.89 | 0.84/0.82   <- best; gripper read well, nonlinearly
    z8_det_gap  0.71/0.72 | 0.63/0.61   (gap 1..16 alone; drawing true/avg/wrong 0.26/0.37/0.44)
    z8_det_rev_gap 0.63/0.48 | 0.51/0.28  FAILED: z ignored till step ~10k (0.527/0.547/0.565)
  -> LAM is fragile (similar setups: one clicks, one doesn't). Seed-check z8_det_rev before WAM-C.
- z8_det_rev seeds: s0 clicks (~step 7k), s1 clicks (~step 6k; nn arm/grip 0.85/0.93, on 3: 0.78/0.75),
  s2 never (z ignored; KL 2000 = z values blew up). 2/3 click.
  "Clicked" is visible without the answer key (right z draws far better than wrong z), so picking a
  clicked run is fair. Robustness fix (z into every decoder layer, bounded z) parked.
  Decision: z8_det_rev is good enough to label video; next = WAM-C.

## Step 2: WAM-C (scripts/wam_c.py)
- Notebook's WAMz + 2 z tokens per chunk (z8_det_rev, std units); VAE/latents exported from the
  notebook cache to cache/lam/vae_lat.pt. 0.94M params, 1036 tokens. From scratch, flat lr 3e-4,
  20k steps, 32 labelled (3 demos) + 32 video per step; sweep 1000-1019 x n_exec 8/16 at the end.
- Control in the same script: --no-z (z hidden + unscored) = co-train at 32+32.
- wamc_l3 (z8_det_rev z, 3 labelled + 80 video, seed 0, 42 min train): **32/40** (16 + 16;
  fails no_grasp 6, dropped 2). Notebook 3-label arms: floor 11, fine-tune all 17, co-train 13,
  frozen 6; 10 labelled 34. val a 0.89 @5k -> 1.28 @20k (overfit offline, sim still 32).
  NOT yet attributable to z: needs --no-z control (same recipe, z hidden).
- wamc_l3_noz (control: same recipe, z hidden): **15/40** (7 + 8; no_grasp 24). Lands with the
  notebook arms (11-17). So z, not the recipe, takes 3 labelled from 15 -> 32/40. One seed each;
  val a ends ~1.27 in both, offline loss can't tell them apart.
- Next: seeds 1, 2 of both arms (~45 min each, ~4.5 h queued).
