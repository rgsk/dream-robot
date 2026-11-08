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
- 2026-11-08 01:23 IST: seeds_t1.sh (detached, powers off at the end): z / no-z x seeds 1, 2,
  same LAM. Results -> seeds_t1_results.txt.
- **Seeds (seeds_t1_results.txt), 3 labelled, /40:**

| arm | seed 0 | seed 1 | seed 2 | total /120 |
|---|---|---|---|---|
| WAM-C with z | 32 | 16 | 13 | 61 (51%) |
| no-z control | 15 | 16 | 17 | 48 (40%) |
- **Seed 0's 32 was an outlier.** Seeds 1-2: z 16, 13 vs no-z 16, 17 -> no gain. Pooled 61 vs 48
  (z ~1.8, carried by one run). Not established; the "z doubles it" headline is withdrawn.
  no-z is very steady (15-17); z swings 13-32. Fruit port also showed no gain (step 21).
- z-use probe (scripts/wam_c_zuse.py, zuse.log), 512 val samples, chunk-1 actions with z denoised
  freely / clamped to the true LAM z / clamped to another sample's z (same noise otherwise):

| run (sim /40) | own z vs true z | arm MSE own / true / wrong | gripper MSE own / true / wrong |
|---|---|---|---|
| z s0 (32) | 0.126 | 0.149 / 0.148 / 0.149 | **0.102** / 0.101 / 0.102 |
| z s1 (16) | 0.106 | 0.142 / 0.141 / 0.143 | 0.143 / 0.144 / 0.145 |
| z s2 (13) | 0.099 | 0.178 / 0.178 / 0.178 | 0.198 / 0.198 / 0.199 |
- **Actions ignore z at run time** in every seed: even a wrong z moves them by < 0.002. z only
  shapes the shared weights during training. Seed 0's edge is a better gripper (0.10 vs 0.14-0.20),
  not z use; its z is no more accurate than the others'. -> 32/40 was a training-seed draw.
- Same probe with chunk-1 **frames** clamped (--frames, fuse.log), all 6 runs, MSE own / true / wrong:

| run (sim) | arm | gripper |
|---|---|---|
| z s0 (32) | 0.149 / 0.146 / 0.152 | 0.102 / 0.102 / 0.118 |
| z s1 (16) | 0.142 / 0.140 / 0.150 | 0.143 / 0.145 / 0.137 |
| z s2 (13) | 0.178 / 0.176 / 0.185 | 0.198 / 0.206 / 0.212 |
| no-z s0 (15) | 0.154 / 0.153 / 0.158 | 0.231 / 0.232 / 0.245 |
| no-z s1 (16) | 0.140 / 0.140 / 0.145 | 0.252 / 0.254 / 0.258 |
| no-z s2 (17) | 0.167 / 0.166 / 0.171 | 0.135 / 0.135 / 0.141 |
- Frames matter a little more than z (wrong frames +0.004-0.016) but barely; real future frames
  help ~0. Actions come from the now frame + joints; the dreamed future is a training-only job.
- Offline gripper MSE doesn't rank the sim either (no-z s2 0.135 -> 17/40; z s0 0.102 -> 32).

## Step 3: actions must read the future (scripts/decoder.py, decoders.log)
- Separate small decoder sees ONLY joints + one input (so nothing leaks via hidden states).
  Trained on the 3 labelled demos, val on 10, fed the REAL future (upper bound), MSE std units:

| decoder input (+ joints) | arm | gripper |
|---|---|---|
| bc: now frame (no future) | 0.139 | 0.210 |
| z: LAM z t->t+8, t+8->t+16 | **0.102** | **0.068** |
| f: now + real frames t+8, t+16 | 0.128 | 0.138 |
| (WAM-C own actions, zuse.log) | 0.14-0.18 | 0.10-0.20 |
- True z makes 3 demos enough: gripper 0.21 -> 0.07 (3x better), arm 0.14 -> 0.10. z is a far
  better action input than raw future latents (0.14). Next: feed the WAMs' dreamed z (and frames).
- Real -> dreamed z (dreamed_z_check.py, dreamed_z.log), 535 val rows, MSE arm / gripper:

| | real z | dreamed z -> decoder | WAM-C own actions |
|---|---|---|---|
| s0 (sim 32) | 0.102 / 0.069 | 0.109 / 0.083 | 0.142 / 0.084 |
| s1 (sim 16) | | 0.108 / 0.077 | 0.132 / 0.098 |
| s2 (sim 13) | | 0.108 / 0.074 | 0.169 / 0.169 |
| bc decoder, no future | 0.139 / 0.211 | | |
- Little is lost: dreamed z keeps most of the gain (arm +0.006, gripper +0.005-0.014). And the
  decoder evens out the seeds (s2 own 0.169/0.169 -> 0.108/0.074). Offline only; sim next.
- **Sim, actions = decoder(dreamed z, joints)** (sweep_decoder.sh): s0 **15**, s1 **11**, s2 **13**
  = 39/120 (no_grasp dominant). WAM own actions 61/120, no-z 48/120. Best offline, worst in the
  sim -- offline action MSE misranked again. LAM line parked here.
- Decoder controls (sweep_decoder2.sh), same small decoder, 3 labelled, sim /40:

| decoder input (+ joints) | s0 | s1 | s2 | total /120 |
|---|---|---|---|---|
| bc: now frame, no future | 7 | (no WAM; one run) | | ~21 (x3) |
| f: now + dreamed frames | 6 | 3 | 8 | 17 |
| z: dreamed z | 15 | 11 | 13 | **39** |
| (WAM-C own actions) | 32 | 16 | 13 | 61 |
| (no-z WAM own actions) | 15 | 16 | 17 | 48 |
- Among decoders, dreamed z wins on every seed (11-15 vs bc 7, frames 3-8): z is the useful
  future signal; dreamed frames even hurt vs no future. But the 3-demo MLP decoder itself is weak
  (bc 7 vs WAM heads 15-17) -> the bottleneck is the decoder, not z. Next if resumed: a stronger
  decoder that reads z (flow-matching head on z + now frame + joints), or z into the WAM head.
