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

## Step 4: 8 flow-matching action heads on the frozen WAM-C trunk (2026-11-08)
- wam_c_heads.py + heads.sh. Each head = small FM transformer (3 layers, D 128), trained 6000 steps
  on the 3 labelled demos, fed the WAM's DREAMED chunk-1 z / frames (4 dreams per row), gradient-cut
  (= training them inside the WAM run). Head init seed fixed at 0; "seed" = WAM trunk seed.
  j and jn don't use the WAM, so their 3 "seeds" are one identical run.

| head | joints + | s0 | s1 | s2 | total /120 |
|---|---|---|---|---|---|
| j | (blind) | 13 | 13 | 13 | 39 (one run) |
| jn | now | 9 | 9 | 9 | 27 (one run) |
| jz | z | 6 | 8 | 5 | 19 |
| jnz | now + z | 11 | 14 | 10 | 35 |
| jf | frames | 13 | 13 | 15 | 41 |
| jnf | now + frames | 12 | 15 | 13 | 40 |
| jzf | z + frames | **22** | **17** | **16** | **55** |
| jnzf | now + z + frames | 13 | 14 | 13 | 40 |
| (WAM-C own actions) | all tokens | 32 | 16 | 13 | 61 |
| (no-z WAM own) | | 15 | 16 | 17 | 48 |
- jzf is the best head on every trunk seed, and the only one above no-z WAM's 48; still below WAM-C own 61.
- The now frame hurts the separate heads (j 13 > jn 9; jzf 55 > jnzf 40). Likely memorising: 630
  rows, now frame -> action; dreams vary per sample, so z/frames-only heads see augmented inputs.
- z alone is the worst (19); z helps only paired with frames. Blind 13/40 ~ most heads: at 3 labelled
  a separate head mostly replays motion from joints.
- Offline MSE misranked again (jzf among the worst offline).
- Reliance probe (heads_reliance.py, heads_reliance.log; offline, 535 val rows, swap one input for
  another row's, same noise; mean squared action change arm/gripper). Same on all 3 trunks; s0:

| head | swap now | swap future (z+f) | swap z | swap f |
|---|---|---|---|---|
| jn | 0.158 / 0.614 | | | |
| jnz | 0.186 / 0.215 | 0.078 / 0.632 | | |
| jf | | 0.283 / 0.413 | | |
| jnf | 0.064 / 0.173 | 0.088 / 0.481 | | |
| jzf | | **0.385** / 0.658 | 0.038 / 0.356 | **0.283** / 0.330 |
| jnzf | **0.023** / 0.265 | 0.212 / 0.425 | 0.122 / 0.148 | **0.052** / 0.193 |
- With the now frame, reliance on the future halves (jzf 0.39 -> jnzf 0.21 arm; jf 0.28 -> jnf 0.09),
  but the now frame itself is barely read (jnzf 0.02, jnf 0.06): the share goes to the joints
  (replay), not to the now frame. In jnzf the arm stops reading dreamed frames (0.28 -> 0.05) and
  reads z more (0.04 -> 0.12): now frame and dreamed frames both say "where", so they compete.
- z mostly drives the gripper (0.3-0.7 change), dreamed frames drive the arm.
- Arm reliance on the future tracks the sim: jzf 0.39 (55/120), jf 0.28-0.34 (41), jnzf 0.21 (40).
- Now-frame dropout (heads_drop.sh): jnzf trained with the now frame blanked in 50% of samples
  (shown always at test). Sim /40: s0 22, s1 16, s2 12 = **50/120** (jnzf 40, jzf 55).
  Reliance (heads_reliance_drop.log): it learned to ignore the now frame completely (0.001 / 0.004),
  gripper reads the future much more (0.43-0.53 -> 0.74-0.89), but the arm still reads the future
  only 0.20-0.22 and barely the dreamed frames (0.05-0.07; jzf 0.28-0.31) -- it reads z + joints.
  So dropout recovers most of jzf's score but not jzf's arm-on-frames.

## Step 5: option b, trunk + jzf head trained together (wam_c_b.py, 2026-11-08)
- Trunk = WAM-C with its action tokens hidden; jzf head reads the trunk's clean guess of the dream
  (x + (1-tau) v, values only) + joints, same tau, all 4 chunks, labelled only. 1.56M params, 46 min.
- b-grad (action loss reaches the trunk), seed 0: **12/40** (7 + 5, no_grasp 27). Dream quality fine
  (val f 0.229, z 0.121 vs WAM-C s1 0.242 / 0.155). Seed 0 refs: jzf on frozen WAM-C 22, WAM-C own 32, no-z 15.
- Reliance probe (b_reliance.py, b_reliance.log), b-grad s0: swap future 0.021 / 0.272, z 0.010 / 0.263,
  frames 0.007 / 0.012 (jzf on frozen trunk: 0.385 / 0.658, 0.038 / 0.356, 0.283 / 0.330).
  The arm ignores the dream entirely -> it acts on joints alone, = the blind head (13/40 vs 12/40).
  Not a leak (a leak would make the swap move the actions). Likely cause: the head reads the dream at
  the SAME tau as its actions, so at low tau it gets a mush guess; it learns the dream is useless.
  b-stop would share this, so it's not the right control. Fix to try: dream finished (or high tau)
  before the head reads it, as jzf had.
- b-clean (--dream-steps 10: head reads the FINISHED dream; 10-step dream in training, grad through
  the last step; rollout dreams first, then actions), seed 0, 1 h 56 min: **15/40** (7 + 8, no_grasp 25).
  vs same-tau b-grad 12, jzf on frozen trunk 22, no-z WAM 15. Clean dream didn't close the gap to jzf.
- Reliance, b-clean s0 (b_reliance.log): future 0.007 / 0.049, z 0.007 / 0.048, frames 0.000 / 0.002.
  Ignores the dream even more than b-grad: the head runs on joints alone in both joint-training runs.
  Likely: trained from scratch together, the dreams are useless early, the head fits the 3 demos from
  joints, and never needs the dream later. jzf read finished dreams from its first step (0.39).

## Step 6: what z holds, T2-fixed (rough/lam.ipynb steps 7-8, 2026-11-08)
- T1 probes (notebook LAM, reverse): kNN arm/gripper change 0.89/0.86 (untrained 0.66/0.61); fit on 3
  demos 0.80/0.61. Position looks known (0.78) but the TRUE change alone gives 0.85: T1's path ties
  move to place, so no extra position.
- T2-fixed LAM (same recipe, clicked at 7k). kNN "which bin?", 5 folds over all 100 demos, by fifth:
  z 42/42/41/61/82%, true move 42/43/45/93/69%, position 53/52/52/81/100% (~42% = fold-vote bias).
- During the swing z = "away from / toward the center", not left/right: centroid distance fwd-A vs
  fwd-B 0.18, fwd vs back ~1.8 (reverse=True makes that the cheapest code; frame t gives the side).
- At release z knows the bin (82% > move 69%): bins look identical, but the arm is a mirrored
  silhouette on each side; pooling drops "where", not edge orientation. z = move + look of the mover.

## Step 7: warm-start fine-tune (wam_c_warm.py, 2026-11-09)
- Start from trained WAM-C s0 trunk + its jzf head; train together 8000 steps, lr 1e-4. Trunk keeps
  wam_c.py's losses; head reads the FINISHED chunk-1 dream (10 steps, grad through the last), as jzf did.
  45 min train + sweeps.
- Sim s0: head **21/40** (jzf frozen 22), trunk's own actions 29/40 (before 32). No gain.
- Reliance (swap dream, arm / gripper): 0.36/0.70 at step 0 -> 0.27/0.67 at 8000 (final, 535 rows:
  0.295/0.644). Unlike training from scratch (0.01), the head keeps reading the dream; arm reads it a bit less.
- Decision: no seeds 1, 2. Two-stage (train WAM, freeze, fit head) is the answer; the action gradient
  into the dream buys nothing here.

## Wrap-up (2026-11-09)
Question: can LAM motion notes z on 80 video-only demos make a WAM act better from 3 labelled (T1)?
- LAM: deterministic z (8 dims) + reversed pairs reads the motion (kNN arm/gripper 0.90/0.89); fragile, 2/3 seeds click.
- 3 labelled, 3 seeds, sim /120:

| arm | total |
|---|---|
| no-z WAM (co-train control) | 48 |
| WAM-C own actions (z in the trunk) | 61 (32/16/13, one outlier) |
| best separate head: jzf on frozen WAM-C (joints + dreamed z + frames) | 55 (22/17/16) |
| trunk + jzf head trained together from scratch (s0, same-tau / finished dream) | 12 / 15 of 40 |
| warm-start fine-tune of WAM-C + jzf (s0) | 21 of 40 (jzf 22) |

- Answer: a small, noisy yes. z lifts 48 -> 55-61/120, carried mostly by seed 0; not a clear win.
- Buys: reliance probe (swap an input, measure the action change) explains the sim results where offline MSE misranks;
  heads that read the dream beat heads with the now frame (the now frame + joints = replay shortcut).
- Fails: WAM-C's actions ignore z at run time; joint training lets heads ignore the dream
  (from scratch: fit 3 demos from joints, never recover); warm start keeps reading but gains nothing.
  Two-stage (train WAM, freeze, fit head) is the working recipe.
- Limit of the testbed: the 80 video demos share the labelled demos' distribution, so video can only add
  "more of the same". Next: a test where video holds what labels lack (T2-fixed hidden goal), via the
  interpretable route (pixel video model + IDM).
