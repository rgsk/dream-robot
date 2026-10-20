# Latency -- what replanning costs when the world doesn't wait

Question: the normal eval pauses physics during every policy call, so planning time is invisible.
If the sim keeps running while the policy thinks, how do success and episode time change with
thinking time (latency) and how often it replans?

## Step 1 · setup (2026-10-20)
- `scripts/lag_sweep.py`: flow matching on T2-fixed (80/80 normally; plans 16, runs `execute`),
  joint noise 0.025 as its eval. Each plan lands `latency x 30` steps after it was asked for.
  - `hold`: arm stands on its last target until the plan lands (freezes = plans x latency).
  - `async`: keeps running the old plan, asks `latency` early, drops the new plan's first
    `delay` actions (already in the past). Only possible when delay + execute <= 16, i.e.
    thinking shorter than the plan: at 0.6 s (18 steps) and up async can't run at all.
  - The 20 s episode limit is the real-time budget.
- Sweep: execute {1, 4, 8, 16} x latency {0, 0.16, 0.3, 0.6, 2} s x {hold, async}, 10 seeds.
  Video per config (first seed): clock, plan count, red "THINKING (frozen)" / amber
  "THINKING (moving on old plan)" bar.
- Latencies: 0.16 / 0.6 s = Cosmos Policy at 1 / 5 denoising steps on an H100; 2 s ~ a slow VLA.
- Sweep not run yet; single filmed episodes below.

## Step 2 · single filmed episodes, hold (2026-10-20)
Seed 1000 each, videos in `videos/`. One episode per row: an illustration, not a rate.
| policy | replan | thinking | result | time | frozen |
|---|---|---|---|---|---|
| FM T2-fixed | every step | 0 | success | 7.1 s | 0% |
| FM T2-fixed | every 8 | 0 | success | 7.2 s | 0% |
| FM T2-fixed | every step | 0.16 s | success | 16.4 s | 83% |
| FM T2-fixed | every 8 | 0.16 s | success | 11.0 s | 39% |
| FM T2-fixed, **async** | every 8 | 0.16 s | success | 7.1 s | 2% (the first plan only) |
| FM T2-fixed, **async** | every 4 | 0.3 s | fail: time ran out (scored wrong_target) | 20 s | 60% (plan lasts 4 steps, thinking 9) |
| ACT T2-fixed (plans + averages every step) | every step | 0 | success | 7.2 s | 0% |
| ACT T2-fixed | every step | 0.16 s | fail: time ran out holding the cube over the table (scored wrong_target) | 20 s | 83% |
| oracle IDM T5, H = full (1 clip) | clip once, read every 10 | 0 | success | 7.1 s | 0% |
| oracle IDM T5, H = full | same | clip 3 s | success | 10.1 s | 30% (the one 3 s wait at the start) |
| oracle IDM T5, H = 10 | new clip every 10 | 0 | dropped (timeout) | 20 s | 0% |
| oracle IDM T5, H = 10 | same | clip 3 s | no_grasp (timeout) | 20 s | 90% (6 clips in 20 s) |
- `scripts/idm_lag.py` (hold only; IDM read cost 0, real ~1 ms).
- One clip per episode pays the clip cost once; a clip every 10 steps pays it 22 times and
  never finishes inside the 20 s budget.

## Step 3 · where the jitter comes from (2026-10-20)
- Training data is not it: the noisy-expert recorder keeps the expert's clean actions as labels;
  noise was only on what it executed.
- Our eval adds joint noise 0.025 rad, fresh every step (copied from the T2-fixed eval).
  `--noise 0` added; clean videos `*_clean.mp4` (no lag 7.2 s, hold 11.3 s / 38%, async 7.2 s / 2%).
- Measured arm joints, FM T2-fixed, replan every 8, 5 seeds (`scripts/jitter.py`), mean per joint:
| setting | noise | step size mrad | wobble (2nd diff) mrad |
|---|---|---|---|
| no lag | 0.025 | 4.44 | 2.14 |
| no lag | 0 | 3.59 | **0.71** |
| hold 0.16 s | 0.025 | 3.44 | 2.06 |
| hold 0.16 s | 0 | 2.38 | 0.59 |
| async 0.16 s | 0.025 | 6.18 | 3.12 |
| async 0.16 s | 0 | 4.86 | **1.53** |
- The noise triples the wobble: most visible jitter was ours. Without it, naive async still has 2x
  the no-lag wobble: the jumps where it drops the new plan's first 5 actions. That is RTC's target.

## Step 4 · RTC (2026-10-20)
- `--modes rtc` in `lag_sweep.py`: async timing, but the new plan is sampled with guidance toward
  the old one (Black et al. 2025): first `delay` actions weight 1 (executed while thinking), the
  old plan's tail fading to 0, rest free. Guided Euler (VJP through the velocity net, beta 5),
  no retraining. Derivation above `rtc_sample`.
- Noise off, 5 seeds, mean per joint (execute 8; 7 at 0.3 s so delay + execute <= 16):
| setting | step size mrad | wobble mrad |
|---|---|---|
| no lag | 3.59 | 0.71 |
| async 0.16 s | 4.86 | 1.53 |
| **rtc 0.16 s** | 3.56 | **0.76** |
| async 0.3 s | 9.59 | 3.58 |
| **rtc 0.3 s** | 3.49 | **0.85** |
- RTC brings the seam back to no-lag smoothness; naive async gets worse with longer thinking
  (more dropped actions per seam), RTC barely moves. Success not recorded in this measurement.
- Video `rtc_exec8_lat0.16_clean.mp4`: success, 7.2 s, 2% frozen (first plan only).
