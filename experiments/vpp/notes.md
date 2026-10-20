# VPP2 on T5 -- big video prior, one pass at run time

Question: if a big video prior is what fixes held-out colour (experiments/cosmos_policy), can we
keep it and drop Cosmos's per-call video denoising? VPP reads a video model's internal features in
one pass and trains a small action head on them (variant 2 in experiments/videogen/notes.md 0c).
Runs only if Cosmos Policy stage B fixes held-out colour; otherwise the next step is the
condition-swap check, and experiments/uwm stays queued for the action-free-video question.
Cheaper alternative to try first: the same fine-tuned Cosmos Policy at 1 denoising step (0.16 s
per call vs 0.6 s; RoboCasa 66.4 vs 67.1 in their paper) -- no new model, just a flag at eval.

## Step 0 · read: VPP and VPP2 (2026-10-21)
- How it works: (1) fine-tune a pretrained video model on robot + human video to predict the
  future from (frame, instruction); (2) at run time, one pass of it, take its internal features
  (not a drawn picture), compress across cameras into a few tokens, a small diffusion action head
  reads them. No video drawn.
- VPP (Hu et al., ICML 2025, `roboterax/video-prediction-policy`, MIT, last push May 2025):
  SVD backbone. `svd-robot` (~8 GB, fine-tuned on sthv2 + OpenX + xhand) is the advised start for
  a new robot. Action model released for CALVIN only (Avg.Len 4.33); no LIBERO. Their training: 8x
  A800/H100.
- VPP2 (`roboterax/video-prediction-policy-2`, MIT code, pushed 2026-10-19): Wan2.1 backbone;
  event-level video pretraining -> one-step visual planning (post-training + distillation) ->
  action expert (mixture-of-transformers). LIBERO checkpoints for both parts: Video-10k +
  Action-30k. Reports LIBERO-OOD 63.93% (2,877 / 4,500) -- the nearest published number to our
  held-out split.
- Unknown: pulls VAE / CLIP / UMT5 from the Wan2.1-I2V-14B-480P repo; whether the video model
  itself is 14B is unconfirmed. If yes, GPU and cost exceed the current budget (~$14). Check
  before any spend.
- Pick: VPP2 over VPP (LIBERO checkpoints, stronger backbone, OOD result).

Ladder (each stage asked first):
| stage | what |
|---|---|
| 0 | read the model size and inference memory from their code / config, no spend |
| A | LIBERO checkpoints on T5 frames, look only; seconds per call |
| B | small action fine-tune on the 150 T5 demos (data/cosmos_policy/t5_224), ~10 eval episodes |
| C | full run, 60 seen + 60 held-out (SmolVLA 22 / 11, oracle IDM 59 / 59, Cosmos: pending) |
