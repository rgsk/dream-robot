# UWM on T5 -- video in training only, skipped at run time

Question: a world-action model that learns video + actions together but runs without imagining
video (Dyna-2's design, variant 5 in experiments/videogen/notes.md step 0c): does it fix colour on
held-out pairs like a big video prior would (experiments/cosmos_policy), and can action-free
video help it? Queued after Cosmos Policy stage B.

## Step 0 · pick: UWM over UVA (2026-10-21)
- UWM (Zhu et al., RSS 2025, `WEIRDLabUW/unified-world-model`): one transformer denoises future
  frames and actions, each with its own noise level. Pure noise in a slot = "unknown".
  Modes from the same weights: policy (video unknown -> actions), forward model (actions given ->
  frames), inverse model (frames given -> actions, cf. our oracle IDM), video only.
  Action-free video: action slot pure noise, loss on video only.
- UVA (Li et al., RSS 2025, `ShuangLI59/unified_video_action`, MIT): shared latent + separate
  video / action diffusion heads; run time calls only the action head (~95 ms / 16 actions,
  paper). Released code does not train on action-free video; LIBERO-10 checkpoint; >=4 GPUs advised.
- Cosmos Policy for contrast: 2B, web-video prior, future latents denoised with actions every call
  (0.6 s on L40S incl. pixel decode).
- Picked UWM: action-free video built in (the cheap-data path), LIBERO-90 pretrained checkpoint
  meant for fine-tuning, simpler to rebuild. Costs: speed unmeasured, last push Oct 2025, no
  licence (fine for research; rebuild our own if it joins the executor).

Ladder (each stage asked first):
| stage | what |
|---|---|
| A | LIBERO-90 checkpoint on T5 frames, look only; measure seconds per call |
| B | small fine-tune on the 150 T5 demos (data already in LIBERO HDF5: data/cosmos_policy/t5_224), ~10 eval episodes |
| C | full run, 60 seen + 60 held-out (SmolVLA 22 / 11, oracle IDM 59 / 59) |
| D | + action-free T5 video: does held-out colour improve? Also: UWM's inverse mode vs our oracle IDM |
