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

## Step 1 · architecture, read from the code (2026-10-21)
- Repo cloned to envs/uwm/src (gitignored), commit b7d4067. Model = `models/uwm/uwm.py` (443 lines).
- One sample, LIBERO setup: a 19-frame window -> current obs = 2 frames (t-1, t), actions = 16
  (t..t+15), next obs = 2 frames after the actions. Two views (agentview + wrist).
- Current obs -> ONE vector per view per frame (ResNet-18, ImageNet init; ViT-B/32 optional),
  concatenated (+ proprio, + CLIP text vector if enabled) -> enters every block as AdaLN
  conditioning, not as tokens. Same global-pooling shape that dropped colour for our FM.
- Tokens the transformer denoises: 16 action tokens + next-obs patches + 24 registers. Next obs is
  first compressed by the frozen SDXL VAE (224 px -> 28x28x4), then cut into (2,4,4) patches:
  49 per view -> 98 video tokens. ~138 tokens total, DiT 12 layers x 768 (~85M trainable + 11M
  ResNet; frozen VAE 84M, CLIP text 63M). Small: likely fits and runs on the 4060 (unmeasured).
- Two noise levels, one per slot (DualTimestepEncoder), DDIM 100 train / 10 inference steps.
  Training: both levels random; action-free clips get action level = max noise and their action
  loss is masked. Modes = which slot is clean / pure noise / denoised:
  policy (video at max noise, actions denoised), forward (actions clean), inverse (next obs clean),
  video-only, joint.
- Correction to step 0: run time does not *skip* video; the 98 video tokens stay in the sequence
  as pure noise, they are just never denoised or decoded. Cheap because there are so few.
- **Language is not wired for LIBERO**: model config `use_language: False`; the robomimic/LIBERO
  loader yields no tokens; the LIBERO-90 checkpoint was trained without any instruction (task
  inferred from the scene). The encoder supports a frozen CLIP text vector; a T5 run must tokenize
  the prompt in our dataset wrapper and train that head from scratch.
- Our data needs: raw RGB (their LIBERO reader takes `agentview_rgb`, ours is JPEG) into their Zarr
  buffer; 8-D actions -> action encoder/decoder reinitialised (7-D in the checkpoint).
- Checkpoint: LIBERO-90 pretrain on Google Drive (README), not HF.
- Labeller for stage D (action-free video): NVIDIA Flora (2026-10-21; Captioner 27B, Rewarder 28B,
  open weights) writes timestamped captions -> instructions for unlabelled clips. ~60 GB GPU each,
  pod-only; at our scale a Claude call per clip may be cheaper.

## Step 2 · the bet: video-drop at run time (2026-10-21)
- User: UWM before UVA (experiments/uva/notes.md); the video-drop variant decides whether we bet
  on UWM. As built, scaling UWM gives Cosmos-like cost per call (video tokens stay in as noise).
- Video-drop: in training, sometimes remove the video tokens from the sequence (attention mask)
  instead of filling them with max noise; at run time always remove them -> cost ~ 16 action
  tokens + condition, independent of video size. Keeps action-free video training, the four
  modes, and video as teacher.
- Decides the bet: on T5, UWM-drop vs UWM-noise (as built), same data and steps --
  success (seen / held-out, wrong cube) and seconds per call. Drop wins or ties on success ->
  bet on UWM; drop loses -> the video tokens carry something at run time, UVA / Cosmos 1-step.

Ladder, revised (each stage asked first):
| stage | what |
|---|---|
| A | LIBERO-90 checkpoint loads on the 4060; memory, seconds per call with video tokens vs dropped (speed only: the checkpoint never trained with drop) |
| B | T5 data in their Zarr buffer + CLIP instruction wired; fine-tune UWM-noise, ~10 eval episodes |
| B2 | same with video-drop mixed into training; eval with video dropped -- the deciding comparison |
| B3 | video-thinking dial on B2's model: k = 0 / 1 / 3 / 10 video denoising steps before the action steps, on easy (cubes apart) vs hard (cubes touching) T5 seeds; eval cost only |
| C | winner, 30 seen + 30 held-out (Cosmos 27 / 27) |
| D | + action-free T5 video (Flora / Claude captions for unlabelled clips) |

## Step 3 · idea: a video-thinking dial (2026-10-21)
- User: how much video thinking is done should depend on the task, like LLM reasoning budgets.
- UWM already supports it: training samples video and action noise levels independently, so any
  run-time schedule is in-distribution -- denoise the future k steps, then the actions. k = 1..10
  needs no retraining; k = 0 is video-drop (B2's training). Also B2: add registers (e.g. 24 -> 64)
  so dropping video removes the noise tokens but not the scratch space.
- Risk for k = 0: video tokens at max noise carry no information about the scene, but may serve as
  scratch space holding the model's one-shot guess of the future that actions read. B2 vs B
  measures exactly that.
- Who sets k: by hand (sweep) -> the LLM planner ("cubes touching, think harder") -> the model
  itself (disagreement across a few cheap action samples).
- First test (B3): hard case = touching cubes (all Cosmos no-grasps, experiments/cosmos_policy
  step 9). Hoped-for result: easy flat in k, hard rising with k.

## Step 4 · stage A: checkpoint on the 4060 (2026-10-21)
- Own env `envs/uwm/.venv` (Python 3.10, their pins: torch 2.2.2 cu121), 13 min install.
  LIBERO-90 checkpoint `libero_90.pt` (4.0 GB, Drive id 1VfdOL8SgCl2hFwQjnr-4KwqVzB-S6La7),
  6.7 min download. Drive folder also has droid_pretrained.pt / droid_cotrained.pt.
- Loads clean: step 100,000, missing 0, unexpected 0. **309M trainable** (not ~100M as guessed
  in step 1: each block's AdaLN maps the 3,584-wide condition -- 4 image vectors x 768 + time 512
  -- to 6 x 768, ~16.5M per block, ~200M of the total), 393M with frozen VAE + CLIP.
- `scripts/stage_a.py`, batch 1, fp32, 10 denoising steps, random inputs (speed only):
| mode | seconds per call |
|---|---|
| policy (video tokens as noise) | 0.094 |
| dropped (video tokens removed; never trained so, time only) | 0.065 |
| joint + decode imagined frames | 0.300 |
  Peak GPU memory 2.2 GB.
- vs Cosmos 1.7 s / call (A100): ~18x faster as built, on a laptop-class GPU. At 30 Hz 0.094 s
  = 3 steps of lag; a 16-action chunk easily covers 2x that.
- Drop saves only 31% here: at 138 tokens the per-step fixed costs (10 sequential steps, image
  encoding, the large AdaLN layers per call) dominate. The saving grows with the video side.
- Fits the 4060 with room: stage B (fine-tune on T5) can likely run here, free. Batch size to
  measure.

## Step 5 · stage B data: T5 in UWM's buffer, with the instruction (2026-10-21)
- `scripts/t5_dataset.py`: their RobomimicDataset reading data/cosmos_policy/t5_224 (same files
  as Cosmos): JPEG decoded once into their Zarr buffer (data/uwm/t5_224/buffer.zarr, 2.9 GB, 30 s),
  8-D joints + gripper, upright frames (no flip), 224 px. Instruction per episode from the file
  name, saved beside the buffer; every sample gets CLIP `input_ids` / `attention_mask` (padded to
  16; our 40 prompts are 10-14 tokens), which their process_batch already forwards.
- 150 episodes, 33,385 steps -> 30,685 windows of 19 (their sampler never pads: the last 18 steps
  of each episode only appear inside earlier windows); val 5% of episodes -> 29,082 / 1,603.
- Checked sample 1234: obs (19, 224, 224, 3) x 2 cameras, action (19, 8), prompt "put the blue
  cube in the left box", arm closing on the blue cube across the window (scratch/uwm_sample_1234.png).

## Step 6 · stage B training script (2026-10-21)
- `scripts/train_t5.py`: one-GPU loop around their model (bf16, AdamW, encoder lr 0.05x as their
  fine-tune). Loads LIBERO-90 into a model that grew:
  - instruction on: the condition every block reads widens 3,584 -> 4,352; AdaLN input layers get
    the pretrained columns + zeros for the text -> step 0 = the pretrained model exactly;
  - 8-D actions: action input / output layers fresh;
  - registers 24 -> N: pretrained prefix copied, rest fresh.
- Trap 3 avoided: their sampler clips every sample to [-1, 1] (DDIM clip_sample) and their LIBERO
  actions are unnormalised (already in range). Ours reach 3.4 rad -> scaled per dimension to
  [-1, 1] by min / max; stats in the checkpoint.
- `--video-drop p`: share of steps trained with the video tokens removed (action loss only, no VAE
  pass). Validation reports action error (data units, 10-step sampling) with tokens, and dropped.
- Only `--help` run so far.

## Decision · code ownership (2026-10-21)
- Experiment phase (B / B2 / B3): their repo untouched; our changes in experiments/uwm/scripts
  (subclass, custom loader, `forward` swapped for video-drop, VAE wrapper). Fragile but small.
- If UWM becomes the bet: **rebuild our own UWM in dream_robot/** (user's pick), their code as
  reference for values and details only. Why: no licence on their repo (all rights reserved by
  default -- fine to run privately, not to copy in); "first impl is the template"; and it drops
  their Python 3.10 / torch 2.2 / hydra stack, so no second env and no action server.

## Step 7 · run B on a pod; first eval at step 2,500 (2026-10-21)
- Pod: secure RTX 4090 ($0.89/h; 4 community 4090 creates failed). Our pinned UWM env 7.5 min,
  LIBERO-90 from Drive in 46 s; our sim env (uv sync, Python 3.14) ~25 min on this pod's network.
  Data and checkpoints now go through private HF repos (rgsk/dream-robot-data, -checkpoints):
  data upload from home 1 min vs 7 min rsync to the pod.
- Pod smoke (200 steps, batch 16): 0.39 s/step, 9.3 GB; action_loss 0.37 -> 0.05;
  val_action_err 0.095 (data units, mean per dim). Local 4060 OOMs (309M trainable + Adam ~5 GB).
- Run B (as built + instruction, 10k steps x 16): val_action_err 0.046 at step 1,000.
- Eval scripts: `scripts/serve.py` (UWM action server, `--drop`), `dream_robot/policies/uwm`
  (2 frames per camera, all 16 actions played, as their rollouts), `scripts/eval_pod.sh`.
  Trap: `X=$(test && echo y)` exits a `set -e` script silently when the test fails.
- **Step 2,500: 0/10 seen, 0/10 held-out**; wrong cube 5 + 5, rest no_grasp / 1 drop. 0.13 s per
  call. Of the 10 wrong-cube episodes, 9 had the target in slot 0: it reaches, grasps, lifts and
  carries well, but takes the cube in slot 1 whatever the colour word (seed 1000: "yellow -> right",
  takes the green). Motor skill learnt, instruction ignored -- so far. The text input started from
  zero weights; steps 5k-10k will show whether it ever gets used.

## Step 8 · colour-swap probe: the instruction does not reach the actions (2026-10-21)
- `scripts/swap_probe.py`: 24 validation windows (8 val episodes x steps 0 / 10 / 20, before the
  arm commits); predict the 16-action chunk with the true prompt and with the colour word swapped
  to each other colour, **same noise seed**; yardstick = same prompt, another seed. Data units.
| ckpt | colour swap (chunk mean / step 16) | resample (sampling spread) |
|---|---|---|
| B 2,500 | 0.0031 / 0.0035 | 0.029 / 0.033 |
| B 5,000 | 0.0024 / 0.0026 | 0.020 / 0.023 |
- Swapping the colour moves the actions ~10x less than redrawing the noise, and the effect
  *shrinks* with training (0.0031 -> 0.0024): the model is learning to ignore the text. Matches
  the sim: step 2,500 0/10 + 0/10 (slot-1 habit), step 5,000 0/10 seen (eval stopped there, user:
  same story).
- So falling loss (val_action_err 0.046 -> 0.022) was all motion, none of it grounding.

## Step 9 · run C: UWM that sees where things are (2026-10-21)
- B stopped at 6,500 (probe trend the wrong way; step 5,000 on HF). B2 as planned is moot while
  the instruction is ignored.
- `scripts/uwm_patch.py` (`--patches`): from parts in their code that no config uses -- every
  block -> AdaLNHybridAttentionBlock (self-attn + MLP + AdaLN load 1:1 from LIBERO-90, plus
  cross-attention to context); context = ResNet-18 7x7 map before pooling (49 tokens x 2 cameras
  x 2 frames = 196) + CLIP per-word states (16), projected to 768, learned positions, layer-norm.
  Cross-attention output projections start at 0.
- Local check (scratch/uwm_patch_check.py): same LIBERO-90 load (438 copied / 13 widened / 3 fresh);
  outputs identical to the model without patches at init (max diff 0.0); 382M trainable (+73M).
- Run C = B + `--patches`, 10k x 16, started 16:13 IST on the 4090: 0.41 s/step, 10.2 GB.
  Swap probe at 2,500 is the early read.
- C swap probe @2,500: colour_swap 0.0040 vs resample 0.026 (B @2,500: 0.0031 vs 0.029). Patches + words did not make the text matter yet. action_loss 0.017. Probe @5,000 armed.
- C swap probe @5,000: colour_swap 0.0062 vs resample 0.023 -- growing (0.004 -> 0.006), unlike B (0.0031 -> 0.0024), still ~4x under the wobble. val_action_err 0.029 @5k (B 0.022 @5k).
- C @5,000 sim eval (10 seen + 10 held-out, seeds 1000+): 0/10 + 0/10; failures mostly NO_GRASP / DROPPED ("other" 9 + 6), wrong cube 1 + 4. Same picture as B @5,000 (seen 0/10, other 8, wrong cube 2). Every UWM run so far: 0 successes, mostly no grasp, while offline val_action_err is ~0.03 rad -> suspect closed-loop pipeline mismatch before more training. eval/c5000/.

## 10. Run D: C + the robot's joints (2026-10-21)
- Why: every UWM eval so far: 0 successes, mostly no grasp, though offline error is ~0.02-0.03 rad. Our actions are absolute joint targets; UWM got cameras + text only (their LIBERO actions are relative EEF moves, so they never needed state). Cosmos got the joints.
- Change: `--joints` -> obs.robot_states (7 joints + gripper, min/max to [-1, 1], 2 frames) into the AdaLN condition (their low-dim slot, their order: images, joints, text); new columns start at 0 -> identical to C at step 0 (checked: 4e-8). Server reads `proprio` (2 frames); the client sends them.
- C stopped at ~7,300 (no more evals).
- D launched 17:14 IST, same settings as C (10k x 16, patches), 0.41 s/step. Probes at 2,500 / 5,000, sim eval 10+10 at 5,000, chained on the pod (`run_d.sh`).
- D val_action_err: 0.040 @1k, 0.030 @2k, 0.032 @3k (C: 0.044, 0.043, 0.033).
- D @2,500: swap probe colour_swap 0.0022 vs resample 0.022 (text ignored). Sim 0/10 seen + 1/10 held-out (first UWM success ever, slot 1, 214 steps); 16 of 20 NO_GRASP. Joints did not fix grasping -> next suspect = closed-loop hookup (replay check). eval/d2500/.
- Replay check (`scripts/replay_check.py`, D @5,000 served locally, 3 demos replayed with their recorded actions): hookup is fine. Sim frames vs recorded: mean 1.9 / 1.7 px levels (JPEG noise), max 2.5; joints identical (gap 0.0); prompt matches; replays succeed. Model chunk vs demo's next 16 actions: 0.015-0.019 from live inputs, 0.015-0.027 from recorded -> the same. So UWM is right *on* the demo path and fails once its own actions take it off it. experiments/uwm/replay/.
- D @5,000: sim 0/10 seen + 1/10 held-out. Failures shift: NO_GRASP 7 (was 16 @2,500), DROPPED 4, WRONG_TARGET 8 -> it grasps more, often the wrong cube. Swap probe 0.0021 vs resample 0.015: colour still ignored.
- D @5,000 joint mode (video + actions denoised together, `serve.py --joint`), same seeds: 0/10 + 1/10; NO_GRASP 6, DROPPED 4, WRONG_TARGET 9 -- same as policy mode (7 / 4 / 8). 0.16 s/call (policy mode ~0.07 locally). Imagining the future does not change the actions here.
- Server now returns the decoded imagined frame (16 steps ahead) in joint mode -> eval video's extra column, as Cosmos. Rerun with frames on seeds 1010+ -> eval/d5000_viz/.
- D @7,500 joint mode + imagined frames (seeds 1010+): 0/10 + 0/10, all 20 NO_GRASP (arm reaches near the cubes, never closes on one). Imagined frames: layout, arm and bins right, but cube colours wrong (red -> magenta, blue -> cyan) and the cubes smeared -- its video doesn't keep which cube is which colour. eval/d7500_viz/ (seen_frames.png).
- Decision (2026-10-21, 18:20 IST): UWM parked on T5. Best = 1/20 (D @2,500 and @5,000); colour ignored in every run (swap probe <= 0.006 vs resample 0.015-0.03); imagined video loses cube colours; actions don't recover off the demo path. Cosmos: 27/30 on the same data. Next per the decision rule = VPP2.
- D stopped at 8,600 (action_loss ~0.010). Pod deleted. Checkpoints on HF rgsk/dream-robot-checkpoints: uwm/c/model.pt, uwm/d/model.pt (last upload = step 7,500); D @5,000 also local at checkpoints/uwm/d_hf/model_5000.pt. Pod logs in pod_logs/. UWM total pod spend ~$4.
