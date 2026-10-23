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

## Step 1 · stage 0: size, memory, cost from their code (2026-10-21, no spend)
- Code read at `roboterax/video-prediction-policy-2` a103a1e (2026-10-19), cloned to envs/vpp2/src (gitignored like the other envs).
- **Video model = Wan2.1 14B** (dim 5120, 40 layers, FFN 13824): LIBERO Video-10k is 32.8 GB in bf16 (~16B params). Frozen during action training.
- Action expert ~1B (hidden 512, 40 layers), LIBERO Action-30k 1.9 GB; reads the video model's K/V (one video pass, video latents all pure noise at t=1000 -> no video drawn). 10 Euler steps for actions, chunk 32, execute 10.
- Encoders: UMT5-xxl 11.4 GB (prompts can be cached once, then unloaded), CLIP 4.8 GB, Wan VAE 0.5 GB. Inputs: head + wrist 224x224 side by side, proprio 8-D, actions 7-D delta EEF (ours 8-D absolute joints -> new action/proprio projections, like Cosmos).
- Memory: ~40 GB of bf16 weights for inference -> needs an 80 GB GPU (A100/H100); L40S 48 GB at best tight. Nothing runs on the 4060.
- Their action training: batch 32/GPU x 8 (global 256), 30k steps, gradient checkpointing -> 32/GPU fits an 80 GB card. Ours: 150 demos, start from Action-30k, ~2k steps at batch 16-32.
- Env: tested torch 2.11 + cu130 -> pod needs a CUDA 13 driver (`--min-cuda-version 13.0`), or try torch>=2.5 per requirements.txt. Data: their LeRobot v2.1 reader (ours v3.0) -> a converter, as for Cosmos.
- Cost guess (A100 80 GB ~$1.6-1.9/h): stage A (download ~50 GB + LIBERO checkpoint on T5 frames, s/call) ~1 h ~$2; stage B (~2k action steps, 14B forward per sample, est. 3-6 s/step) ~2-4 h ~$5-8 + evals. RunPod balance ~$4 -> top-up needed before stage A+B.
- vs Cosmos (2B, 27/30 + 27/30): VPP2 is 7x bigger at run time; speed per call unknown until stage A (one 14B pass + 10 small action steps).

## Step 2 · stage A prepared (2026-10-21, no spend)
- `scripts/stage_a.py`: builds their LIBERO policy as their eval does (worker_config: Video-10k + Action-30k, tokenizer 128, 10 Euler steps, shift 5); times calls with / without the one-step video; on 12 T5 samples (6 instructions x start / a third in) saves now | imagined +16 | real +16 | imagined +64 | real +64 (their 17 frames are t0, t4 ... t64). Proprio = middle of LIBERO's state range (-> 0 after their min/max).
- `scripts/pod_setup.sh`: Python 3.10 venv, torch 2.8 cu128 (their requirements allow >=2.5; tested 2.11+cu130 would need a CUDA 13 host), their requirements.txt, weights via HF (~51 GB), T5 data from rgsk/dream-robot-data. Plain SDPA attention, no flash-attn.
- Dry check on CPU (envs/vpp2/.venv-cpu): config + processor + LIBERO stats load, proprio normalises, our frames compose to (224, 448). The 14B model itself only on the pod.
- Data converter (their LeRobot v2.1) deferred to stage B: stage A reads our HDF5 directly.

## Step 3 · stage A on the pod: LIBERO checkpoints on T5 frames (2026-10-21)
- Pod: A100 SXM 80 GB secure, $1.79/h. Setup 4 min (env 1 min, 51 GB weights via HF in 3 min). One path fix (their loader resolves a relative video path from the action file's folder -> absolute paths).
- Load 244 s, 44.6 GB; peak 46.7 GB -> needs an 80 GB card (or 48 GB, very tight).
- **1.4 s per call** actions only (10 Euler steps + one 14B pass), 1.75 s with the one-step video. Cosmos stage B: 1.7 s (5 steps). No speed win as is.
- **Zero-shot colour grounding in the imagined video:** at +64 steps the imagined wrist view shows the *named* cube in hand / filling the view in 9 of 9 samples checked (00-06, 08, 09), whichever of the two cubes is nearer -- e.g. blue chosen when it is the lower cube (00), green when upper (02), yellow (05). Top view at +64 shows the named cube gone from the table (00). Never trained on T5. Images are one-step: blurry, arm a ghost. Pictures: stage_a/*.png (now | imagined +16 | real +16 | imagined +64 | real +64).
- Read: the frozen LIBERO video model already knows "which cube" for our scene -> stage B can train actions only with video frozen (their recipe), no LoRA video step needed first.

## Step 4 · stage B glue (2026-10-21)
- `scripts/to_lerobot21.py`: our HDF5 -> their LeRobot v2.1 layout (their vendored writer no longer saves images, so files are written directly): 150 episodes, 33,385 frames, 6 tasks, H.264 all-key-frame videos, 153 MB, 53 s locally. Their dataset reads it on CPU: 2 cams x 65 frames x 224, action (32, 8), proprio (65, 8), stats computed from our data.
- `scripts/train_t5.py`: their CLI pins the LIBERO paper setup (7-D, four suites, batch 256), so it calls their pieces directly. Config = their train_action.yaml with 8-D action/state, no delta dims (absolute joint targets), our data / text cache, video frozen. **Init** = Action-30k with every trained tensor kept (backbone, the video-K/V readers, proprio encoder) and only the 7-D action in/out layers fresh at 8-D. (Their init path drops `video_kv_projections` / `video_k_norms` too -- the parts stage A showed working.) Single GPU, no DeepSpeed.
- `scripts/serve.py` (Cosmos protocol, port 6002, optional one-step imagined frame 16 steps on) + registry `vpp2` (Cosmos client, execute 16 like Cosmos) + `eval_t5.py --policy vpp2`.
- Pod: smoke (text cache, init, 20 steps at batch 16) and our sim env install running.

## Step 5 · stage B run (2026-10-21, launched 19:40 IST)
- Smoke: batch 16 OOM (frozen video K/V for 16 samples + action expert > 80 GB); **batch 8 fits, 3.7 s/step**, loss 0.88 -> 0.48 in 20 steps. Init check: 1,011 Action-30k tensors kept, 3 fresh (action_encoder.weight, head.weight/bias, 7 -> 8).
- Their full resume state is 38 GB a save (frozen video + optimizer) -> patched to save only the compact 1.8 GB action checkpoint, pushed to HF rgsk/dream-robot-checkpoints vpp2/b/ as it lands.
- Our sim env on the pod: apt update hit a 503 on an unrelated PPA and skipped the EGL install -> reinstalled, sim renders.
- Run: 2,000 steps x batch 8, lr 1e-4, warm-up 100; then eval step 2,000, 10 seen + 10 held-out, execute 16, imagined frame (+16) in the videos (`scripts/eval_pod.sh`). Expected end ~22:05 IST, ~$4.50.

## Step 6 · step-500 eval, gripper probe, call profile (2026-10-21)
- Stopped at step 500 (on the hub) to test the full hookup early. **Eval 5 + 5: 0/10, all NO_GRASP.** Seen 1000 ("yellow"): the arm goes to the yellow (named) cube, descends with the fingers around it, never closes, lifts off. 1.7 s/call (10 steps + imagined frame). stage_b/b500/.
- `scripts/grip_probe.py` on recorded demos: gripper opens 1.05-1.1, closes to 0.1-0.15 on time (mid-chunk where the demo does), joint err 0.03 rad -> no convention bug; on-path correct, fails off-path (as UWM), undertrained at 500 (a quarter epoch).
- `scripts/profile_call.py` (A100, 8 recorded samples): 1.29 s = 14B pass 0.53 s + 10 action steps x 66 ms + encoders 0.05 s; prompt encoding only 30 ms. **4 action steps: 0.86 s, same accuracy** (joint 0.030 vs 0.032 rad); 2 steps 0.76 s (0.038); 1 step 0.68 s (0.052). Floor = the 14B pass (~0.53 s A100, H100 roughly half, unmeasured). Chunk 32 = 1.07 s of motion -> async playback hides ~0.9 s if ~26 of 32 are played.
- User: >1 s per call not acceptable even if it works; considered LoRA (accuracy only, not speed) or a 2B swap (speed). Decision: continue to 1,000 steps first, eval at 4 action steps.
- Launched 21:01 IST: 500 more steps from step 500 (fresh optimizer + 100 warm-up), then eval 10 + 10 at 4 steps (`serve.py --steps`). ETA ~21:35 train, ~22:00 eval.

## Step 7 · step-1,000 eval and decision: VPP2 parked (2026-10-21)
- Steps 500 -> 1,000 continued from the compact checkpoint: fresh optimizer, new 100-step warm-up and a cosine over just these 500 -> roughly half the effective learning of an uninterrupted run (~750 "full" steps in total). Lesson: save the action expert's optimizer state (a few GB), not their 38 GB full state, and resume exactly.
- **Eval at 4 action steps: 0/10 seen, 0/10 held-out.** Seen: 8 NO_GRASP, 1 wrong cube, 1 dropped (first grasps); held-out: 9 NO_GRASP, 1 dropped. 1.32 s/call (4 steps + one-step imagined frame decoded).
- Videos (stage_b/b1000/, seen_ep1_frames.png, seen_ep2_frames.png): it goes to the *named* cube, fingers either side, hovers a little high, lifts and returns, never closes; later can drift to the other cube. Imagined frames show no grasp either. Read: grounding is right; precision at the last few cm off the demo path is not (no recovery data), as UWM.
- Totals: VPP2 best 0/20 (step 1,000), Cosmos 27/30 + 27/30. Speed: 14B pass is a 0.53 s floor on an A100, 0.86 s at 4 steps.
- Remaining VPP2 options (not taken now): LoRA-tune the 14B video on T5 then retrain actions (their full recipe; ~$12-16; no speed gain); 2B video swap (speed; new K/V connectors + action expert mostly from scratch, days of work; best backbone = a T5-tuned Cosmos 2B).
- **Decision: park VPP2; next = Cosmos stage B retrain keeping the checkpoint (HF), then 1-2 step denoising for speed** -- also yields the T5-tuned 2B backbone a 2B-VPP would need. Recovery data (DAgger with the oracle expert) for the shared "off-path, no recovery" failure.
- Pod deleted 22:03 IST. Checkpoints on HF rgsk/dream-robot-checkpoints vpp2/b/step_000500.pt, vpp2/b_cont/step_000500.pt (= step 1,000), + stats. Pod logs: pod_logs/. VPP2 pod spend ~$5.80 (balance $13.88 -> $9.35 incl. idle time).
