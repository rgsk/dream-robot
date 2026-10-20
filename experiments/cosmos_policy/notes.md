# Cosmos Policy on T5 -- a video model as the policy's backbone

Question: does a pretrained video model fine-tuned as a policy (Cosmos Policy: actions, future
frames and value as latent frames of Cosmos-Predict2-2B) beat SmolVLA on T5 (22 / 11 of 60
seen / held-out), and how close does it get to the oracle IDM (59 / 59)? Why this and not
pixel video + IDM: experiments/videogen/notes.md steps 0c-0d, experiments/latency/notes.md.

## Step 1 · the repo, read (2026-10-20)
- NVlabs/cosmos-policy cloned to `envs/cosmos_policy/src` (gitignored, commit 18a2acc).
  Runs in their Docker image (`SETUP.md`), uv, Python 3.10, CUDA 12.8.
- Data it reads (`cosmos_policy/datasets/libero_dataset.py`): HDF5 per task,
  `data/demo_i/obs/{agentview_rgb, eye_in_hand_rgb}` (T, H, W, 3) uint8, `actions` (T, 7),
  `robot_states` (T, 9); the instruction comes from the file name; T5 text embeddings
  precomputed into a pickle (`save_libero_t5_text_embeddings.py`). Images resized to 224.
- One sample = 9 latent frames: blank, proprio, wrist, top, action chunk, future proprio,
  future wrist, future top (both `chunk` steps ahead), value. Each image is repeated 4x to fill
  one latent frame (the Wan tokenizer packs 4 frames into 1). Actions / proprio / value are
  written into a 16x28x28 latent: any dimension fits, so our 8-D joints + gripper go in as is.
- **LIBERO's layout is T5's**: one scene camera + one wrist camera, robosuite Panda. The LIBERO
  config is the template; only data paths, dims and stats change.
- LIBERO config: chunk 16, lr 1e-4, batch 30/GPU, warm-up 1k, decay over 30k steps, from
  `Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt`. The value head trains on demos + rollouts
  (`rollout_data_dir`, 50/50); with demos only it still gets returns (success at the end).
- Inference 6.8 GB (LIBERO): the 4060 has ~6.8 GB free with the desktop up. Too tight to count
  on; evals on the pod (also the repo's advice: eval on the GPU type it trained on).

Choices (recommended, pending OK):
| choice | pick | why |
|---|---|---|
| start from | the LIBERO Cosmos Policy checkpoint | same sim, arm, camera pair; only the action meaning changes (their 7-D delta EEF, our 8-D absolute joints) |
| images | re-record the 150 T5 demos at 224 px (local) | the model's input size; our 128 px upsampled is blurrier than anything it saw |
| chunk | 32, execute 16 (RoboCasa's) | 1.07 s at 30 Hz; 16 would be 0.53 s, under 2x its own 0.6 s thinking time for real-time later |
| value head | demos only first | no rollouts yet; add failed rollouts later if planning is tried |

Next: converter T5 (LeRobot) -> their HDF5 + T5 embeddings, local; then stage A on the pod
(load the LIBERO checkpoint, one forward pass on a T5 frame), ~$5, asked first.

## Step 2 · data: 224 px re-record + converter (2026-10-20)
- Choices above accepted (user). `record.py --camera-size 224` added (T5 recorder); same seeds
  0-149, clean expert, so the same 150 episodes with bigger pictures ->
  `data/robosuite/instructed_pick_place_clean_n150_224`.
- `scripts/to_hdf5.py`: LeRobot -> one HDF5 per instruction, `data/demo_i/obs/{agentview,
  eye_in_hand}_rgb_jpeg` (JPEG q95, as their eval compresses live frames), `actions` /
  `robot_states` (T, 8), `manifest.json` back to our episode + seed. Frames upright (no flip).
- Smoke on the 128 px set: 150 demos in 6 files (the 6 seen pairs: 16-33 demos each), 315 MB;
  instruction parses back from the file name exactly; frames decode (top + wrist checked by eye).
- T5 text embeddings still to do (needs their T5 encoder; on the pod): the 6 training
  instructions + every eval wording (held-out pairs, paraphrases).

## Step 3 · 224 px data ready; stage A scripted; spend gated (2026-10-20)
- Re-record: 150/150, 33,385 frames, seeds / lengths / prompts identical to the 128 px set.
  HDF5: `data/cosmos_policy/t5_224`, 626 MB, 6 files (frame cache `decoded/` 9.4 GB).
- `prompts.json`: all 40 T5 wordings (8 pairs x template + paraphrases) for the text encoder.
- User: no $100 before seeing anything. Ladder instead, each step earns the next:
  A zero-shot look (~$3-5) -> B small fine-tune, 1 H100 1-2 h, ~10 eval episodes (~$5-10) ->
  C full run (~$80-100). No network volume: the 11B T5 text encoder is only needed once (A),
  its pickle comes home; pods download the Cosmos checkpoint to their own disk.
- `scripts/pod_setup.sh` (no Docker: RunPod PyTorch template + their apt list + uv) and
  `scripts/stage_a.py`: embeddings -> their README sample (time, memory) -> LIBERO checkpoint on
  6 T5 frames, saving now | imagined +16 | real +16 (does it imagine the named cube?).
- Blocked: RunPod balance $4.43; ~$20 top-up for A + B.

## Step 4 · stage A, first pod lost (2026-10-20)
- Pod 1: community L40S. Setup fine (5 min); T5-11B download fine (68 GB in /root/hf); then
  "CUDA unknown error": host driver 550 (CUDA 12.4), their torch is cu128. Created without
  `--min-cuda-version 12.8` (runpod.md says to). Terminated; spend so far $0.31 (incl. a
  duplicate pod created by a retry, killed after 29 s).
- pod_setup.sh now answers LIBERO's first-import question and asserts CUDA works.

## Step 5 · stage A result (2026-10-21)
- Pod 2 (secure L40S, EU-NL-1, CUDA 13.2 host) after a stuck pod (never pulled its image, killed).
  Two auth fixes on the way: accept the license of the gated base repo
  `nvidia/Cosmos-Predict2-2B-Video2World` (the LIBERO config loads it), and the user's
  fine-grained HF token needed "read public gated repos" ticked.
- T5 text embeddings for all 40 wordings -> `data/cosmos_policy/t5_224/t5_embeddings.pkl` (42 MB,
  local copy). The 68 GB encoder is never needed again.
- LIBERO checkpoint loads clean (no missing / unexpected keys). One call on the L40S: 0.6 s warm
  (2.2 s first), 16 x 7 actions, peak 6.2 GB.
- Zero-shot on 6 T5 frames (`stage_a/*.png`: now | imagined +16 | real +16, top over wrist):
  the imagined frames keep our scene (bins, both cubes, their colours, the wrist view's layout)
  but blur it, and the arm turns into a smear; no motion toward any cube. Mid-episode (holding
  the blue cube, t75) it keeps the cube in the hand, blurred. So it reads our pictures but has no
  idea of this robot's motion yet -- expected before fine-tuning (and its proprio input was a
  placeholder). Nothing learned about colour following from this.
- Spend: $0.56 so far in total (balance $13.88). Pod terminated.
