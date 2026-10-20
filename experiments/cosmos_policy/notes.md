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

## Step 6 · stage B config (2026-10-21)
- `scripts/t5_config.py`: experiment `cosmos_policy_t5_from_libero` = their LIBERO config with our
  data, chunk 32, demos only, start = LIBERO policy `.pt`. pod_setup.sh copies it into their
  experiment package (their loader imports every module there; no edits to their files).
- How dims flow: training reads action / proprio width from the HDF5 arrays (8-D passes as is);
  stats (min/max -> [-1, 1]) are computed on first load into `<data>/dataset_statistics.json`
  (eval needs it). Only inference has a fixed width: `cosmos_utils.ACTION_DIM`, chosen by
  sniffing argv for "libero"/"robocasa"/"aloha" -> 7. Eval must set it to 8.
- Traps found in their code:
  - their DCP checkpointer **silently ignores a `.pt` load_path** and logs "Training from
    scratch". `PtInitCheckpointer` loads the `.pt` the way their inference loader does and asserts
    no network weights are missing. Smoke run must show "loaded N tensors, missing 0".
  - policy_fsdp shards over 8 GPUs by default -> `fsdp_shard_size=1`.
  - action L1 only went to wandb -> small callback prints `[t5] iter N action_l1 ...` every 10.
- T5 data: 150 eps, 33,385 steps (203-255 per episode), gripper action 0/1, joints within
  [-2.7, 3.4].
- Batch 16 for now (LIBERO: 30/GPU, but with optimizer states sharded 8 ways; one GPU holds
  ~32 GB of weights + grads + Adam). Smoke run sizes it from peak memory. Schedule: warm-up 200,
  decay to 0.3x by 3000; checkpoint every 500 (~25 GB each with optimizer: pod disk >= 150 GB).
- Untested until the pod: config composition (child override of the callbacks list).
Next: eval loop (T5 env + get_action, ACTION_DIM 8, seeds 1000+, seen / held-out).

## Step 7 · eval loop (2026-10-21)
- Two envs can't share a process (theirs Python 3.10 + cu128 torch, ours 3.14 + robosuite), so:
  - `scripts/serve.py` (their env): loads the fine-tuned DCP checkpoint through their
    `get_model` / `get_action` (inference config `cosmos_policy_t5_from_libero__inference`,
    sigma 80 -> 4 as theirs), sets `ACTION_DIM = 8`, JPEG q95, no flip, 5 denoising steps,
    actions only (future frames not decoded). Localhost socket; messages are plain
    str/bytes/lists (numpy 2 pickles don't load under numpy 1).
  - `dream_robot/policies/cosmos_policy` (ours): registry entry `cosmos_policy`, sends top +
    wrist uint8, state, prompt; plays the first `execute` of each 32-action chunk (default 16).
  - `eval_t5.py`: `--policy cosmos_policy --checkpoint localhost:6000 --camera-size 224`; same
    seeds 1000+, seen / held-out, wrong_cube / wrong_bin; results.json also records the server's
    checkpoint and mean seconds per call.
- All 40 prompts the env can say (8 pairs x every wording, seen + held-out) are
  in the embeddings pickle: none missing.
- Checked: imports, `--help`, syntax. Not run: anything touching a model.
- Local plumbing check (free): stub server with the same protocol, answering "hold still,
  gripper open"; 1 episode, seed 1000. Request arrived as sent (prompt, 224x224 top + wrist,
  8-D state); 38 calls in 600 steps (one per 16); NO_GRASP as expected; results.json carries
  server / execute / seconds per call; video written. One fix: at `--camera-size 224` the video
  render is scaled to 2x the camera height (the panel stacks top over wrist beside it).
- Imagined frames kept in the eval (decided in another chat): stage A's 0.6 s/call already
  included decoding the future frames (parallel path, cosmos_utils.py:740), so they are nearly
  free. serve.py now calls with parallel=True and returns the imagined top + wrist (`chunk` steps
  ahead) with each chunk; the client puts them in `aim_images`, the eval video's third column.
  Question it answers per failure: wrong cube imagined, or right cube imagined and missed?

## Step 8 · stage B pod + smoke (2026-10-21)
- Community A100 80 GB listed but 4 creates failed ("no instances"); secure A100 SXM 80 GB,
  US-KS-2, $1.79/h, 200 GB disk, CUDA 13.2 host. The image route publishes no 22/tcp: add it
  with `runpodctl pod update <id> --ports 22/tcp`. A create that hung had succeeded (pod list).
- Sync: exclude `*.safetensors` / `checkpoints` (SmolVLA checkpoints, 1.7 GB, crawled at ~2 MB/s);
  data 626 MB took 6.6 min. Token must be on the pod before any gated download.
- Smoke, 20 iters, batch 16 (`scripts/stage_b_train.sh`):
  - LIBERO policy loaded: 687 tensors, **missing 0**, 28 unused.
  - action L1 (normalised) 0.52 at iter 10 -> 0.37 at iter 20; loss 1.53 -> 1.01.
  - 3.23 s/iter (5 samples/s); first iters slower (tokenizer compile, ~40 s once).
  - peak GPU memory reserved 70.7 GB at batch 16: no room for their 30.
  - checkpoint 19 GB (DCP, with optimizer), at /root/out/cosmos_policy/cosmos_policy_t5/<name>/.
  - stats written to t5_224/dataset_statistics.json (needed by serve.py).
- Our env on the pod: plain `uv sync` into /root/venv-dream-robot, 1.5 min (no network volume, so
  scripts/runpod_setup.sh doesn't apply; pod_setup.sh already put in EGL + ffmpeg). 224 px reset OK.
- Trap 2: their inference loader reads a local checkpoint as ONE torch file, with strict=False.
  `scripts/to_pt.py` converts DCP -> .pt and checks against the LIBERO .pt: smoke iter 20 has all
  574 network tensors, 0 shape mismatches, 504 changed. (The 28 `attn_op._extra_state` are
  TransformerEngine bookkeeping: the 28 "unused" at load.) Conversion ~3 min on CPU.
- `scripts/stage_b_eval.sh`: convert -> serve.py -> eval_t5 seen + held_out -> server killed.
- Run b1 (2000 iters, decay to 0.3x by 2000, save every 500) started 00:52 IST; action L1 0.52
  (10) -> 0.23 (30) -> 0.11 (140).
