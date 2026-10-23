# UVA on T5 -- video + actions in training, actions only at run time

Question: a small video-action model whose run-time path never draws or carries video -- does it
pick the named cube on held-out pairs (Cosmos Policy 27/30, SmolVLA 11/60), at Cosmos-beating
speed? Parked behind UWM (user, 2026-10-21): UWM first, because its video-drop variant decides
whether we bet on UWM; UVA is the ready-made version of that run-time design to compare against.

## Step 0 · read from the code (2026-10-21)
- Repo `ShuangLI59/unified_video_action` (MIT), cloned to envs/uva/src (gitignored), commit
  fe3dec8. Built on MAR (masked autoregressive image generation, `mar_base`) + a KL-16 VAE.
- LIBERO-10 setup (`config/task/libero10.yaml`, `config/model/uva.yaml`):
  - **instruction wired**: CLIP text embedding (`language_emb_model: clip`), multi-task;
  - current scene as **patches**: 256 VAE tokens per frame, 4 past frames as condition (as
    Cosmos sees it, unlike UWM's one vector per image);
  - **top camera only** (agentview 128 px, resized 256); a `predict_wrist_img` option exists;
  - actions 10-D (abs position + 6-D rotation + gripper), horizon 32, 8 executed;
  - action diffusion head (MLP, 6 x 1024), 100 sampling steps; actions masked in training at
    ratio 0.5 (no released path for action-free datasets);
  - run time: one backbone pass, then the action head; video not generated (~95 ms per 16
    actions, paper).
- Checkpoints on Google Drive: PushT, PushT-M, LIBERO-10, UMI multitask; MAR + VAE pretrained
  weights via their download script. Training: "at least 4 GPUs" for video + action.

Compared with UWM (experiments/uwm/notes.md step 1):
| | UWM | UVA |
|---|---|---|
| instruction in LIBERO setup | no (we wire CLIP) | yes (CLIP) |
| current scene | 1 vector per image (AdaLN) | 256 patches per frame |
| video at run time | in the input as noise (fix: drop it) | skipped by design |
| cameras | top + wrist | top only |
| size | ~100M trainable | MAR-base backbone, bigger |
| action-free video | built in | not released |

Ladder (each stage asked first):
| stage | what |
|---|---|
| A | LIBERO-10 checkpoint loads; memory + seconds per call on the 4060 |
| B | fine-tune on the 150 T5 demos (top camera, 8-D joints), ~10 eval episodes |
| C | 30 seen + 30 held-out (Cosmos 27 / 27) |
