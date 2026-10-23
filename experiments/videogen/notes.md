# Video model -- the imagining half of our executor (ROADMAP "Next: our own executor", step 2)

Question: can a learned model, from the current top + wrist frames and the instruction, draw the
clip the oracle drew (a frame every 10 steps until the task is done), well enough that the k=10
IDM (experiments/idm/k10) still succeeds on T5? Bar: oracle 59/59 of 60 (seen / held-out),
SmolVLA 22/11. Same eval: `eval_t5.py`, 60 episodes per split.

## Step 0 · facts and choices (2026-10-20)
- Data: 150 demos, 33,385 frames, decoded/{top,wrist}.npy at 128 px. Episodes 203-255 steps,
  so a whole clip is 21-26 frames at s = 10. Transitions at s = 10 from every start step: ~32k.
- world-model-final's UNet (3.15M): 2 previous frames stacked as channels, action embedding
  added to the time embedding, reaching every ResBlock as a per-channel *shift* (no scale).
  Trained on a 64 px synthetic world only.
- Held-out pairs (yellow->left, green->right) are never in training, so the generator must
  compose colour and side, not memorise pairs.
- Colour history: the probe's FiLM fix (100%) did not carry over to the FM policy
  (experiments/t5/notes.md step 13). A pixel generator has an easier job: the cube's colour is
  in its input pixels, it only has to move the right blob.

Choices (recommended, pending OK):
| choice | pick | why |
|---|---|---|
| order | autoregressive, one frame per 10 steps, the whole clip drawn before the arm moves | ~32k training pairs vs 150 clips; reuses the UNet as is; H stays a knob |
| cameras | top + wrist, 6 channels in one UNet | the IDM reads both; the slot-0 descent is near-invisible from top |
| resolution | decided by step 1 (64 if the IDM survives 64 px futures) | 4x cheaper, but cubes are ~3-4 px at 64 |
| condition | colour + side embeddings summed into the time embedding, + FiLM scale | composes held-out pairs; paraphrases later via a text encoder |
| compute | local 4060 at 64 px; RunPod (A100/H100, ~$2-3/h) if 128 px | GPU-bound with the frame cache |

Next (step 1, no training): oracle IDM with its imagined frames downsampled to 64 px and back up
to 128, 60 seen episodes. Says whether 64 px is enough before a generator exists.

## Step 0b · direction change: fine-tune a pretrained video model (2026-10-20)
- User: the pretrained prior is what carries to a real robot, so the main line is a pretrained
  image+text->video model fine-tuned on T5; world-model-final becomes the small one we read later
  (as SmolVLA -> our FM). Zero-shot APIs (Veo, Sora, Kling) dropped: no fine-tuning offered.
- Dense frames (16-30 fps) free the IDM's k: train k = 3, 5, 10 later, pick on generated clips
  (small k reads generator jitter as motion).

Survey of fine-tunable open models (sources in chat; numbers are vendor docs, unmeasured):
| model | size | fine-tune | memory | notes |
|---|---|---|---|---|
| **Cosmos-Predict2.5-2B** | 2B | full + LoRA; robot action-cond recipe on 1 GPU | ~33 GB infer (Predict2-2B) | image/video+text->video; robot ckpts (RoboCasa, LIBERO, Feb 2026); repo archived, pointing to Cosmos 3 |
| Cosmos 3 Nano | 16B | SFT, DROID/LIBERO action recipes | tested on 8xH100 | May 2026; too big to iterate on now |
| Cosmos 3 Edge | 2B | full FT, vision SFT | 4-GPU node | world-model ckpt "later release" |
| Wan 2.2 TI2V-5B | 5B | LoRA (DiffSynth) | ~31 GB LoRA | big community, no robot recipe |
| LTX-2.x | 19B | LoRA / full | 32 GB (int8) - 80 GB | licence: fine-tune transfer may need paid licence |

- Prior art: exactly our pipeline is UniPi (2023, video + IDM) and NVIDIA's DreamGen / GR00T-Dreams
  (Cosmos-Predict2 fine-tuned on the robot, IDM labels the frames). Later variants read actions
  from the video model's features (VPP, Video2Act) or train video+action jointly (Cosmos Policy,
  DreamZero) -- that is where steps 3-4 lead.
- RunPod (Jul 2026 list): H100 80GB from ~$2/h, A100 80GB ~$1.2/h, L40S 48GB ~$0.8/h.

Pick: **Cosmos-Predict2.5-2B, Video2World**, on 1x H100. Smallest with a robot recipe and a
1-GPU path, same family as DreamGen; Cosmos 3 when its 2B world model ships.
- Setup choices for it: top + wrist tiled side by side in one frame; low res as in NVIDIA's
  robot recipe (256x320); demos subsampled 30 -> ~10 fps so a whole episode fits one generation;
  caption = the instruction.

Costed plan (each stage asked before it runs):
| stage | what | est. |
|---|---|---|
| A | pod setup + base model zero-shot on 5 T5 start frames (look only) | 1-2 h, ~$5 |
| B | data conversion (local) + LoRA/full fine-tune smoke, 50 steps | ~1 h, ~$3 |
| C | real fine-tune on 140 demos | 3-10 h, $10-40 |
| D | IDM eval, 60 seen + 60 held-out, one clip per episode | 1-3 h, ~$5 |
| total | first loop | ~$25-60 |

## Step 0c · runtime question: video at run time or not (2026-10-20)
- User doubt: on a real robot, "draw clip -> act -> redraw" would be needed so often it may not work.
- The field's answer is world-action models (WAMs): video and actions in one model, video mostly a
  training signal. Variants, most separate -> most joined:
  1. pixel clip + separate IDM: UniPi, ours; DreamGen uses it offline to make training data
  2. video-model features + action head: VPP, Video2Act
  3. actions as extra latent frames in the video sequence: Cosmos Policy
  4. joint video+action, chunk by chunk: DreamZero (14B from Wan, 7 Hz); Dream-1 (private, cites it)
  5. video and action stacks attend to each other, action side never reads the noisy video, so
     video is skipped at run time: Dyna-2 (1M h human video). Dyna-2.1 = VLM orchestrator + Dyna-2 +
     RL whole-body controller -- our "LLM plans, model executes" split, shipping.
- So the thesis holds; runtime pixel drawing is the doubtful part.

## Step 0d · Cosmos Policy, read (2026-10-20)
- Repo NVlabs/cosmos-policy (Apache-2.0), base Cosmos-Predict2-2B-Video2World, released
  checkpoints for LIBERO, RoboCasa, ALOHA (HF collection nvidia/cosmos-policy).
- Layout: [blank, proprio, wrist, 3rd-person..., action chunk, future proprio, future images,
  value] as latent frames; actions normalised to [-1, 1], flattened, tiled into a frame. Any
  action dimension fits. Text through T5 embeddings.
- Run time: 0.61 s per chunk on 1 H100 at 5 denoising steps (future + value drawn in parallel,
  can be ignored); 0.16 s at 1 step (RoboCasa 66.4 vs 67.1). Planning (best-of-N) ~5 s per chunk.
- Inference memory 6.8 GB (LIBERO), 8.9 GB (RoboCasa): fits the 4060 on paper. Repo warns: eval on
  the GPU type it was trained on, or success drops.
- Training: ALOHA (<200 demos, ours is 150) 8x H100 x 48 h, 50k steps, from the base model.
  Train to action L1 ~0.015. Fewer GPUs = gradient accumulation.
- Results: LIBERO 98.5 (pi0.5 96.9), RoboCasa 67.1 with 50 demos/task, ALOHA 93.6 (pi0.5 88.6).
- **LIBERO is robosuite + Panda with a 3rd-person + wrist camera, like T5.** Starting from the
  LIBERO checkpoint, not the base, should shorten our fine-tune a lot (unmeasured). Our actions are
  8-D (LIBERO 7-D delta EEF), so the action head relearns either way. Our frames are 128 px,
  theirs 256 (ALOHA downsized to 256x256): upsample, or re-record T5 at 256.
