# UVA: reading before the walkthrough

UVA (Unified Video Action model, Li, Gao, Sadigh, Song, RSS 2025) in one line:

> One transformer reads the past frames, predicts a **latent for the future frames**, and two
> small diffusion heads read that latent: one draws the future video, one draws the action chunk.
> At run time only the action head runs, so it plans "in video space" without drawing pixels.

Everything below comes from their code (`envs/uva/src`, commit fe3dec8), not only the paper, so
the shapes are the real ones.

---

## 1. What to read, in order (~3 h total)

| # | what | why | time |
|---|---|---|---|
| 1 | **UVA paper**, arXiv 2503.00200: abstract, Fig. 2 (architecture), Sec. 3, Table 1 | the whole thing | 60 min |
| 2 | **MAR**: Li et al. 2024, *Autoregressive Image Generation without Vector Quantization*, arXiv 2406.11838: Sec. 3 + Fig. 2 | UVA *is* MAR with video and actions bolted on; the "diffusion loss per token" idea is the key | 45 min |
| 3 | **MAE** (He et al. 2021, arXiv 2111.06377): Fig. 1 only | MAR's encoder/decoder split is MAE's: encoder sees only visible tokens, decoder fills masked slots | 10 min |
| 4 | **MaskGIT** (Chang et al. 2022, arXiv 2202.04200): Fig. 2 + the cosine schedule | how MAR fills masked tokens over a few rounds; UVA's *video* sampling uses this, its *policy* call skips it | 15 min |
| 5 | **UWM** notes, `experiments/uwm/notes.md` steps 1-2, 8, 10 | the contrast: what failed on T5, and why patches might fix it | 15 min |
| 6 | Optional: **Diffusion Policy** (Chi et al. 2023) Sec. 3 | you already know chunked actions + a denoising head from ACT / FM; this is the family UVA's action head comes from | 20 min |

Read with the questions in section 5 open.

---

## 2. The architecture, as it is in the code

```
past frames (4, from 16 obs steps)        instruction (text)
  │ 128 px → 256 px                          │
  ▼                                          ▼
VAE encoder (KL-16, frozen)              CLIP text encoder (frozen)
  │ each frame → 16×16 = 256 tokens, 16 ch    │ 1 vector → repeated as 64 text tokens
  ▼                                          │
cond tokens [4 frames × 256, 16]             │
                                             │
future-frame slots [4 × 256]  ← MASKED (a learned "fake latent") at run time
action slots: 16 actions, each repeated 64× → [4 × 256]  ← MASKED at run time
                                             │
per position: concat( future-slot, cond, action-slot ) → linear → 768-d
  + temporal pos-emb (which frame) + spatial pos-emb (which patch)
  prepend the 64 text tokens                 │
  ▼                                          ▼
MAR-base: encoder 12 blocks + decoder 12 blocks, width 768, 12 heads
  ▼
z : one 768-d vector per (frame, patch) → [4 × 256, 768]
  ├──► video head: tiny MLP diffusion *per token* (6 × 1024) → future VAE latents → VAE decoder → pixels
  └──► action head: per frame, conv over the 16×16 map → fc → stretch 4 frames to 16 steps
                    → tiny MLP diffusion *per action* (6 × 1024, 100 steps) → 16 actions (10-D)
```

Shapes to keep in your head:

- **one frame = 256 tokens** (16×16 grid of VAE latents). UWM used **1 vector** per image.
- **4 frames × 256 = 1024 positions**, plus 64 text tokens. Everything is fused **per position**
  (concatenated channels), not as separate token types. Actions don't get their own tokens; each
  action is copied over 64 positions of "its" frame.
- **policy call** (`sample_tokens`, task_mode `policy_model`): future frames and future actions all
  masked → **one** encoder+decoder pass → action head → return. No MaskGIT rounds, no pixels.
  That's where the ~95 ms per 16 actions comes from.
- actions: LIBERO uses 10-D (absolute position + 6-D rotation + gripper), chunk of 16, **8
  executed**: the ACT chunk-and-replan idea.

### The training trick: masking chooses the "mode"

One set of weights, trained with different things hidden:

| mode | given | predicted |
|---|---|---|
| policy | past frames, text | future actions (video masked) |
| video model | past frames | future frames |
| forward dynamics | past frames + actions | future frames |
| inverse dynamics | past + future frames | actions |
| full dynamics | past frames | frames and actions together |

Video is a training signal even in policy mode: the decoder has to make `z` good enough for the
video head, so `z` must carry *what happens next* in the scene. The action head then reads that
richer `z`. That is the whole bet.

---

## 3. What you already know (from your notebooks)

| UVA piece | where you've seen it |
|---|---|
| action chunk of 16, execute 8 | ACT: chunk + replan (`act.ipynb`) |
| diffusion head that draws actions from noise given a condition | `flow_matching.ipynb` (FM is the ODE cousin of their DDPM) |
| transformer over image patches + text | ACT's transformer; SmolVLA |
| frozen image encoder → tokens | ResNet / spatial-softmax features in BC; here a VAE |
| masked slots filled by a learned vector | ACT's queries; UWM's pure-noise slots |

New for you: **(a)** a VAE latent grid as the "pixels", **(b)** MAE/MAR masked prediction,
**(c)** a diffusion loss *per token* instead of one over the whole output.

---

## 4. The simplest form we'll rebuild

Drop MAR's pretrained weights, the VAE, MaskGIT rounds and CLIP; keep the idea:

```
past frame → small CNN → k×k grid of tokens (patches, not one vector)
future-frame tokens: masked slot           action chunk: masked slot
          └────────── transformer (few layers) ──────────┘
                              z
             ├─ video head:  predict next frame's grid (train only)
             └─ action head: FM over the 16-action chunk  (train + run)
```

Run time = one transformer pass + the action head. Steps 1 and 2 are the parts you've already
built (ACT data, chunks, FM head); the new parts arrive one per step: the patch grid, the masked
future slot, the video loss, the masking modes.

---

## 5. Questions to hold while reading (answer them in your head, we'll check in the notebook)

1. At run time the future frames are fully masked and never drawn. So **how can training on video
   still help the actions?** What has to be true of `z`?
2. Why does UVA give each frame **256 tokens** where UWM gave one vector, and why might that be the
   difference for T5 (two cubes, colour picks which one)?
3. MAR puts a **small diffusion MLP on each token** instead of quantizing images into a codebook.
   What does that buy over predicting the latent with an MSE loss?
4. The action head **copies each action over 64 positions**. What would break if actions were just
   32 extra tokens instead? (Hint: what does "per position, concat channels" assume?)
5. Training masks actions at ratio 0.5. **Why train with actions sometimes visible**, if the policy
   never sees future actions at run time?
6. UWM got 1/20 on T5, mostly NO_GRASP while offline error was low. Is anything in UVA's design
   aimed at that failure, or is that a hookup problem either model would hit?

---

## 6. Where to look in the code, if curious

| what | file : lines |
|---|---|
| model sizes (`mar_base`) | `envs/uva/src/unified_video_action/model/autoregressive/mar_con_unified.py:1192` |
| fusion of frame / cond / action / text per position | same file, `forward_mae_encoder`, ~445-660 |
| run-time policy path (one pass, early return) | same file, `sample_tokens`, ~945-1030 |
| action head (conv → fc → stretch → diffusion) | `.../autoregressive/diffusion_action_loss.py:109` |
| the policy wrapper (normalise, VAE, text, call) | `.../policy/unified_video_action_policy.py:221` |
| LIBERO-10 config (16 obs steps, horizon 32, 8 executed) | `.../config/task/libero10.yaml`, `.../config/model/uva.yaml` |

---

## 7. The same thing, in plain words

**The problem.** A normal policy (BC, ACT) looks at the camera and outputs "move the arm like
this". It learns only from the demos' actions. That's a thin signal: a few numbers per step.

**UVA's idea.** Make the model *also* guess what the camera will see next. Predicting the
future scene is a much richer signal (thousands of numbers per frame), and to do it well the
model has to understand the scene: where the cubes are, which one the arm is heading for, what
happens when the gripper closes. If the model has learned that, picking the actions is easier.

**How, step by step:**

1. **Shrink the images.** A frozen image compressor (the VAE) turns each camera frame into a
   16×16 grid of small "patch codes". The model never touches raw pixels.
2. **Lay out a board.** Picture 4 rows (the next 4 moments in time) × 256 squares (the patches).
   On each square, stack three things:
   - what that patch looked like **in the past** (known),
   - what that patch will look like **in the future** (blank, to be filled),
   - the **actions** for that moment (blank, to be filled).
   The instruction ("pick the red cube") sits next to the board.
3. **One big transformer reads the whole board** and writes a "thought" on every square: its
   understanding of what is happening there and what comes next.
4. **Two small heads read the thoughts.**
   - The **video head** turns them into the future frames. Used only in training, as the
     teacher that forces the thoughts to be about the real future.
   - The **action head** turns them into the next 16 arm moves. Used in training *and* at run time.
5. **At run time, skip the video head.** The model still forms its thoughts about the future, but
   never spends time drawing it. One pass, then actions. That's why it's fast.

**Masking is the training trick.** During training we randomly blank out different things: the
future frames, the actions, or both. Then the model learns to fill in whatever is blank. That's
why one model can work as a policy (actions blank), a video predictor (frames blank), or
"what moves got us from this frame to that one" (actions blank, future frames given).

**Why we care for T5.** UWM squashed each image into **one** vector and then forgot which cube
was which colour. UVA keeps **256 patches per image**, so "red cube here, blue cube there" can
survive all the way to the action head. Whether that actually fixes it is what stage B/C test.

**One-sentence version:** UVA learns to imagine the next few frames so its understanding of the
scene gets sharp, then at run time it skips the imagining and just uses that understanding to act.
