# UWM: reading before the walkthrough

UWM (Unified World Models, Zhu et al., RSS 2025) in one line:

> One transformer denoises the **next frames** and the **action chunk** together, and each gets
> **its own noise level**. Which slot is clean, which is pure noise, and which is being denoised
> decides what the model does: policy, forward model, inverse model, or video predictor.

Everything below comes from their code (`envs/uwm/src`, commit b7d4067, `models/uwm/uwm.py`) and our
runs (`experiments/uwm/notes.md`), so the shapes and numbers are real.

---

## 1. What to read, in order (~2.5 h total)

| # | what | why | time |
|---|---|---|---|
| 1 | **UWM paper**, arXiv 2504.02792: abstract, Fig. 2 (the modes), Sec. 3, the action-free video experiment | the whole thing | 60 min |
| 2 | **DiT** (Peebles & Xie 2022, arXiv 2212.09748): Sec. 3.2, Fig. 3 | UWM's transformer *is* a DiT: the condition enters every block through **AdaLN** (scale / shift of the layer norm), not as tokens | 25 min |
| 3 | **Diffusion Forcing** (Chen et al. 2024, arXiv 2407.01392): Sec. 1 + Fig. 1 | the "each part gets its own noise level" idea, and why pure noise = "unknown" | 20 min |
| 4 | **DDIM** (Song et al. 2020): just the idea that 100 training steps can be sampled in 10 | UWM trains with 100, samples with 10 | 10 min |
| 5 | `experiments/uwm/notes.md` steps 1-4, 8, 10 | what it looked like on T5, and where it broke | 20 min |
| 6 | `rough/uva_reading.md` section 2 | the contrast | 15 min |

---

## 2. The architecture, as it is in the code

```
current obs: 2 frames (t-1, t) × 2 views (top + wrist)      [+ robot state] [+ CLIP text, if wired]
  │
  ▼
ResNet-18 (ImageNet init) → ONE 768-d vector per view per frame
  │ 4 vectors × 768  ⊕  timestep embedding (512)  →  3,584-d condition
  │
  └──────────── enters EVERY block as AdaLN (scale/shift), never as tokens ─────────┐
                                                                                     │
tokens the transformer denoises (~138):                                              │
  16 action tokens          (noisy actions,  noise level t_a)                         │
  98 video tokens           (noisy next frames, noise level t_v):                     │
      next 2 frames → frozen SDXL VAE (224 px → 28×28×4) → (2,4,4) patches → 49 / view│
  24 registers              (learned scratch space)                                   │
  │                                                                                  │
  ▼                                                                                  │
DiT: 12 blocks × 768, 12 heads  ◄──────────────────────────────────────────────────┘
  │
  ├─► action slice → action decoder → predicted noise on the actions
  └─► video slice  → unpatchify     → predicted noise on the next-frame latents
```

Training (`UnifiedWorldModel.forward`): draw `t_a` and `t_v` **independently** at random, noise
each slot to its level, predict both noises, loss = action MSE + video MSE. For action-free clips,
`t_a` = max noise and the action loss is masked.

Sizes we measured (stage A): **309M trainable** (most of it the AdaLN layers: each block maps the
3,584-d condition to 6 × 768), 393M with frozen VAE + CLIP. 2.2 GB on the 4060.

### The trick: noise levels choose the "mode"

| mode | actions | next frames | in the code |
|---|---|---|---|
| policy | denoised 10 steps | pure noise, never touched | `sample_marginal_action` |
| forward model | clean (given) | denoised | `sample_forward_dynamics` |
| inverse model | denoised | clean (given) | `sample_inverse_dynamics` |
| video only | pure noise | denoised | `sample_marginal_next_obs` |
| joint | denoised | denoised, together | `sample_joint` |

"Pure noise" = "I don't know this". Because training sampled every combination of levels,
every row is something the model has seen.

---

## 3. What you already know

| UWM piece | where you've seen it |
|---|---|
| denoise an action chunk over steps | `flow_matching.ipynb` (DDPM/DDIM ≈ FM's noisy cousin) |
| chunk of 16 actions | ACT |
| image → one global vector | BC with global pooling; and our finding that pooling **dropped colour** (spatial softmax, t5-status) |
| a transformer over action tokens | ACT's decoder |

New for you: **(a)** AdaLN conditioning (condition modulates every layer instead of being a token),
**(b)** two noise levels in one sequence, **(c)** registers.

---

## 4. What happened on T5 (why it's parked)

- Stage A: 0.094 s per 16 actions on the 4060 (0.065 with video tokens dropped, 0.300 joint + decode).
- Runs B, C, D: **best 1/20**. Mostly NO_GRASP early, WRONG_TARGET later.
- **Colour ignored** in every run: swap red↔blue in the instruction and the actions barely move
  (swap probe ≈ 0.002 vs 0.015-0.03 for a resample).
- Imagined frames: layout right, **cube colours wrong** (red → magenta, blue → cyan), cubes smeared.
- Suspect: the scene enters as **one vector per image**, so "which cube is where and what colour"
  is squashed before the transformer ever sees it. This is exactly what UVA's 256 patches change.

---

## 5. Questions to hold while reading

1. Policy mode leaves 98 video tokens in the sequence as **pure noise**. They carry no information
   about the scene, so why might removing them still change the actions?
2. The scene enters through **AdaLN**, not as tokens. What can attention do with tokens that it
   can't do with a condition that only scales and shifts each layer?
3. Training draws `t_a` and `t_v` independently. Why does that make the forward and inverse modes
   come for free?
4. Action-free video: the action slot is pure noise and its loss is masked. What does the model
   still learn from such a clip, and how could that help the policy?
5. Given our T5 results, what is the smallest change to UWM that would test the "one vector per
   image loses colour" theory, without switching to UVA?

---

## 6. Where to look in the code

`envs/uwm/src` currently has **54 tracked files deleted** from the working tree (same thing happened
to `envs/uva/src`). `git show HEAD:<path>` reads them anyway, or restore with `git checkout -- .`
inside that folder.

| what | file : lines (at HEAD) |
|---|---|
| video patchifier (VAE latents → tokens) | `models/uwm/uwm.py:13` |
| two-timestep embedding | `models/uwm/uwm.py:66` |
| the DiT forward: actions + video + registers, AdaLN cond | `models/uwm/uwm.py:181` |
| training loss with two noise levels | `models/uwm/uwm.py:279` |
| the five sampling modes | `models/uwm/uwm.py:322-443` |
| our T5 scripts (train, serve, swap probe) | `experiments/uwm/scripts/` |

---

## 7. The same thing, in plain words

**The idea.** Train one model to un-blur two things at once: the robot's next 16 moves, and what
the camera will see next. Each is blurred (noised) by a different, random amount.

**How, step by step:**

1. **Look at the scene.** A ResNet turns each camera image into **one summary vector**. Those
   summaries (plus the instruction, if wired) become a "context" that nudges every layer.
2. **Lay out the slots.** A row of 16 action tokens, 98 tokens for the next frames, and 24 blank
   scratch tokens.
3. **Blur them.** In training, pick a random blur level for the actions and another for the
   frames. Fully blurred = "unknown". Not blurred = "given".
4. **Un-blur.** The transformer predicts the blur on both, guided by the context. Repeat 10 times
   at run time to go from pure noise to clean actions.

**The modes come from choosing what's blurred.** Blur only the actions → it's a policy. Give the
actions and blur the frames → it predicts what those moves will do. Give the next frames and blur
the actions → it figures out what moves got there. Blur both → it imagines a whole future.

**Videos with no actions** (e.g. people's videos) can still train it: blur the actions fully,
ignore their loss, and learn from the frames alone.

**Why it failed us on T5.** The summary vector per image is too coarse: by the time the scene
reaches the transformer, "red cube left, blue cube right" is mashed together. So it learned to
reach for *a* cube, not *the named* cube. UVA keeps a 16×16 grid of patches per image instead.

**UWM vs UVA in one line:** UWM = one model un-blurring actions and video together, from a coarse
summary of the scene; UVA = one model forming a detailed patch-by-patch picture of the future and
reading actions off it, never drawing the video at run time.
