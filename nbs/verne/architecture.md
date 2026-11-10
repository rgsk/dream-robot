Here's the whole model, piece by piece, in the order data flows through it. Everything comes from reading the code in [verne-v0.5](../verne-v0.5). The training code isn't released, so the training part at the end is inferred.

```
3 cameras ─► canvas ─► VAE ─► latents ─┬─► memory packer ─► past tokens ─┐
                                        └─► patcher ───────► now tokens ──┤
                                                                          ▼
instruction ─► UMT5 text encoder ─► text rows ──┐        VIDEO TRANSFORMER (Wan 2.2, ~5B)
robot state ─► linear ─────────────► state row ─┴──►     runs ONCE per chunk, keeps its K,V
                                                                          │ (each layer's K,V)
noise [32 x 30] ─► ACTION TRANSFORMER (narrow) ◄──────────────────────────┘
                   4 denoising steps ─► action chunk [32 x 30] ─► IK ─► joint commands
```

## 1. Camera canvas

**Three images become one wide image**, so the video model sees one picture per moment, the kind of input it was pretrained on.

- Each camera (left wrist, right wrist, head) is resized to 256×256.
- The head camera is first cropped to its centre 87.5%.
- The three are placed side by side: a 256×768 canvas.

## 2. VAE (frozen, from Wan 2.2)

**Turns pixels into latents**, so the transformer never handles raw pixels.

- A latent is a short summary per area: each 16×16 pixel square becomes one cell of 48 numbers. The 256×768 canvas becomes a 16×48 grid. That's about 590k pixel values down to 37k numbers, with the irrelevant detail thrown away and what matters (edges, objects, colours) kept.
- Each frame is encoded on its own, never mixed with neighbouring frames. So a frame encoded once can be reused: it's "now" this chunk, and it moves into memory for later chunks. No frame is ever encoded twice.
- Same idea as the VAE in our dreamzero notebook (128×128 → 16×16×4), only much bigger.

## 3. Patcher: latents to tokens

**Turns the latent grid into a sequence of tokens** for the transformer.

- A small conv cuts the grid into patches and maps each to one vector. Wan's patch is 2×2 latent cells, so roughly 8×24 ≈ 192 tokens per frame. That patch size is Wan's default; I haven't seen the config.

## 4. Memory: the past, packed

**Lets the model remember about the last 17 seconds** without that becoming too many tokens.

- One past frame is kept every 133 ms. There are 16 memory slots, each covering about 1.07 s.
- Recent slots keep fine patches. Older slots are packed coarsely, merging up to 4 frames in time and 4× in space into one token. Like human memory: the last second in detail, a few seconds ago as a rough picture.
- Each slot gets its own learned attention bias, so the model can learn how much to trust old memories.
- This is what our WAMz was missing in the 1013 failure. It saw only "now", so it couldn't remember "I already tried and the cube is still there".

## 5. Text encoder (frozen, UMT5-XXL)

**Turns the instruction into vectors** whose meaning was learned from large amounts of text.

- "Put the cup in the sink" becomes one vector per word piece. It's encoded once per instruction and cached.
- Same idea as Cosmos's T5. It's how colour words get real meaning instead of a lookup-table entry like our colour/side table.

## 6. Robot state: one extra "word"

**Tells the model where the arms are now.**

- The state is the poses and grippers of both arms, plus the right hand's pose relative to the left. It covers the current and previous step, which gives a sense of velocity. 58 numbers in all.
- One linear layer turns it into a single row appended to the text rows. The model reads it exactly like one more word of the instruction.

## 7. Video transformer (Wan 2.2's 5B DiT, the big one)

**The "world understanding" part**: what's where, what's happening, what it means for the task.

- Input: past-memory tokens plus "now" tokens. Each layer also cross-attends to the text and state rows.
- An attention mask enforces time order: each memory slot sees only itself and older slots, and "now" sees all memory plus itself. Nothing looks into the future.
- All video tokens are marked clean (flow time 0). At run time nothing is drawn, so the model is only reading the scene.
- **It runs once per chunk.** At selected layers it saves its keys and values, the "here's what I see" summaries. That's the expensive pass, done once.

## 8. Action transformer (narrow, the "hands")

**Turns that understanding into motion.**

- Input: a noisy action chunk of 32 steps × 30 numbers, each step embedded as one token.
- It has one layer per saved video layer. At each layer, the action tokens attend to:
  - that video layer's saved keys and values (the scene),
  - each other (smooth motion across the chunk),
  - the text and state rows (the goal and where the arms are).
- This "two transformers attending jointly" design is called a mixture of transformers. The video side never reads the actions, which is why its pass can be cached.
- The noise level comes in through adaptive layer norm, the transformer version of our FiLM.

## 9. Sampler: flow matching, 4 steps

**Takes pure noise to a clean action chunk**, exactly like our flow-matching notebook.

- 4 Euler steps. Each step runs only the cheap action transformer against the cached video, which is why a plan takes under 0.4 s despite 8B parameters.
- The step sizes are shifted so more of the effort goes to the noisy early steps.

## 10. Action format: gripper poses, not joints

Each of the 32 rows (1/30 s apart, 1.07 s in total) holds 30 numbers:

```
left arm:  x y z + rotation(6)  + jaw      = 10
right arm: x y z + rotation(6)  + jaw      = 10
base:      x y + heading (as a pose, 9)    =  9
progress:  0..1, how much of the task is done = 1
```

- **Rotation as 6 numbers:** the first two rows of the rotation matrix. That's smoother for a network to output than angles, which wrap around.
- **Relative:** poses are relative to where the hand is now, so "move 2 cm down" looks the same anywhere.
- **Robot-agnostic:** gripper poses don't care about link lengths. That's how one model serves different robots.
- **Progress** works like Cosmos's value. You read it to know when to give the next instruction.

## 11. Real-time chunking

**Keeps motion smooth between plans.** The robot keeps moving while the next plan is computed. Those already-committed steps, up to 12, are pinned as the start of the new chunk, and the model fills in the rest so it continues seamlessly instead of jumping.

## 12. IK, outside the model

**Gripper poses to joint angles**, using the robot's own geometry. This is a small optimisation solved every control tick, not learned. Their example is in [arm_ik.py](../verne-v0.5/examples/robotwin/Verne/arm_ik.py).

## Training (inferred, the code isn't released)

The README says future video is a training-only output. The likely setup: during training, some future frames are added as noisy tokens and the video transformer learns to denoise them, while the action transformer learns to denoise actions. Predicting the future forces the video side to understand dynamics, and the action side benefits. At run time that branch is simply never used. It's the same idea as our WAMz, minus drawing the frames at run time.

**In one line:** a big pretrained video model looks at the present and a compressed past, a narrow action model reads what it saw, and 4 cheap denoising steps produce gripper motion that IK turns into joint angles.
