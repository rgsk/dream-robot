# Experiments

Experiments to take Verne to once T1 / T5 are solved (spread, step 22: T5 93% over 5 seeds, wrong cube 0):
harder tasks, and checks of the idea itself. Each entry: what it is, why it's hard, how to train it, how to test
it, the first cheap step. Not started unless marked.

## Two-clause T5: "red left, blue right" from single-move demos (proposed 2026-11-20)

- **Task:** one instruction, two moves: red cube into the left box, then blue into the right. Trained only
  on single moves (no demo ever shows both).
- **Why it's hard:**
  1. **The instruction loses order.** Today it's an 8-word bag (WORDS one-hot): "red left, blue right" sets
     the same 4 words as "red right, blue left". Fix: two ordered clause slots (8 + 8).
  2. **The second move starts somewhere new.** Demos always start at home with both cubes on the table;
     after move 1 the arm is over a box and one cube is already in it.
  3. **Progress and stopping.** Nothing says "red is done, now blue": the policy must see it in the frame,
     and stop once both are done.
- **Training (still single moves):** the scripted reset sometimes starts with clause 1 already done (red in
  the left box, arm at a random pose); the demo, labelled with the full two-clause instruction, does only
  blue -> right. Sometimes both are done: the demo goes home. Rule learned: find the first clause not done
  yet, do it. At test time, doing move 1 sets up move 2.
- **Aux label:** the step 22 position target becomes **the cube of the first unfinished clause** (from the
  sim at record time). The patches must track progress, not just colour: the present-patches idea under
  a moving target.
- **Test (rollouts):** clauses done per episode (0 / 1 / 2), in order or not, wrong cube, still after done.
- **Baseline to beat: the LLM split.** The planner sends "red left", waits, then sends "blue right"; the
  policy only has to handle problem 2. Two arms:
  - (a) split: today's spread model, one clause at a time, from a mid-task start. No new training.
  - (b) in-policy: two-clause instruction + done-clause demos + moving position target.
  (b) ≈ (a) means progress tracking lives in the patches.
- **First step (eval only):** step 22 spread model, reset with one cube already in a box, prompt the other
  cube's clause. If it fails, mid-task starts in the demos come first; if it works, go to (b).
- Later: the clause list comes from an LLM in one go (same policy input), then the planner with retries.
