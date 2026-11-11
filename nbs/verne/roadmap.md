# Verne, small: roadmap

Step 0 (`dream_robot/policies/verne/model.py`) is Verne V0.5's shape with every box at its dumbest.
Each item below is one box: what step 0 does, what Verne does, and why it might matter.

## Boxes to open

- **Time embedding + adaptive norm** (action side)
  - step 0: raw τ (one number, 0..1) → MLP (1 → 128 → 128), added to every action token.
  - Verne (`action_dit.py`): τ → sinusoid features (cos/sin at many frequencies) → MLP → 6 vectors
    per layer (shift, scale, gate for attention and MLP: adaptive layer norm).
  - why: sinusoids let the network tell nearby noise levels apart (0.90 vs 0.95) and act
    differently at each; raw τ through one SiLU only gives smooth, gentle functions of τ. Step 0
    gets away with it because it samples just 10 evenly spaced levels.
  - test: swap in sinusoids (+ adaptive norm), compare held-out loss and success.
