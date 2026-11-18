"""Step 14 stage 1 labels: what the captioned-denoising pretraining sees for a few T5 reset scenes.

Each scene (reset seed, no demo) has two cubes: slot 0 'near' (nearer the robot), slot 1 'far'. A
caption names ONE of them, by place + colour ('near red'); training draws which one per sample.
Writes caption_labels.png: the top|wrist canvas per scene, titled with both possible captions.

    python experiments/verne/scripts/caption_labels.py [out.png]
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

from dream_robot.policies.verne.data import WORDS
from dream_robot.policies.verne.rollouts import make_env

out = sys.argv[1] if len(sys.argv) > 1 else 'caption_labels.png'
env = make_env('t5')
seeds = range(5000, 5008)
fig, axes = plt.subplots(4, 2, figsize=(12, 9))
for seed, ax in zip(seeds, axes.flat):
    obs = env.reset(seed=seed)
    ins = env.instruction
    colours = [ins.distractor_color, ins.distractor_color]
    colours[ins.target_slot] = ins.color
    captions = [f'{place} {c}' for place, c in zip(('near', 'far'), colours)]
    rows = [' '.join(f'{int(w in cap.split())}' for w in WORDS) for cap in captions]
    print(f'seed {seed}: slot 0 (near) {colours[0]:6s}  slot 1 (far) {colours[1]:6s}   captions: '
          + '  |  '.join(f'{cap!r} -> [{r}]' for cap, r in zip(captions, rows)))
    ax.imshow(np.concatenate([obs.images['top'], obs.images['wrist']], 1))  # [H, W, 3]: side by side
    ax.set_title(f'seed {seed}:  "{captions[0]}"  or  "{captions[1]}"', fontsize=10)
    ax.axis('off')
env.close()
print('one-hot order:', ' '.join(WORDS))
fig.tight_layout()
fig.savefig(out, dpi=80)
print('wrote', out)
