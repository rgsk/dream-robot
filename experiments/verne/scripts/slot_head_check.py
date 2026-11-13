# step 09 local check: slot labels line up with each episode's prompt, shapes, and 600 training
# steps on T5 (flow loss and slot cross-entropy both go down; held-out slot accuracy printed)
import json

import numpy as np
import torch

from dream_robot.policies.verne.data import REPO, load_instructions, one_hot, load_table, to_gpu
from dream_robot.policies.verne.run import load_task
from dream_robot.policies.verne.train import train

name = 'instructed_pick_place_square_noise005_n150'
kept = [a for a in json.loads((REPO / 'data/robosuite' / name / 'instructions.json').read_text())['attempts'] if a['kept']]
ins, episode = load_instructions(name), load_table(name)[2]
first = np.searchsorted(episode, np.arange(len(kept)))
bad = [i for i, a in enumerate(kept) if not (one_hot(a['prompt']) == ins[first[i]]).all()]
print(f'{len(kept)} kept attempts, prompt mismatches vs episode task: {len(bad)}')

state, action, episode, latent, instruction, slot, train_ids, val_ids, scene = load_task('t5')
print('slot', slot.shape, 'labelled share', (slot >= 0).mean().round(2), 'slot-1 share of labelled', slot[slot >= 0].mean().round(2))
d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
run = train(d, train_ids, val_ids, 3000, seed=0)
print('log', run['log'])
