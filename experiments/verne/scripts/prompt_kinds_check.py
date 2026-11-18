"""Step 13 check: near / far words and the three prompt kinds (colour / position / both).

- reword on a recorded prompt, numpy and torch, every kind and slot; T1 rows (slot -1) unchanged
- the T5 demos' slots from instructions.json: share near / far
- get_batch: mix=True draws the kinds about evenly, mix=False keeps the recorded prompt
- the eval prompts (rollouts.prompt_of) on a few live T5 scenes: the place word matches the cube
  nearer the robot base (x -0.56)
"""

import numpy as np
import torch

from dream_robot.policies.verne import rollouts
from dream_robot.policies.verne.data import (
    KINDS,
    WORDS,
    get_batch,
    load_slots,
    one_hot,
    reword,
    to_gpu,
)
from dream_robot.policies.verne.model import Verne


def words(v) -> str:
    return ' '.join(w for w, on in zip(WORDS, np.asarray(v)) if on) or '(none)'


ins = one_hot('put the red cube in the left box')
print('recorded:', words(ins))
for slot in (0, 1):
    for k, kind in enumerate(KINDS):
        a = reword(ins, np.array(slot), np.array(k))
        b = reword(torch.tensor(ins), torch.tensor(slot), torch.tensor(k)).numpy()
        assert (a == b).all()
        print(f'  slot {slot} {kind:8s} -> {words(a)}')
assert (reword(np.zeros(8, np.float32), np.array(-1), np.array(1)) == 0).all()  # T1 stays empty

name = 'instructed_pick_place_square_noise005_n150'
slot = load_slots(name)
print(
    f'T5 demo rows: near {np.mean(slot == 0):.2f}, far {np.mean(slot == 1):.2f}, missing {np.sum(slot < 0)}'
)

# a toy Demos: 2 T5 episodes (slot 0 then 1) + 1 T1 episode, red / left prompt
N = 9
d = to_gpu(
    latent=np.zeros((N, 4, 16, 32), np.float16),
    state=np.random.rand(N, 8).astype(np.float32),
    instruction=np.stack([ins] * 6 + [np.zeros(8, np.float32)] * 3),
    slot=np.array([0] * 3 + [1] * 3 + [-1] * 3),
    action=np.random.rand(N, 8).astype(np.float32),
    episode=np.repeat([0, 1, 2], 3),
    train_ids=np.array([0, 1, 2]),
    dev='cpu',
)
b = get_batch(d, np.array([0, 1]), 3000, np.random.default_rng(0))
seen = {}
for row in b.instruction.numpy():
    seen[words(row)] = seen.get(words(row), 0) + 1
print('mixed batch prompts:', seen)
b = get_batch(d, np.array([0, 1, 2]), 300, np.random.default_rng(0), mix=False)
print('mix=False prompts:', {words(r) for r in b.instruction.numpy()})
print('word table rows:', Verne().video_words.num_embeddings)

env = rollouts.make_env('t5')
for seed in (1000, 1001, 1002, 1003):
    env.reset(seed=seed)
    near = int(np.argmin(np.abs(env._cube_positions()[:, 0] - -0.56)))
    print(
        f'seed {seed}: {env.prompt!r}, named slot {env.instruction.target_slot}, nearer slot {near}'
    )
    for kind in KINDS:
        print(f'    {kind:8s} -> {words(rollouts.prompt_of(env, kind))}')
env.close()
