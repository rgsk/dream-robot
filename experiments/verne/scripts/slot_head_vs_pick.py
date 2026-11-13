# step 09: per rollout seed, the slot head's answer on the first frame (home pose) next to the slot the arm
# closed on (probe.json, nearer cube) and the named slot -> head right/wrong x arm right/wrong.
# usage: python experiments/verne/scripts/slot_head_vs_pick.py experiments/verne/runs/09_slot_head/t5/seed0
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.data import frame_of, one_hot
from dream_robot.policies.verne.rollouts import load, make_env
from dream_robot.policies.verne.run import load_task
from dream_robot.policies.verne.data import to_gpu

run = Path(sys.argv[1])
p = json.loads((run / 'probe.json').read_text())
state, action, episode, latent, instruction, slot, train_ids, _, _ = load_task('t5')
d = to_gpu(latent[:1], state, instruction[:1], slot[:1], action, episode, train_ids)  # stats only
net = load(torch.load(run / 'weights.pt'))
vae = vae_mod.load(vae_mod.WEIGHTS)
env = make_env('t5')
table, rows = Counter(), []
for seed, (cube, grip) in p['points'].items():
    obs = env.reset(seed=int(seed))
    named = env.instruction.target_slot
    s = (torch.from_numpy(obs.state)[None].cuda() - d.s_mean) / d.s_std
    i = torch.from_numpy(one_hot(env.prompt))[None].cuda()
    with torch.no_grad():
        prob = net.see(frame_of(obs, vae), s, i)[1].float().softmax(-1)[0]
    head = int(prob.argmax())
    xs = env._cube_positions()[:, 0]
    arm = None if grip is None else int(abs(grip[0] - xs[1]) < abs(grip[0] - xs[0]))
    table[('head ' + ('right' if head == named else 'wrong'),
           'arm ' + ('none' if arm is None else 'right' if arm == named else 'wrong'),
           'arm = head' if arm == head else 'arm != head')] += 1
    rows.append((seed, env.prompt, named, head, round(float(prob[named]), 2), arm))
env.close()
for k, v in sorted(table.items()):
    print(f'{v:3d}  {k}')
print('seed, prompt, named slot, head, p(named), arm slot')
for r in rows:
    print(*r, sep='  ')
