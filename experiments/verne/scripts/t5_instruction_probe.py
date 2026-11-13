# Does a T5 policy use the colour word? On the eval seeds, the cube it closes on (nearer slot at the first
# commanded close) under three instructions, same torch noise per episode:
#   true     the real one-hot
#   swapped  colour switched to the other cube's colour, side kept
#   zeroed   colour part zeroed, side kept
# A policy that reads colour goes for the other cube when swapped. Offline too: first-chunk change from the
# swap vs from sampling noise alone (same instruction, another noise seed), in raw joint units.
# usage: python experiments/verne/scripts/t5_instruction_probe.py 07_instruction t5 0
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import torch

from dream_robot.policies.verne import rollouts, vae
from dream_robot.policies.verne.data import WORDS, load_table, one_hot
from dream_robot.policies.verne.run import RUNS, TASKS, split

step, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
name, scene = TASKS[task]
state, action, episode = load_table(name)
lab = np.isin(episode, split(task, int(episode.max()) + 1)[0])
d = SimpleNamespace(**{
    k: torch.tensor(v, device='cuda') for k, v in dict(
        s_mean=state[lab].mean(0), s_std=state[lab].std(0), a_mean=action[lab].mean(0), a_std=action[lab].std(0),
    ).items()
})
net = rollouts.load(torch.load(RUNS / step / task / f'seed{seed}' / 'weights.pt'))
enc, env, n_exec = vae.load(vae.WEIGHTS), rollouts.make_env(scene), 8


def variant(kind, ins_obj):
    v = one_hot(env.prompt)
    if kind == 'swapped':
        v[:4] = 0
        v[WORDS.index(ins_obj.distractor_color)] = 1
    elif kind == 'zeroed':
        v[:4] = 0
    return v


def picked_slot(rs, kind):
    # replay to the first commanded close; -> slot of the nearer cube (or None), the named slot
    torch.manual_seed(rs * 100 + n_exec)
    obs, plan = env.reset(seed=rs), []
    ins = variant(kind, env.instruction)
    for _ in range(env._cfg.horizon_steps):
        if not plan:
            plan = list(rollouts.act(net, enc, d, obs, ins)[:n_exec])
        a = plan.pop(0)
        if a[7] < 0.5:
            g = np.asarray(env._raw['robot0_eef_pos'][:2])
            xy = env._cube_positions()[:, :2]
            return int(np.argmin(np.linalg.norm(xy - g, axis=1))), env.instruction.target_slot
        r = env.step(a)
        obs = r.observation
        if r.terminated or r.truncated:
            break
    return None, env.instruction.target_slot


# offline: the first chunk at reset
swap_d, noise_d = [], []
for rs in rollouts.SEEDS:
    obs = env.reset(seed=rs)
    true_v, swap_v = variant('true', env.instruction), variant('swapped', env.instruction)
    chunks = []
    for v, ns in ((true_v, 0), (swap_v, 0), (true_v, 1)):
        torch.manual_seed(rs * 100 + ns)
        chunks.append(rollouts.act(net, enc, d, obs, v)[:, :7])
    swap_d.append(np.abs(chunks[1] - chunks[0]).mean())
    noise_d.append(np.abs(chunks[2] - chunks[0]).mean())
print(f'first chunk, mean |joint change| (rad): colour swapped {np.mean(swap_d):.4f}, '
      f'noise only {np.mean(noise_d):.4f}', flush=True)

# closed loop: which cube
picks = {k: {} for k in ('true', 'swapped', 'zeroed')}
for kind in picks:
    for rs in rollouts.SEEDS:
        picks[kind][rs] = picked_slot(rs, kind)
    named = sum(p == t for p, t in picks[kind].values())
    print(f'{kind:8s}: closes on the named cube {named}/50 '
          f'(named = the true instruction\'s cube)', flush=True)
same_sw = sum(picks['true'][s][0] == picks['swapped'][s][0] for s in rollouts.SEEDS)
same_z = sum(picks['true'][s][0] == picks['zeroed'][s][0] for s in rollouts.SEEDS)
print(f'same cube as with the true instruction: swapped {same_sw}/50, zeroed {same_z}/50')
print('swapped, went to the cube now named (the true distractor):',
      sum(picks['swapped'][s][0] == 1 - picks['swapped'][s][1] for s in rollouts.SEEDS), '/50')
env.close()
