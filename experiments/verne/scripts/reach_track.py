# Track the gripper's xy every step for two Verne runs on the same rollout seeds: where it parks
# relative to the cube, and whether it ever moves back toward the robot base after its furthest reach.
# usage: python experiments/verne/scripts/reach_track.py 02_cosine_lr t1-wide 0 1 10   (.. n episodes)
import sys
from types import SimpleNamespace

import numpy as np
import torch

from dream_robot.policies.verne import rollouts
from dream_robot.policies.verne.data import REPO, load_table
from dream_robot.policies.verne.run import RUNS, TASKS, split

step, task, a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
name, wide = TASKS[task]
state, action, episode = load_table(name)
lab = np.isin(episode, split(task)[0])
d = SimpleNamespace(**{k: torch.tensor(v, device='cuda') for k, v in dict(
    s_mean=state[lab].mean(0), s_std=state[lab].std(0, ddof=1),
    a_mean=action[lab].mean(0), a_std=action[lab].std(0, ddof=1)).items()})
env = rollouts.make_env(wide)
base = np.asarray(env._env.robots[0].base_pos if hasattr(env, '_env') else [np.nan] * 3)
print('robot base', base, flush=True)
out = {}
for s in (a, b):
    net = rollouts.load(torch.load(RUNS / step / task / f'seed{s}' / 'weights.pt'))
    for seed in range(1000, 1000 + n):
        torch.manual_seed(seed * 100 + 8)
        obs, plan, eef = env.reset(seed=seed), [], []
        cube = env.cube_pos[:2].copy()
        for _ in range(600):
            eef.append(np.asarray(env._raw['robot0_eef_pos'], dtype=float).copy())
            if not plan:
                plan = list(rollouts.act(net, d, obs)[:8])
            r = env.step(plan.pop(0))
            obs = r.observation
            if r.terminated or r.truncated:
                break
        out[s, seed] = (cube, np.array(eef), bool(r.success))
np.save(REPO / 'scratch/reach_track.npy', {'base': base, 'runs': out}, allow_pickle=True)
for (s, seed), (cube, eef, ok) in out.items():
    i = np.argmax(eef[:, 0])  # furthest +x
    back = eef[i, 0] - eef[i:, 0].min()  # how far x comes back after its max
    tail = eef[-100:, :2].mean(0) - cube  # where it parks, rel. to the cube (last 100 steps)
    print(f'seed {s} ep {seed} ok {ok:d}  cube {cube[0]*100:+5.1f} {cube[1]*100:+5.1f}  '
          f'max x {eef[i,0]*100:+5.1f} at {i:3d}  back after {back*100:4.1f} cm  '
          f'park-cube dx {tail[0]*100:+5.1f} dy {tail[1]*100:+5.1f}', flush=True)
print('------ FINISHED ------')
