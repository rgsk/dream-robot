"""Step 20b check, before any training: drive the sim with T5 demos' own recorded targets as hand moves (data.targets:
relative to the hand at each chunk's start) through hand.Driver (back to world poses, converged IK to joints),
replanning every 8 steps from the arm's actual pose, as a rollout does. Against replaying the recorded joint
targets as they are. Both open loop on the recorded episode (the demos were recorded with joint noise, so neither
replay retraces it exactly).

    python experiments/verne/scripts/hand_replay_check.py [episodes]
"""

import json
import sys

import numpy as np

from dream_robot.policies.verne import hand
from dream_robot.policies.verne.data import REPO, chunk_rows, targets, to_gpu
from dream_robot.policies.verne.rollouts import make_env
from dream_robot.policies.verne.run import TASKS, load_task

n_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
state, action, episode, latent, instruction, slot, train_ids, val_ids, _ = load_task('t5')
d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
print('action stats  mean', np.round(d.a_mean.cpu().numpy(), 3))
print('              std ', np.round(d.a_std.cpu().numpy(), 3))

# targets rebuild the recorded goal poses
now = np.flatnonzero(np.isin(episode, train_ids))[::50]
rows = chunk_rows(d, now)
raw = targets(d, now, rows).cpu().numpy()
p, R = hand.absolute(d.hand_pos[now].cpu().numpy()[:, None], d.hand_rot[now].cpu().numpy()[:, None], raw[..., :9])
print(f'targets -> goal poses: {np.abs(p - d.goal_pos[rows].cpu().numpy()).max() * 1e6:.2f} um, '
      f'{hand.rotation_error_deg(R, d.goal_rot[rows].cpu().numpy()).max():.4f} deg')

kept = [a for a in json.loads((REPO / 'data/robosuite' / TASKS['t5'][0] / 'instructions.json').read_text())['attempts'] if a['kept']]
env = make_env('t5')
results = {'joints (recorded)': [], 'hand moves (Driver)': []}
for e in range(n_eps):
    first = int(np.searchsorted(episode, e))
    last = int(np.searchsorted(episode, e, side='right')) - 1
    for how in results:
        obs, plan = env.reset(seed=kept[e]['seed']), []
        drv = hand.Driver(env)
        for step in range(env._cfg.horizon_steps):
            row = min(first + step, last)
            if how.startswith('joints'):
                a = action[row]
            else:
                if not plan:
                    chunk = targets(d, np.array([row]), chunk_rows(d, np.array([row])))[0].cpu().numpy()
                    plan = drv.plan(obs.state[:7], chunk[:8])
                a = drv.action(obs.state[:7], plan.pop(0))
            r = env.step(a)
            obs = r.observation
            if r.terminated or r.truncated:
                break
        results[how].append(bool(r.success))
    print(f'episode {e:3d} seed {kept[e]["seed"]:4d}: ' + '  '.join(f'{k} {v[-1]}' for k, v in results.items()), flush=True)
for k, v in results.items():
    print(f'{k:22s} {sum(v)}/{len(v)}')
print('------ FINISHED ------')
