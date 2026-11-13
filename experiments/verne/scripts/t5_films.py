# Films a T5 run on chosen rollout seeds: [front | top | wrist] at the control rate, the prompt and the
# outcome in the file name. Same per-episode torch seed as the sweeps, so outcomes match sweep_<n>.json.
# usage: python experiments/verne/scripts/t5_films.py 07_instruction t5 0 1001 1000 1014   (step, task, seed, rollout seeds...)
import sys
from types import SimpleNamespace

import numpy as np
import torch

from dream_robot.core.video import observation_panel, write_video
from dream_robot.policies.verne import rollouts, vae
from dream_robot.policies.verne.data import REPO, load_table, one_hot
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
for rs in map(int, sys.argv[4:]):
    torch.manual_seed(rs * 100 + n_exec)
    obs, plan, frames = env.reset(seed=rs), [], []
    ins = one_hot(env.prompt)
    for _ in range(env._cfg.horizon_steps):
        if not plan:
            plan = list(rollouts.act(net, enc, d, obs, ins)[:n_exec])
        frames.append(observation_panel(env.render(), obs.images))
        r = env.step(plan.pop(0))
        obs = r.observation
        if r.terminated or r.truncated:
            break
    why = 'placed' if r.success else rollouts.failure_of(env, r)
    words = '_'.join(env.prompt.split()[2:3] + env.prompt.split()[-2:-1])
    path = write_video(frames + [frames[-1]] * 15, REPO / f'scratch/verne07/t5_seed{rs}_{words}_{why}.mp4', fps=env.control_hz)
    print(f'{rs} "{env.prompt}": {why} -> {path.relative_to(REPO)}', flush=True)
env.close()
