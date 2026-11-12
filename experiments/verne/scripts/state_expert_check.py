# StateExpert (expert.py), the DAgger labeller: (1) alone on spawns 0..29; (2) taking over from a policy's
# own failed episodes, 40 steps after the policy's first close. Films a few takeovers.
# usage: python experiments/verne/scripts/state_expert_check.py 05_square t1-wide-square 0
import json
import sys
from types import SimpleNamespace

import numpy as np
import torch

from dream_robot.core.video import write_video
from dream_robot.policies.verne import rollouts, vae
from dream_robot.policies.verne.data import REPO, load_table
from dream_robot.policies.verne.run import RUNS, TASKS, split
from dream_robot.sims.robosuite.tasks.pick_place_cube.demo import observation_panel
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import StateExpert

step, task, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
env = rollouts.make_env(True)
expert = StateExpert(env, square_up=True)

ok, phases = 0, {}
for s in range(30):
    obs = env.reset(seed=s)
    expert.reset()
    for _ in range(600):
        r = env.step(expert(obs))
        obs = r.observation
        phases[expert.phase] = phases.get(expert.phase, 0) + 1
        if r.terminated or r.truncated:
            break
    ok += bool(r.success)
print(f'alone: {ok}/30 placed; steps per phase {phases}', flush=True)

name, wide = TASKS[task]
state, action, episode = load_table(name)
lab = np.isin(episode, split(task)[0])
d = SimpleNamespace(
    **{
        k: torch.tensor(v, device='cuda')
        for k, v in dict(
            s_mean=state[lab].mean(0),
            s_std=state[lab].std(0, ddof=1),
            a_mean=action[lab].mean(0),
            a_std=action[lab].std(0, ddof=1),
        ).items()
    }
)
run = RUNS / step / task / f'seed{seed}'
net, enc = rollouts.load(torch.load(run / 'weights.pt')), vae.load(vae.WEIGHTS)
fails = [int(e) for e, r in json.loads((run / 'sweep_8.json').read_text()).items() if not r[0]]
rescued, filmed = 0, 0
for ep in fails:
    torch.manual_seed(ep * 100 + 8)
    obs, plan, frames, was_open, closed_at, t = env.reset(seed=ep), [], [], False, None, 0
    expert.reset()
    for t in range(600):  # the policy drives until 40 steps after its first close
        frames.append(observation_panel(env.render(), obs.images))
        if not plan:
            plan = list(rollouts.act(net, enc, d, obs)[:8])
        a = plan.pop(0)
        was_open |= a[7] >= 0.5
        if a[7] < 0.5 and was_open and closed_at is None:
            closed_at = t
        expert(obs)  # keep its grip-settle window current
        r = env.step(a)
        obs = r.observation
        if r.terminated or r.truncated or (closed_at is not None and t >= closed_at + 40):
            break
    took = t
    if r.terminated or r.truncated:  # the policy's episode ended before a takeover
        print(
            f'ep {ep}: ended under the policy at step {t} -> {"placed" if r.success else r.failure_mode}',
            flush=True,
        )
        continue
    seen = []
    for _ in range(600 - t):
        frames.append(observation_panel(env.render(), obs.images))
        r = env.step(expert(obs))
        obs = r.observation
        if not seen or seen[-1] != expert.phase:
            seen.append(expert.phase)
        if r.terminated or r.truncated:
            break
    why = 'placed' if r.success else str(r.failure_mode)
    rescued += bool(r.success)
    print(f'ep {ep}: takeover at step {took}, phases {" > ".join(seen)} -> {why}', flush=True)
    if ep in (1024, 1025) or (filmed < 3 and 'reopen' in seen):
        path = write_video(
            frames,
            str(REPO / f'scratch/verne_films/state_expert_takeover_ep{ep}_{why}.mp4'),
            fps=env.control_hz,
        )
        print(f'   film -> {path}', flush=True)
        filmed += 1
print(f'takeover from {step} {task} seed {seed} failures: {rescued}/{len(fails)} placed')
print('------ FINISHED ------')
