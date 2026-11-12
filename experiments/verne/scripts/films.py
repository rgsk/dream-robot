# Film two Verne runs on the same rollout seeds, side by side: [top | wrist] of run A above run B,
# 3x upscaled, at the task's control rate, the cube's spawn xy and each episode's outcome in the label. Same per-episode torch
# seed as the sweeps (seed * 100 + n_exec), so outcomes match sweep_<n>.json.
# usage: python experiments/verne/scripts/films.py 02_cosine_lr t1-wide 0 1   (step, task, seed A, seed B)
#        rollout seeds: the 4 most extreme cube spawns in probe.json (x min/max, y min/max)
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from dream_robot.core.video import write_video
from dream_robot.policies.verne import rollouts
from dream_robot.policies.verne.data import REPO, load_table
from dream_robot.policies.verne.run import RUNS, TASKS, split

step, task, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
name, wide = TASKS[task]
state, action, episode = load_table(name)
lab = np.isin(episode, split(task)[0])
d = SimpleNamespace(
    **{
        k: torch.tensor(v, device='cuda')
        for k, v in dict(
            s_mean=state[lab].mean(0), s_std=state[lab].std(0, ddof=1),
            a_mean=action[lab].mean(0), a_std=action[lab].std(0, ddof=1),
        ).items()
    }
)
nets = {s: rollouts.load(torch.load(RUNS / step / task / f'seed{s}' / 'weights.pt')) for s in (a, b)}
env = rollouts.make_env(wide)

pts = json.loads((RUNS / step / task / f'seed{a}' / 'probe.json').read_text())['points']
cube = {int(k): v[0] for k, v in pts.items()}
seeds = list(dict.fromkeys(
    [min(cube, key=lambda s: cube[s][0]), max(cube, key=lambda s: cube[s][0]),
     min(cube, key=lambda s: cube[s][1]), max(cube, key=lambda s: cube[s][1])]
))


def episode_frames(net, seed, n_exec=8):
    torch.manual_seed(seed * 100 + n_exec)  # as rollouts.rollout
    obs, plan, frames = env.reset(seed=seed), [], []
    for _ in range(600):
        frames.append(np.concatenate([obs.images['top'], obs.images['wrist']], axis=1))
        if not plan:
            plan = list(rollouts.act(net, d, obs)[:n_exec])
        r = env.step(plan.pop(0))
        obs = r.observation
        if r.terminated or r.truncated:
            break
    frames.append(np.concatenate([obs.images['top'], obs.images['wrist']], axis=1))
    return frames, 'ok' if r.success else str(r.failure_mode)


def label(img, text):
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((28, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


out = REPO / 'scratch/verne_films'
out.mkdir(parents=True, exist_ok=True)
for seed in seeds:
    runs = {s: episode_frames(nets[s], seed) for s in (a, b)}
    n = max(len(f) for f, _ in runs.values())
    x, y = (100 * c for c in cube[seed])
    rows = []
    for s in (a, b):
        f, why = runs[s]
        f = f + [f[-1]] * (n - len(f))  # the shorter episode holds its last frame
        rows.append([label(fr, f'train seed {s}: {why}  (cube x {x:+.1f} y {y:+.1f} cm)') for fr in f])
    frames = [np.concatenate(pair, axis=0) for pair in zip(*rows)]
    tag = '_'.join(runs[s][1] for s in (a, b))
    path = write_video(frames, str(out / f'{step}_{task}_s{a}_vs_s{b}_ep{seed}_{tag}.mp4'), fps=env.control_hz)
    print(seed, {s: runs[s][1] for s in (a, b)}, path, flush=True)
print('------ FINISHED ------')
