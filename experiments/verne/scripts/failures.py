# Film one Verne run's failed episodes (and one success for reference): [top | wrist], 3x upscaled, at
# the task's control rate, label = outcome + cube xy; and print where the gripper is, relative to the
# cube, each time it commands close. Same per-episode torch seed as the sweeps, so outcomes match
# sweep_<n>.json (up to GPU differences).
# usage: python experiments/verne/scripts/failures.py 03_vae_latents t1-wide 0 6   (step, task, seed, n failures)
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from dream_robot.core.video import write_video
from dream_robot.policies.verne import rollouts, vae
from dream_robot.policies.verne.data import REPO, load_table
from dream_robot.policies.verne.run import RUNS, TASKS, split

step, task, seed, n_fail = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
name, wide = TASKS[task]
state, action, episode = load_table(name)
lab = np.isin(episode, split(task)[0])
d = SimpleNamespace(**{k: torch.tensor(v, device='cuda') for k, v in dict(
    s_mean=state[lab].mean(0), s_std=state[lab].std(0, ddof=1),
    a_mean=action[lab].mean(0), a_std=action[lab].std(0, ddof=1)).items()})
run = RUNS / step / task / f'seed{seed}'
net, enc = rollouts.load(torch.load(run / 'weights.pt')), vae.load(vae.WEIGHTS)
env = rollouts.make_env(wide)
sweep = json.loads((run / 'sweep_8.json').read_text())
fails = [int(s) for s, r in sweep.items() if not r[0]][:n_fail]
ok = [int(s) for s, r in sweep.items() if r[0]][:1]


def label(img, text):
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((28, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


out = REPO / 'scratch/verne_films'
out.mkdir(parents=True, exist_ok=True)
for ep in fails + ok:
    torch.manual_seed(ep * 100 + 8)  # as rollouts.rollout, n_exec 8
    obs, plan, frames, closes, was_open = env.reset(seed=ep), [], [], [], True
    cube0 = env.cube_pos.copy()
    for t in range(600):
        frames.append(np.concatenate([obs.images['top'], obs.images['wrist']], axis=1))
        if not plan:
            plan = list(rollouts.act(net, enc, d, obs)[:8])
        a = plan.pop(0)
        if a[7] < 0.5 and was_open:  # a new close command: gripper vs cube now, in cm
            g, c = np.asarray(env._raw['robot0_eef_pos'], dtype=float), env.cube_pos
            closes.append((t, *np.round((g - c) * 100, 1)))
        was_open = a[7] >= 0.5
        r = env.step(a)
        obs = r.observation
        if r.terminated or r.truncated:
            break
    frames.append(np.concatenate([obs.images['top'], obs.images['wrist']], axis=1))
    why = 'ok' if r.success else str(r.failure_mode)
    text = f'{step} {task} seed {seed}, ep {ep}: {why}  (cube x {cube0[0]*100:+.1f} y {cube0[1]*100:+.1f} cm)'
    path = write_video([label(f, text) for f in frames], str(out / f'{step}_{task}_s{seed}_ep{ep}_{why}.mp4'), fps=env.control_hz)
    print(f'ep {ep}: {why} (sweep: {sweep[str(ep)][1]}), {len(frames)} steps, {path.name}', flush=True)
    print(f'   closes (step, gripper - cube dx dy dz cm): {closes[:6]}{" ..." if len(closes) > 6 else ""}', flush=True)
print('------ FINISHED ------')
