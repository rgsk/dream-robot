# Does a policy square its wrist up to the cube? For each eval episode (n_exec 8, the sweep's per-episode
# torch seed): how far off square the fingers are at the first close (0 = square; never turning gives the
# cube's own angle, ~21 deg mean), how far the wrist turned, and the outcome. Films the episodes whose
# sweep outcome is in FILM to scratch/verne_films/.
# usage: python experiments/verne/scripts/wrist_check.py 05_square t1-wide-square 0 wrong_target
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

step, task, seed, film = sys.argv[1], sys.argv[2], int(sys.argv[3]), set(sys.argv[4:])
name, wide = TASKS[task]
state, action, episode = load_table(name)
lab = np.isin(episode, split(task)[0])
d = SimpleNamespace(**{k: torch.tensor(v, device='cuda') for k, v in dict(
    s_mean=state[lab].mean(0), s_std=state[lab].std(0, ddof=1),
    a_mean=action[lab].mean(0), a_std=action[lab].std(0, ddof=1)).items()})
run = RUNS / step / task / f'seed{seed}'
net, enc, env = rollouts.load(torch.load(run / 'weights.pt')), vae.load(vae.WEIGHTS), rollouts.make_env(wide)
sweep = json.loads((run / 'sweep_8.json').read_text())
wrap = lambda a: (a + np.pi / 4) % (np.pi / 2) - np.pi / 4  # angle mod 90 deg, in [-45, 45)


def finger_angle():
    model, data = env.mujoco  # live after every reset
    ax = data.xaxis[model.joint('gripper0_right_finger_joint1').id]
    return float(np.arctan2(ax[1], ax[0]))


def label(img, text):
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    bar = np.zeros((28, img.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0)


rows = []
for ep in range(1000, 1050):
    torch.manual_seed(ep * 100 + 8)
    obs, plan, frames, was_open, at_close = env.reset(seed=ep), [], [], False, None
    yaw, a0 = env.cube_yaw, finger_angle()
    for t in range(600):
        frames.append(np.concatenate([obs.images['top'], obs.images['wrist']], axis=1))
        if not plan:
            plan = list(rollouts.act(net, enc, d, obs)[:8])
        a = plan.pop(0)
        was_open |= a[7] >= 0.5
        if a[7] < 0.5 and was_open and at_close is None:
            turned = finger_angle() - a0
            at_close = (abs(np.degrees(wrap(turned - wrap(yaw)))), np.degrees(wrap(turned)))
        r = env.step(a)
        obs = r.observation
        if r.terminated or r.truncated:
            break
    why = 'ok' if r.success else str(r.failure_mode)
    off, turned = at_close or (np.nan, np.nan)
    rows.append((ep, np.degrees(wrap(yaw)), turned, off, why))
    print(f'ep {ep}: cube {np.degrees(wrap(yaw)):+5.1f} deg, wrist turned {turned:+5.1f}, off square at close {off:4.1f}  -> {why} (sweep {sweep[str(ep)][1]})', flush=True)
    if sweep[str(ep)][1] in film:
        text = f'{step} {task} seed {seed}, ep {ep}: {why}  cube {np.degrees(wrap(yaw)):+.0f} deg, turned {turned:+.0f}'
        path = write_video([label(f, text) for f in frames], str(REPO / 'scratch/verne_films' / f'{step}_s{seed}_ep{ep}_{why}.mp4'), fps=env.control_hz)
        print(f'   film -> {path}', flush=True)
off = np.array([r[3] for r in rows])
ok = np.array([r[4] == 'ok' for r in rows])
cube = np.abs([r[1] for r in rows])
print(f'off square at close: mean {np.nanmean(off):.1f} deg (successes {np.nanmean(off[ok]):.1f}, failures {np.nanmean(off[~ok]):.1f}); cube itself {cube.mean():.1f}')
t, c = np.array([r[2] for r in rows]), np.array([r[1] for r in rows])
m = ~np.isnan(t)
print(f'wrist turn vs cube angle: slope {np.polyfit(c[m], t[m], 1)[0]:+.2f}, corr {np.corrcoef(c[m], t[m])[0, 1]:+.2f}  (expert: slope 1)')
print('------ FINISHED ------')
