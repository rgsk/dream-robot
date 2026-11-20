# Noisy replays (DART-style, 2026-11-21): each TRAINING demo (episodes 5-49) of a task replayed once with our goal-pose
# driver, with one push that knocks the hand off the demo's path; the demo's clock pauses (from the push) until the hand is back
# (within HOLD of one of its next path points, never jumping past a grip change), so the replay recovers instead of
# skipping ahead. Labels are what the driver AIMED at each step (the held demo point while paused, then the path),
# never the push: frame off the path -> "go back". Only replays the task's own check calls a success are kept (one
# retry with a fresh push). Same keys as to_npz.py's npz, into data/libero/npz/<suite>_noisy/<task>.npz; episode ids
# 1000 + demo index (libero.py holds out episodes < 5, so none of these).
#
# Push: a direction (side / ahead along the hand's travel / back / up / down), PUSH_STEPS steps of PUSH (OSC units,
# x 5 cm per step at most), ending ~5 steps before the grasp (grasp, 50%) or before the release (place, 50%); no
# mid-carry pushes (2026-11-21 failure modes of the 63/100 run: missed_plate 19, no_grasp 14, dropped 4).
#
#   MUJOCO_GL=egl envs/libero/.venv/bin/python experiments/libero/scripts/noisy_replays.py data/libero/libero_spatial/X_demo.hdf5 ...
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import to_npz  # noqa: E402

PUSH_STEPS, PUSH, HOLD = 10, 0.6, 0.004
WAYS = ('side', 'ahead', 'back', 'up', 'down')
FIRST_TRAIN = 5  # demos 0-4 are libero.py's held-out ones: no replays of them
OUT = to_npz.REPO / 'data/libero/npz'


def hand(env):
    c = env.env.robots[0].controller
    c.update(force=True)
    return np.array(c.ee_pos), np.array(c.ee_ori_mat)


def command(p_now, r_now, p, r, grip) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    a = np.empty(7)
    a[:3] = np.clip((p - p_now) / to_npz.POS_MAX, -1, 1)
    a[3:6] = np.clip(Rotation.from_matrix(r @ r_now.T).as_rotvec() / to_npz.ROT_MAX, -1, 1)
    a[6] = 1.0 if grip > 0 else -1.0
    return a


def replay(env, e, where, rng) -> dict | None:
    # one noisy replay of demo e -> its rows (to_npz's keys), or None if the task wasn't solved
    acts, states = e['actions'][()], e['states'][()]
    sim = env.env.sim
    hp, hr = [], []
    for s in states:  # the demo's hand path and aims (as to_npz)
        sim.set_state_from_flattened(s)
        sim.forward()
        p, r = hand(env)
        hp.append(p)
        hr.append(r)
    path = np.array(hp)
    g_pos, g_rot = to_npz.goals(path, np.array(hr), acts)
    grip = np.sign(acts[:, 6])
    grasp = int(np.argmax(acts[:, 6] > 0))
    release = grasp + int(np.argmax(acts[grasp:, 6] < 0)) if (acts[grasp:, 6] < 0).any() else len(acts) - 1
    way = WAYS[rng.integers(len(WAYS))]
    phase = 'grasp' if rng.random() < 0.5 else 'place'
    at = (grasp if phase == 'grasp' else release) - PUSH_STEPS - 5 + int(rng.integers(-5, 6))
    at = int(np.clip(at, 1, len(acts) - PUSH_STEPS - 1))
    travel = path[min(at + 5, len(path) - 1)] - path[at]
    travel /= np.linalg.norm(travel) + 1e-9
    side = np.cross(travel, [0, 0, 1.0])
    side /= np.linalg.norm(side) + 1e-9
    side *= rng.choice([-1, 1])
    d = {'side': side, 'ahead': travel, 'back': -travel, 'up': np.array([0, 0, 1.0]), 'down': np.array([0, 0, -1.0])}[way]

    env.reset()
    obs = env.set_init_state(states[0])
    sim = env.env.sim  # reset rebuilds it
    names = [n for n in to_npz.NAMES]
    rows = {k: [] for k in ('top', 'wrist', 'joints', 'gripper', 'hand_pos', 'hand_rot', 'goal_pos', 'goal_rot', 'grip',
                            'pix', 'seen')}
    i, ok, recovering = 0, False, False
    for t in range(len(acts) + 150):
        if i >= len(acts):
            break
        p_now, r_now = hand(env)
        rows['top'].append(obs['agentview_image'][::-1])
        rows['wrist'].append(obs['robot0_eye_in_hand_image'][::-1])
        rows['joints'].append(obs['robot0_joint_pos'].astype(np.float32))
        rows['gripper'].append(obs['robot0_gripper_qpos'].astype(np.float32))
        rows['hand_pos'].append(p_now)
        rows['hand_rot'].append(r_now)
        rows['goal_pos'].append(g_pos[i])  # the driver's aim this step (held while the clock is paused)
        rows['goal_rot'].append(g_rot[i])
        rows['grip'].append(acts[i, 6])
        pix = np.full((len(names), 2, 2), np.nan, np.float32)
        seen = np.zeros((len(names), 2), bool)
        for j, n in enumerate(names):
            if n in where:
                pix[j], seen[j] = to_npz.pixels(sim, to_npz.world_point(sim, where[n]))
        rows['pix'].append(pix)
        rows['seen'].append(seen)
        pushing = at <= t < at + PUSH_STEPS
        cmd = command(p_now, r_now, g_pos[i], g_rot[i], acts[i, 6])
        if pushing:
            cmd[:3] = np.clip(PUSH * d, -1, 1)
        obs, _, _, _ = env.step(cmd)
        if pushing:
            recovering = True
        elif not recovering:
            i += 1  # on the path: the demo's clock runs as recorded
        else:  # after the push: hold until the hand is back on one of the next path points, then resume from there
            h = hand(env)[0]
            nxt = np.flatnonzero(grip[i:] != grip[i])
            near = np.linalg.norm(path[i:i + min(40, nxt[0] if len(nxt) else 40) + 1] - h, axis=1)
            if near.min() < HOLD:
                i += int(near.argmin()) + 1
                recovering = False
        if env.check_success():
            ok = True
            break
    if not ok:
        return None
    out = {k: np.stack(v) for k, v in rows.items()}
    out['case'] = f'{way}_{phase}'
    return out


def convert(path: Path, seed: int = 0) -> None:
    stem = path.name.removesuffix('_demo.hdf5')
    verb, obj, target = to_npz.SKILLS[stem]
    where = to_npz.WHERE[stem]
    rng = np.random.default_rng(seed)
    with h5py.File(path) as f:
        env = to_npz.make_env(f, stem)
        demos = sorted(f['data'], key=lambda s: int(s.split('_')[1]))
        demos = demos[:FIRST_TRAIN + int(__import__('os').environ.get('LIMIT', 10**6))]  # LIMIT: a quick test
        parts, cases, tries = [], [], 0
        for k in demos:
            n = int(k.split('_')[1])
            if n < FIRST_TRAIN:
                continue
            for _ in range(2):  # one retry with a fresh push
                tries += 1
                r = replay(env, f['data'][k], where, rng)
                if r is not None:
                    r['episode'] = np.full(len(r['top']), 1000 + n, np.int32)
                    parts.append(r)
                    cases.append(r.pop('case'))
                    break
        env.close()
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    N = len(cat['top'])
    action = np.zeros((N, 7), np.float32)
    action[:, 6] = cat.pop('grip')
    out = OUT / f'{path.parent.name}_noisy'
    out.mkdir(parents=True, exist_ok=True)
    tokens = np.array([to_npz.VERBS.index(verb), to_npz.NAMES.index(obj), to_npz.NAMES.index(target)], np.int64)
    np.savez(out / f'{stem}.npz', top=cat['top'], wrist=cat['wrist'], joints=cat['joints'], gripper=cat['gripper'],
             ee_pos=np.zeros((N, 3), np.float32), ee_ori=np.zeros((N, 3), np.float32), action=action,
             episode=cat['episode'], pix=cat['pix'], seen=cat['seen'],
             hand_pos=cat['hand_pos'].astype(np.float32), hand_rot=cat['hand_rot'].astype(np.float32),
             goal_pos=cat['goal_pos'].astype(np.float32), goal_rot=cat['goal_rot'].astype(np.float32),
             tokens=tokens, verbs=to_npz.VERBS, names=to_npz.NAMES,
             skill=f'{verb}({obj}{", " + target if target != "none" else ""})', cases=np.array(cases))
    kinds = {c: cases.count(c) for c in sorted(set(cases))}
    print(f'{stem}: {len(parts)} replays kept of {tries} tries ({len(demos) - FIRST_TRAIN} demos), {N} steps; {kinds}',
          flush=True)


if __name__ == '__main__':
    import os

    for p in sys.argv[1:]:
        convert(Path(p), seed=int(os.environ.get('SEED', 0)))
