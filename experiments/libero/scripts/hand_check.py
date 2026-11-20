# Option B check (2026-11-20, as T5 step 20b; LEADS: aim 1-3 recorded steps ahead): can LIBERO's arm be driven by HAND POSES instead of its recorded
# commands? Per demo: from its first saved state, every control step the target = the controller's hand pose at the
# demo's NEXT recorded state (read by setting that state on a private copy), turned into LIBERO's OSC_POSE delta
# command by the driver below; gripper = the recorded command. Closed loop: the delta is from the live hand.
#
#   OSC_POSE (control_delta): a command a in [-1, 1]^6 moves the goal by  Δp = 0.05 · a[:3]  (metres) and
#   R_goal = exp(0.5 · a[3:]) · R_now  (axis-angle, world frame). So to reach (p, R):
#       a[:3] = clip((p − p_now) / 0.05),   a[3:] = clip(axis_angle(R · R_nowᵀ) / 0.5)
#
# Prints per task: success replaying the recorded commands (the reference) vs driving by hand poses; the hand's
# distance from the recorded path. Runs in envs/libero/.venv:
#   MUJOCO_GL=egl envs/libero/.venv/bin/python experiments/libero/scripts/hand_check.py data/libero/libero_90/*.hdf5
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

os.environ.setdefault('LIBERO_CONFIG_PATH', str(Path(__file__).resolve().parents[3] / 'envs/libero/config'))
N_DEMOS = 10
LEADS = ()  # 2026-11-20: was (1, 2, 3); aim-ahead didn't fix turn_off. aim this many recorded steps ahead (the controller only covers part of a step's move)
POS_MAX, ROT_MAX = 0.05, 0.5  # the controller's output_max (env_args)


def make_env(f):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    name = Path(json.loads(f['data'].attrs['env_args'])['env_kwargs']['bddl_file_name']).name
    bddl = next(Path(get_libero_path('bddl_files')).glob(f'*/{name}'))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    env.seed(0)
    env.reset()
    return env


def hand(env) -> tuple[np.ndarray, np.ndarray]:
    # the pose OSC moves: the controller's end effector, position [3] and rotation [3, 3], world frame
    c = env.env.robots[0].controller
    c.update(force=True)
    return np.array(c.ee_pos), np.array(c.ee_ori_mat)


def recorded_hands(env, states) -> tuple[np.ndarray, np.ndarray]:
    # the hand at every recorded state (set the state, read the controller); restores nothing: call before a replay
    pos, rot = [], []
    for s in states:
        env.env.sim.set_state_from_flattened(s)
        env.env.sim.forward()
        p, r = hand(env)
        pos.append(p)
        rot.append(r)
    return np.array(pos), np.array(rot)


def goals(pos, rot, acts) -> tuple[np.ndarray, np.ndarray]:
    # where the demo hand was AIMING at each step: the recorded command applied to the recorded hand (header's
    # OSC rule): p_goal = p_t + 0.05·a[:3], R_goal = exp(0.5·a[3:])·R_t. Under contact the hand stops short of it
    n = len(acts)
    gp = pos[:n] + POS_MAX * acts[:, :3]
    gr = Rotation.from_rotvec(ROT_MAX * acts[:, 3:6]).as_matrix() @ rot[:n]
    return gp, gr


def command(p_now, r_now, p, r, grip) -> np.ndarray:
    # the driver (header): world target pose -> OSC delta command [7]
    a = np.empty(7)
    a[:3] = np.clip((p - p_now) / POS_MAX, -1, 1)
    a[3:6] = np.clip(Rotation.from_matrix(r @ r_now.T).as_rotvec() / ROT_MAX, -1, 1)
    a[6] = grip
    return a


def run(env, e, mode: str, pos=None, rot=None, lead: int = 1, goal=None) -> tuple[bool, float]:
    # one demo from its first state; mode 'commands' (recorded) or 'hands' (driver to the next recorded hand pose)
    # -> success, mean distance of the hand from the recorded path (mm)
    env.reset()
    env.set_init_state(e['states'][0])
    acts = e['actions'][()]
    dist = []
    for t, a in enumerate(acts):
        if mode == 'goal':  # aim at the demo's goal pose of this step
            p_now, r_now = hand(env)
            a = command(p_now, r_now, goal[0][t], goal[1][t], a[6])
        elif mode == 'hands':
            p_now, r_now = hand(env)
            k = min(t + lead, len(pos) - 1)
            a = command(p_now, r_now, pos[k], rot[k], a[6])
        env.step(a)
        if pos is not None:
            dist.append(np.linalg.norm(hand(env)[0] - pos[min(t + 1, len(pos) - 1)]))
    return bool(env.check_success()), 1000 * float(np.mean(dist)) if dist else float('nan')


def main(paths):
    for path in map(Path, paths):
        with h5py.File(path) as f:
            env = make_env(f)
            d = f['data']
            res = {'commands': [], 'goal': []} | {f'hands+{n}': [] for n in LEADS}
            for k in sorted(d, key=lambda s: int(s.split('_')[1]))[:N_DEMOS]:
                e = d[k]
                pos, rot = recorded_hands(env, e['states'][()])
                res['commands'].append(run(env, e, 'commands', pos, rot))
                res['goal'].append(run(env, e, 'goal', pos, rot, goal=goals(pos, rot, e['actions'][()])))
                for n in LEADS:
                    res[f'hands+{n}'].append(run(env, e, 'hands', pos, rot, n))
            env.close()
        line = '  '.join(f'{m}: {sum(ok for ok, _ in r)}/{len(r)} (path {np.mean([x for _, x in r]):.1f} mm)'
                         for m, r in res.items())
        print(f'{path.name.removesuffix("_demo.hdf5")}: {line}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
