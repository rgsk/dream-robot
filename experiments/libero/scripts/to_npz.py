# LIBERO demos -> one .npz per task, the seam between the LIBERO venv (h5py + its sim) and Verne's (main venv).
# Per demo step: both cameras (flipped upright), robot state, the recorded action, the skill call as fixed-vocab
# tokens (verb, object, target; see notes.md "Design"), and position labels for EVERY name in the scene (not just
# the skill's): its centre in each camera + an in-view flag, from setting the recorded sim state and projecting
# the named body / site. Position pretraining asks about all of them. Since 2026-11-20 also the controller's hand pose
# per step (hand_pos [N, 3], hand_rot [N, 3, 3]) and where the demo was aiming it (goal_pos, goal_rot: the recorded
# command applied to the hand), Verne's option-B action targets.
# Also a check film per task (demo_0, labels drawn: object red, target blue).
# Runs in envs/libero/.venv:
#   MUJOCO_GL=egl envs/libero/.venv/bin/python experiments/libero/scripts/to_npz.py data/libero/libero_90/X_demo.hdf5 ...
import json
import os
import sys
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault('LIBERO_CONFIG_PATH', str(REPO / 'envs/libero/config'))
OUT = REPO / 'data/libero/npz'
FILMS = REPO / 'scratch_nov/libero/npz_check'
HW = 128
CAMERAS = ('agentview', 'robot0_eye_in_hand')  # -> 'top', 'wrist' in the npz

# fixed vocabularies: the LLM picks from these; a new word = a new entry at the end (ids stay stable)
VERBS = ('turn_on', 'turn_off', 'put')
# things a slot can name, and that position pretraining asks about. 'stove' = its cook region (a put target),
# 'stove_knob' = the button turn_on / turn_off press: one word per place the hand goes
NAMES = ('none', 'stove_knob', 'stove', 'moka_pot', 'frying_pan')
# where each name sits in a scene: (kind, sim name); missing = not in that scene, or ambiguous (scene 8 has two
# moka pots: 'moka_pot' alone can't say which, so it isn't labelled there)
WHERE = {
    'KITCHEN_SCENE3': {'stove_knob': ('body', 'flat_stove_1_button'), 'stove': ('site', 'flat_stove_1_cook_region'),
                       'moka_pot': ('body', 'moka_pot_1_main'), 'frying_pan': ('body', 'chefmate_8_frypan_1_main')},
    'KITCHEN_SCENE8': {'stove_knob': ('body', 'flat_stove_1_button'), 'stove': ('site', 'flat_stove_1_cook_region')},
}
# task -> the skill call (verb, object, target)
SKILLS = {
    'KITCHEN_SCENE3_turn_on_the_stove': ('turn_on', 'stove_knob', 'none'),
    'KITCHEN_SCENE8_turn_off_the_stove': ('turn_off', 'stove_knob', 'none'),
    'KITCHEN_SCENE3_put_the_moka_pot_on_the_stove': ('put', 'moka_pot', 'stove'),
    'KITCHEN_SCENE3_put_the_frying_pan_on_the_stove': ('put', 'frying_pan', 'stove'),
}


def make_env(f):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    name = Path(json.loads(f['data'].attrs['env_args'])['env_kwargs']['bddl_file_name']).name
    bddl = next(Path(get_libero_path('bddl_files')).glob(f'*/{name}'))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=HW, camera_widths=HW)
    env.seed(0)
    env.reset()
    return env


def world_point(sim, spec):
    kind, name = spec  # ('body' | 'site', sim name)
    if kind == 'body':
        return sim.data.body_xpos[sim.model.body_name2id(name)].copy()
    return sim.data.site_xpos[sim.model.site_name2id(name)].copy()


POS_MAX, ROT_MAX = 0.05, 0.5  # OSC_POSE output_max (the demos' env_args): a command a moves the goal by these


def hand(env) -> tuple[np.ndarray, np.ndarray]:
    # the pose OSC moves (scripts/hand_check.py): the controller's end effector, world frame. The recorded obs
    # ee_pos is ~8 mm off it, so it isn't used
    c = env.env.robots[0].controller
    c.update(force=True)
    return np.array(c.ee_pos), np.array(c.ee_ori_mat)


def goals(pos, rot, acts) -> tuple[np.ndarray, np.ndarray]:
    # where the hand was AIMING each step: the recorded command applied to the hand, p + 0.05·a[:3],
    # exp(0.5·a[3:])·R (hand_check.py: driving to these replays as well as the commands; the achieved next pose
    # drops the push into a knob)
    from scipy.spatial.transform import Rotation

    return pos + POS_MAX * acts[:, :3], Rotation.from_rotvec(ROT_MAX * acts[:, 3:6]).as_matrix() @ rot


def pixels(sim, p):
    # world point -> [2 cameras, (row, col)] in the upright image (the npz's frames), and in-view flags [2].
    # robosuite's own projection rounds to whole pixels and CLAMPS off-image points to the edge (a wrist label sat
    # on the bottom edge while the stove was out of view), so project here: float, no clamp. Its rows are already
    # upright (a flip put the moka pot's dot 40 px off). The matrix is rebuilt every call: the wrist camera moves.
    # Occlusion (an object behind the arm) still counts as in view.
    from robosuite.utils.camera_utils import get_camera_transform_matrix

    out, seen = np.empty((len(CAMERAS), 2), np.float32), np.empty(len(CAMERAS), bool)
    for k, cam in enumerate(CAMERAS):
        h = get_camera_transform_matrix(sim, cam, HW, HW) @ np.append(p, 1.0)
        col, row = h[0] / h[2], h[1] / h[2]
        out[k] = row, col
        seen[k] = h[2] > 0 and 0 <= row <= HW - 1 and 0 <= col <= HW - 1
    return out, seen


def convert(path: Path) -> None:
    stem = path.name.removesuffix('_demo.hdf5')
    verb, obj, target = SKILLS[stem]
    where = WHERE[stem[:stem.index('_', len('KITCHEN_SCENE'))]]
    with h5py.File(path) as f:
        env = make_env(f)
        sim = env.env.sim
        d = f['data']
        demos = sorted(d, key=lambda s: int(s.split('_')[1]))
        cols = {k: [] for k in ('top', 'wrist', 'joints', 'gripper', 'ee_pos', 'ee_ori', 'action', 'episode',
                                'pix', 'seen', 'hand_pos', 'hand_rot', 'goal_pos', 'goal_rot')}
        for i, k in enumerate(demos):
            e, o = d[k], d[k]['obs']
            n = e['actions'].shape[0]
            cols['top'].append(o['agentview_rgb'][()][:, ::-1])
            cols['wrist'].append(o['eye_in_hand_rgb'][()][:, ::-1])
            for c, key in (('joints', 'joint_states'), ('gripper', 'gripper_states'), ('ee_pos', 'ee_pos'),
                           ('ee_ori', 'ee_ori')):
                cols[c].append(o[key][()].astype(np.float32))
            cols['action'].append(e['actions'][()].astype(np.float32))
            cols['episode'].append(np.full(n, i, np.int32))
            # every name in every frame (NaN / unseen where the scene doesn't have it)
            pix = np.full((n, len(NAMES), 2, 2), np.nan, np.float32)
            seen = np.zeros((n, len(NAMES), 2), bool)
            hp, hr = np.empty((n, 3)), np.empty((n, 3, 3))
            for t, s in enumerate(e['states'][()]):
                sim.set_state_from_flattened(s)
                sim.forward()
                hp[t], hr[t] = hand(env)
                for j, name in enumerate(NAMES):
                    if name in where:
                        pix[t, j], seen[t, j] = pixels(sim, world_point(sim, where[name]))
            cols['pix'].append(pix)
            cols['seen'].append(seen)
            gp, gr = goals(hp, hr, e['actions'][()])
            for c, v in (('hand_pos', hp), ('hand_rot', hr), ('goal_pos', gp), ('goal_rot', gr)):
                cols[c].append(v.astype(np.float32))
        env.close()
    arr = {c: np.concatenate(v) for c, v in cols.items()}
    tokens = np.array([VERBS.index(verb), NAMES.index(obj), NAMES.index(target)], np.int64)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'{stem}.npz', **arr, tokens=tokens, verbs=VERBS, names=NAMES,
             skill=f'{verb}({obj}{", " + target if target != "none" else ""})')
    vis = ', '.join(f'{nm} {arr["seen"][:, j, 0].mean():.0%}/{arr["seen"][:, j, 1].mean():.0%}'
                    for j, nm in enumerate(NAMES) if nm in where)
    print(f'{stem}: {verb}({obj}, {target}) {len(demos)} demos, {len(arr["action"])} steps; in view top/wrist: {vis}',
          flush=True)
    film(arr, stem, tokens)


def film(arr, stem: str, tokens) -> None:
    # demo_0 with the skill's labels drawn where in view: object red, target blue, on both cameras
    rows = arr['episode'] == 0
    frames = []
    for top, wrist, pix, seen in zip(arr['top'][rows], arr['wrist'][rows], arr['pix'][rows], arr['seen'][rows]):
        pair = [top.copy(), wrist.copy()]
        for k in range(2):
            for j, colour in ((tokens[1], (255, 0, 0)), (tokens[2], (0, 80, 255))):
                if j and seen[j, k]:
                    r, c = int(round(pix[j, k, 0])), int(round(pix[j, k, 1]))
                    pair[k][max(r - 2, 0):r + 3, max(c - 2, 0):c + 3] = colour
        frames.append(np.concatenate(pair, axis=1))
    FILMS.mkdir(parents=True, exist_ok=True)
    big = np.stack(frames).repeat(2, axis=1).repeat(2, axis=2)  # 256 px tall, easier to see
    iio.imwrite(FILMS / f'{stem}_labels.mp4', big, fps=20)


if __name__ == '__main__':
    for p in sys.argv[1:]:
        convert(Path(p))
