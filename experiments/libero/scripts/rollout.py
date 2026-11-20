# A trained LIBERO Verne (verne/libero.py stage 2) in LIBERO's sim: each skill from LIBERO's own fixed start states
# (init_files/<suite>/<task>.pruned_init), success = the task's own check. Runs in envs/libero/.venv. The suite is
# libero.py's (VERNE_LIBERO_SUITE): libero_90 = our 4 scene-3 / scene-8 skills; libero_spatial = the benchmark's 10
# tasks, 50 start states each, MAX_STEPS 220 (the published protocol's limit), and the bowl POINT: here the sim's
# (bowl_1's body projected into the settled first frame's top camera, as to_npz labels it) = the ceiling for an LLM's.
#
# Per chunk (every N_EXEC steps): the cameras (rows flipped as to_npz stored them) + [previous, now] joints and
# fingers + the skill words -> Verne samples 16 x 11 (std units) -> raw: 16 aiming poses relative to the hand now,
# gripper, progress -> world poses from the controller's hand at the chunk start (hand.absolute). Each executed step:
# the driver (scripts/hand_check.py) turns that step's pose into LIBERO's OSC delta from the LIVE hand; gripper sign.
#
#   [VERNE_LIBERO_SUITE=libero_spatial] MUJOCO_GL=egl envs/libero/.venv/bin/python \
#       experiments/libero/scripts/rollout.py RUN_DIR WEIGHTS [episodes] [films] [bowl | other]
# 'other' (Spatial, 2026-11-21): point at the OTHER bowl (bowl_2, which no demo of that task moves); success = bowl_2
# on the plate (LIBERO's On), and 'wrong' = bowl_1 ended on the plate instead. A test that the arm follows the point,
# not the layout.
#   e.g. experiments/libero/runs/skills_spread/seed0 ckpt/weights_step10000.pt 20 2
# -> RUN_DIR/rollouts_<weights stem>.json (task -> per episode [success, steps]) and films in
#    scratch_nov/libero/rollouts/<run>_<weights stem>/
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault('LIBERO_CONFIG_PATH', str(REPO / 'envs/libero/config'))

from dream_robot.policies.verne import hand as hand_mod  # noqa: E402
from dream_robot.policies.verne import libero as L  # noqa: E402
from dream_robot.policies.verne.model import Verne, sample  # noqa: E402

N_EXEC = 8  # executed steps per chunk (T5's n_exec 8)
POS_MAX, ROT_MAX = 0.05, 0.5  # OSC_POSE output_max
SUITE = L.SUITE
if SUITE == 'libero_90':
    MAX_STEPS = 400  # 20 s at 20 Hz; the longest training demo is 283 steps
    # task -> the skill tokens to_npz gave it (verb, object, target; into VERBS, ('none',) + NAMES)
    TASKS = {
        'KITCHEN_SCENE3_turn_on_the_stove': (0, 1, 0),
        'KITCHEN_SCENE3_put_the_moka_pot_on_the_stove': (2, 3, 2),
        'KITCHEN_SCENE3_put_the_frying_pan_on_the_stove': (2, 4, 2),
        'KITCHEN_SCENE8_turn_off_the_stove': (1, 1, 0),
    }
    POINT_BODY = None
else:  # libero_spatial: put(bowl, plate) everywhere; longest training demo 197 steps
    MAX_STEPS = 220
    put = (L.VERBS.index('put'), L.NAMES.index('bowl') + 1, L.NAMES.index('plate') + 1)
    TASKS = {f'pick_up_the_black_bowl_{w}_and_place_it_on_the_plate': put for w in (
        'between_the_plate_and_the_ramekin', 'from_table_center', 'in_the_top_drawer_of_the_wooden_cabinet',
        'next_to_the_cookie_box', 'next_to_the_plate', 'next_to_the_ramekin', 'on_the_cookie_box', 'on_the_ramekin',
        'on_the_stove', 'on_the_wooden_cabinet')}
    POINT_BODY = 'akita_black_bowl_1_main'


def make_env(task: str):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path('bddl_files')) / SUITE / f'{task}.bddl'
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    env.seed(0)
    init = torch.load(Path(get_libero_path('init_states')) / SUITE / f'{task}.pruned_init', weights_only=False)
    return env, init


TARGET = 'bowl'  # main() sets it: 'bowl' (the task's) or 'other'


def on_plate(env, obj: str) -> bool:
    od = env.env.object_states_dict
    return bool(od['plate_1'].check_ontop(od[obj]))


def solved(env) -> bool:
    return on_plate(env, 'akita_black_bowl_2') if TARGET == 'other' else env.check_success()


def body_pos(env, name: str) -> np.ndarray:
    sim = env.env.sim
    return sim.data.body_xpos[sim.model.body_name2id(name)].copy()


def failure_mode(env, z0: dict, zmax: dict) -> str:
    # Spatial, a failed episode (as T5's failure modes): the pointed bowl never lifted 3 cm -> no_grasp (or the other
    # bowl lifted -> wrong_bowl); lifted and still 3 cm up at the end -> still_holding; back down within 10 cm of the
    # plate (xy) -> missed_plate (e.g. on its rim); elsewhere -> dropped
    mine = 'akita_black_bowl_2_main' if TARGET == 'other' else POINT_BODY
    theirs = 'akita_black_bowl_1_main' if TARGET == 'other' else 'akita_black_bowl_2_main'
    if zmax[mine] - z0[mine] < 0.03:
        return 'wrong_bowl' if zmax[theirs] - z0[theirs] >= 0.03 else 'no_grasp'
    p = body_pos(env, mine)
    if p[2] - z0[mine] >= 0.03:
        return 'still_holding'
    near = np.linalg.norm((p - body_pos(env, 'plate_1_main'))[:2]) < 0.10
    return 'missed_plate' if near else 'dropped'


def sim_point(env) -> tuple[float, float]:
    # bowl_1's body centre in the upright 128 px top camera (to_npz's own projection), (row, col)
    sys.path.insert(0, str(REPO / 'experiments/libero/scripts'))
    import to_npz

    sim = env.env.sim
    body = 'akita_black_bowl_2_main' if TARGET == 'other' else POINT_BODY
    pix, _ = to_npz.pixels(sim, sim.data.body_xpos[sim.model.body_name2id(body)].copy())
    return float(pix[0, 0]), float(pix[0, 1])


def hand(env):
    c = env.env.robots[0].controller
    c.update(force=True)
    return np.array(c.ee_pos), np.array(c.ee_ori_mat)


def command(p_now, r_now, p, r, grip) -> np.ndarray:
    a = np.empty(7)
    a[:3] = np.clip((p - p_now) / POS_MAX, -1, 1)
    a[3:6] = np.clip(Rotation.from_matrix(r @ r_now.T).as_rotvec() / ROT_MAX, -1, 1)
    a[6] = 1.0 if grip > 0 else -1.0
    return a


def observe(obs) -> tuple[np.ndarray, np.ndarray]:
    # -> canvas [3, 128, 256] uint8 (agentview | wrist, rows flipped upright as to_npz), state [9]
    top, wrist = obs['agentview_image'][::-1], obs['robot0_eye_in_hand_image'][::-1]
    canvas = np.concatenate([top, wrist], 1).transpose(2, 0, 1)
    return canvas, np.concatenate([obs['robot0_joint_pos'], obs['robot0_gripper_qpos']]).astype(np.float32)


def episode(net, st, env, init_state, ins, seed: int, film: list | None) -> tuple[bool, int, tuple | None, str]:
    torch.manual_seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(10):  # LIBERO's eval settles the scene with a few no-op steps first
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    canvas, state = observe(obs)
    prev = state
    plan = []
    point = sim_point(env) if POINT_BODY else None  # given once, held all episode
    bowls = ('akita_black_bowl_1_main', 'akita_black_bowl_2_main') if POINT_BODY else ()
    z0 = {b: body_pos(env, b)[2] for b in bowls}
    zmax = dict(z0)
    pt = None if point is None else L.point_input(torch.tensor([point], dtype=torch.float32).cuda() / 16)
    for t in range(MAX_STEPS):
        if not plan:
            frame = torch.from_numpy(canvas)[None].cuda().float() / 127.5 - 1
            s = L.join(torch.from_numpy(prev)[None].cuda(), torch.from_numpy(state)[None].cuda(), st['s_mean'],
                       st['s_std'], pt)
            raw = (sample(net, frame, s, ins) * st['a_std'] + st['a_mean'])[0].cpu().numpy()  # [16, 11]
            p0, r0 = hand(env)
            pos, rot = hand_mod.absolute(p0[None], r0[None], raw[:, :9])
            plan = [(pos[k], rot[k], raw[k, 9]) for k in range(N_EXEC)]
        p, r, g = plan.pop(0)
        p_now, r_now = hand(env)
        obs, _, done, _ = env.step(command(p_now, r_now, p, r, g))
        if film is not None:
            top = obs['agentview_image'][::-1].copy()
            if point is not None:  # the given point: a red square on the top camera
                r, c = int(round(point[0])), int(round(point[1]))
                top[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = (255, 0, 0)
            film.append(np.concatenate([top, obs['robot0_eye_in_hand_image'][::-1]], 1))
        prev, (canvas, state) = state, observe(obs)
        for b in bowls:
            zmax[b] = max(zmax[b], body_pos(env, b)[2])
        if (done and TARGET == 'bowl') or solved(env):
            return True, t + 1, point, 'none'
    mode = failure_mode(env, z0, zmax) if bowls else 'fail'
    if TARGET == 'other' and on_plate(env, 'akita_black_bowl_1'):
        return False, -1, point, mode  # steps -1: the task's own bowl ended on the plate (wrong bowl)
    return False, MAX_STEPS, point, mode


def main(run: Path, weights: str, episodes: int = 20, films: int = 2, target: str = 'bowl') -> None:
    import imageio.v3 as iio

    global TARGET
    TARGET = target

    st = {k: v.cuda() for k, v in torch.load(run / 'stats.pt').items()}
    net = Verne(n_words=len(L.WORDS), state_dim=L.STATE, frames='pixels').cuda().eval()
    net.load_state_dict(torch.load(run / weights))
    tag = Path(weights).stem + ('' if target == 'bowl' else f'_{target}')
    film_dir = REPO / 'scratch_nov/libero/rollouts' / f'{run.parent.name}_{run.name}_{tag}'
    film_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for task, tokens in TASKS.items():
        env, init = make_env(task)
        ins = L.skill_words(tokens)[None].cuda()
        res = []
        for i in range(min(episodes, len(init))):
            film = [] if i < films else None
            ok, n, point, mode = episode(net, st, env, init[i], ins, seed=i, film=film)
            res.append([ok, n, point, mode])
            if film:
                iio.imwrite(film_dir / f'{task}_ep{i}_{"success" if ok else "fail"}.mp4', np.stack(film), fps=20)
        env.close()
        out[task] = res
        print(f'{task}: {sum(r[0] for r in res)}/{len(res)}  (steps of successes: '
              f'{[r[1] for r in res if r[0]]})' + (f'  wrong bowl {sum(r[1] == -1 for r in res)}' if target != 'bowl' else '')
              + f'  failures {dict(Counter(r[3] for r in res if not r[0]))}', flush=True)
    (run / f'rollouts_{tag}.json').write_text(json.dumps(out, indent=1))
    total = [r[0] for res in out.values() for r in res]
    print(f'total {sum(total)}/{len(total)} = {np.mean(total):.1%}   failures '
          f'{dict(Counter(r[3] for res in out.values() for r in res if not r[0]))}', flush=True)
    print(f'films: {film_dir}\n------ FINISHED ------', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(Path(a[0]), a[1], int(a[2]) if len(a) > 2 else 20, int(a[3]) if len(a) > 3 else 2, a[4] if len(a) > 4 else 'bowl')
