# A trained LIBERO Verne (verne/libero.py stage 2) in LIBERO's sim: each skill from LIBERO's own fixed start states
# (init_files/libero_90/<task>.pruned_init), success = the task's own check. Runs in envs/libero/.venv.
#
# Per chunk (every N_EXEC steps): the cameras (rows flipped as to_npz stored them) + [previous, now] joints and
# fingers + the skill words -> Verne samples 16 x 11 (std units) -> raw: 16 aiming poses relative to the hand now,
# gripper, progress -> world poses from the controller's hand at the chunk start (hand.absolute). Each executed step:
# the driver (scripts/hand_check.py) turns that step's pose into LIBERO's OSC delta from the LIVE hand; gripper sign.
#
#   MUJOCO_GL=egl envs/libero/.venv/bin/python experiments/libero/scripts/rollout.py RUN_DIR WEIGHTS [episodes] [films]
#   e.g. experiments/libero/runs/skills_spread/seed0 ckpt/weights_step10000.pt 20 2
# -> RUN_DIR/rollouts_<weights stem>.json (task -> per episode [success, steps]) and films in
#    scratch_nov/libero/rollouts/<run>_<weights stem>/
import json
import os
import sys
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
MAX_STEPS = 400  # 20 s at 20 Hz; the longest training demo is 283 steps
POS_MAX, ROT_MAX = 0.05, 0.5  # OSC_POSE output_max
# task -> its bddl file (libero_90) and the skill tokens to_npz gave it (verb, object, target)
TASKS = {
    'KITCHEN_SCENE3_turn_on_the_stove': (0, 1, 0),
    'KITCHEN_SCENE3_put_the_moka_pot_on_the_stove': (2, 3, 2),
    'KITCHEN_SCENE3_put_the_frying_pan_on_the_stove': (2, 4, 2),
    'KITCHEN_SCENE8_turn_off_the_stove': (1, 1, 0),
}


def make_env(task: str):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path('bddl_files')) / 'libero_90' / f'{task}.bddl'
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
    env.seed(0)
    init = torch.load(Path(get_libero_path('init_states')) / 'libero_90' / f'{task}.pruned_init', weights_only=False)
    return env, init


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


def episode(net, st, env, init_state, ins, seed: int, film: list | None) -> tuple[bool, int]:
    torch.manual_seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(10):  # LIBERO's eval settles the scene with a few no-op steps first
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0]))
    canvas, state = observe(obs)
    prev = state
    plan = []
    for t in range(MAX_STEPS):
        if not plan:
            frame = torch.from_numpy(canvas)[None].cuda().float() / 127.5 - 1
            s = (torch.from_numpy(np.concatenate([prev, state]))[None].cuda() - st['s_mean'].repeat(2)) / st['s_std'].repeat(2)
            raw = (sample(net, frame, s, ins) * st['a_std'] + st['a_mean'])[0].cpu().numpy()  # [16, 11]
            p0, r0 = hand(env)
            pos, rot = hand_mod.absolute(p0[None], r0[None], raw[:, :9])
            plan = [(pos[k], rot[k], raw[k, 9]) for k in range(N_EXEC)]
        p, r, g = plan.pop(0)
        p_now, r_now = hand(env)
        obs, _, done, _ = env.step(command(p_now, r_now, p, r, g))
        if film is not None:
            film.append(np.concatenate([obs['agentview_image'][::-1], obs['robot0_eye_in_hand_image'][::-1]], 1))
        prev, (canvas, state) = state, observe(obs)
        if done or env.check_success():
            return True, t + 1
    return False, MAX_STEPS


def main(run: Path, weights: str, episodes: int = 20, films: int = 2) -> None:
    import imageio.v3 as iio

    st = {k: v.cuda() for k, v in torch.load(run / 'stats.pt').items()}
    net = Verne(n_words=len(L.WORDS), state_dim=L.STATE, frames='pixels').cuda().eval()
    net.load_state_dict(torch.load(run / weights))
    tag = Path(weights).stem
    film_dir = REPO / 'scratch_nov/libero/rollouts' / f'{run.parent.name}_{run.name}_{tag}'
    film_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for task, tokens in TASKS.items():
        env, init = make_env(task)
        ins = L.skill_words(tokens)[None].cuda()
        res = []
        for i in range(min(episodes, len(init))):
            film = [] if i < films else None
            ok, n = episode(net, st, env, init[i], ins, seed=i, film=film)
            res.append([ok, n])
            if film:
                iio.imwrite(film_dir / f'{task}_ep{i}_{"success" if ok else "fail"}.mp4', np.stack(film), fps=20)
        env.close()
        out[task] = res
        print(f'{task}: {sum(ok for ok, _ in res)}/{len(res)}  (steps of successes: '
              f'{[n for ok, n in res if ok]})', flush=True)
    (run / f'rollouts_{tag}.json').write_text(json.dumps(out, indent=1))
    print(f'films: {film_dir}\n------ FINISHED ------', flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(Path(a[0]), a[1], int(a[2]) if len(a) > 2 else 20, int(a[3]) if len(a) > 3 else 2)
