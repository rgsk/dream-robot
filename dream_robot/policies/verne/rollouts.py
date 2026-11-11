"""A trained Verne in the sim: one rollout, a sweep over the eval seeds, and the close-point
probe (does the gripper close where the cube is?).

T1 is the cube task as recorded (cube spawn +-3 cm); T1-wide spawns it over x +-10 cm,
y -10..+6 cm, so the policy has to find the cube to grasp it.
"""

import logging
import os
from collections import Counter

import numpy as np
import torch

from .data import Demos, frame_of
from .model import Verne, sample

SEEDS = range(1000, 1050)  # rollout seeds, the same for every model
WIDE_XY = ((-0.10, 0.10), (-0.10, 0.06))  # T1-wide's cube spawn, x and y [min, max] in m


def make_env(wide: bool):
    # T1 or T1-wide; robosuite's import-time warnings silenced
    os.environ.setdefault('MUJOCO_GL', 'egl')
    logging.disable(logging.WARNING)
    import robosuite  # noqa: F401

    logging.disable(logging.NOTSET)
    logging.getLogger('robosuite_logs').setLevel(logging.ERROR)
    from dataclasses import replace

    from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube, TaskConfig

    config = TaskConfig.load()
    return PickPlaceCube(replace(config, cube_xy_range=WIDE_XY) if wide else config)


def load(weights: dict) -> Verne:
    net = Verne().cuda().eval()
    net.load_state_dict(weights)
    return net


def act(net: Verne, d: Demos, obs) -> np.ndarray:
    # the observation -> [CHUNK, 8] raw actions
    s = (torch.from_numpy(obs.state)[None].cuda() - d.s_mean) / d.s_std
    chunk = (sample(net, frame_of(obs), s) * d.a_std + d.a_mean)[0].cpu().numpy()
    chunk[:, 7] = np.clip(chunk[:, 7], 0.0, 1.0)  # gripper commands stay in [0, 1]
    return chunk


def rollout(net: Verne, d: Demos, env, seed: int, n_exec: int) -> tuple[bool, str, int]:
    # one episode, replanning after n_exec actions -> success, failure mode, steps
    obs, plan = env.reset(seed=seed), []
    for step in range(600):
        if not plan:
            plan = list(act(net, d, obs)[:n_exec])
        r = env.step(plan.pop(0))
        obs = r.observation
        if r.terminated or r.truncated:
            break
    return r.success, str(r.failure_mode), step + 1


def sweep(net: Verne, d: Demos, env, n_exec: int) -> dict[int, tuple[bool, str, int]]:
    # every eval seed; prints the score and the failure modes
    torch.manual_seed(n_exec)
    res = {seed: rollout(net, d, env, seed, n_exec) for seed in SEEDS}
    fails = Counter(mode for ok, mode, _ in res.values() if not ok)
    n_ok = sum(ok for ok, _, _ in res.values())
    print(f'n_exec {n_exec:2d}: {n_ok:2d}/{len(res)}   failures {dict(fails)}')
    return res


def close_point(net: Verne, d: Demos, env, seed: int, n_exec: int):
    # -> cube xy at spawn, gripper xy at the first commanded close (None if it never closes)
    obs, plan = env.reset(seed=seed), []
    cube = env.cube_pos[:2]
    for _ in range(600):
        if not plan:
            plan = list(act(net, d, obs)[:n_exec])
        a = plan.pop(0)
        if a[7] < 0.5:
            return cube, np.asarray(env._raw['robot0_eef_pos'][:2], dtype=float).copy()
        r = env.step(a)
        obs = r.observation
        if r.terminated or r.truncated:
            break
    return cube, None


def close_points(net: Verne, d: Demos, env, n_exec: int) -> dict:
    # close_point for every eval seed
    torch.manual_seed(n_exec)
    return {seed: close_point(net, d, env, seed, n_exec) for seed in SEEDS}


def close_fit(points: dict) -> dict[str, float]:
    # how the close point follows the cube (slope 1 = on it, 0 = blind); prints and returns it
    done = [(c, g) for c, g in points.values() if g is not None]
    C, G = np.array([c for c, _ in done]), np.array([g for _, g in done])
    miss = np.linalg.norm(G - C, axis=1).mean() * 100
    fit = {'closes': len(done), 'miss_cm': miss}
    for k, name in enumerate('xy'):
        slope, icpt = np.polyfit(C[:, k], G[:, k], 1)
        fit |= {f'slope_{name}': slope, f'icpt_{name}_cm': icpt * 100}
        fit[f'corr_{name}'] = np.corrcoef(C[:, k], G[:, k])[0, 1]
    print(
        f'closes {len(done)}/{len(points)}, miss {miss:4.1f} cm, '
        f'x slope {fit["slope_x"]:+.2f} (corr {fit["corr_x"]:+.2f}, intercept {fit["icpt_x_cm"]:+.1f} cm), '
        f'y slope {fit["slope_y"]:+.2f} (corr {fit["corr_y"]:+.2f}, intercept {fit["icpt_y_cm"]:+.1f} cm)'
    )
    return fit
