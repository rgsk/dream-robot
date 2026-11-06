"""A person's-eye camera for the kitchen: the sim stand-in for head-camera human video.

Reuses RoboCasa's unused robot0_agentview_left camera: after each reset it is moved to a fixed
spot (a person standing by the counter) and aimed at the object's start, then rendered with
RoboCasa's own renderer (correct textures). The camera's parent is the mobile base, which
never drives here, so the view is fixed for the episode -- a watcher, not a camera on the gripper.

    MUJOCO_GL=egl envs/robocasa/.venv/bin/python experiments/pick_shapes/scripts/head_camera.py
draws scratch/head_views.png: one row per candidate placement (+ wrist), 5 moments of a fruit pick.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

CAM = 'robot0_agentview_left'      # not in the observation; free to move
# name: (ahead, left, height above the object) in the robot's frame, m; optional 4th = field of view, deg
VIEWS = {
    'over_shoulder': (-0.55, 0.25, 0.65),    # behind the gripper and above, like looking over the robot
    'left_side':     (-0.10, 0.65, 0.55),    # a person standing to the robot's left
    'right_side':    (-0.10, -0.65, 0.55),   # ... to its right
    'left_close':    (-0.07, 0.45, 0.40),    # left_side, 30% closer
    'left_zoom':     (-0.10, 0.65, 0.55, 40.0),  # left_side, narrower lens
}
SHOW = ('left_side', 'left_close', 'left_zoom')   # rows drawn by main()


def robot_frame(env) -> tuple[np.ndarray, np.ndarray]:
    # (forward, left) unit vectors of the robot base in the world, horizontal
    qx, qy, qz, qw = env._raw['robot0_base_quat']                 # robosuite: xyzw
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return fwd, np.array([-fwd[1], fwd[0], 0.0])


def place(env, view: str, target: np.ndarray) -> None:
    """Move CAM to VIEWS[view] around target and aim it at target (call after every reset)."""
    model, data = env.mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM)
    fwd, left = robot_frame(env)
    a, l, h, *fov = VIEWS[view]
    pos = target + a * fwd + l * left + np.array([0.0, 0.0, h])
    f = (target - pos) / np.linalg.norm(target - pos)             # camera looks along -z, y up
    r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R_cam = np.stack([r, u, -f], 1)                               # world <- camera
    bid = model.cam_bodyid[cid]
    R_b, p_b = data.xmat[bid].reshape(3, 3), data.xpos[bid]
    model.cam_pos[cid] = R_b.T @ (pos - p_b)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_b.T @ R_cam).ravel())
    model.cam_quat[cid] = q
    model.cam_fovy[cid] = fov[0] if fov else 60.0
    mujoco.mj_camlight(model, data)    # refresh camera poses now (else the render uses the old one)


def frame(env, hw: tuple[int, int] = (256, 256)) -> np.ndarray:
    h, w = hw
    return np.ascontiguousarray(env._env.sim.render(width=w, height=h, camera_name=CAM)[::-1])


def main() -> None:
    import imageio

    from rung0_scripted_pick import PickKitchen, episode, make_skills, mesh_for, pool

    env = PickKitchen()
    skills = make_skills(env)
    seed = 1300
    env.obj_xml = mesh_for(pool('fruit'), seed)
    frames = {v: [] for v in SHOW} | {'wrist': []}
    target = []

    def on_step(obs, action):
        if not target:                    # first step after episode()'s reset: fix the aim point
            target.append(env.obj_pos().copy())
        for v in SHOW:                    # one camera, moved per view (cheap: only cam_pos/quat change)
            place(env, v, target[0])
            frames[v].append(frame(env))
        frames['wrist'].append(obs.images['wrist'])

    ok, _, _ = episode(env, skills, seed, on_step=on_step)
    ts = np.linspace(0, len(frames['wrist']) - 1, 5).round().astype(int)
    rows = list(SHOW) + ['wrist']                 # one row each, top to bottom; 5 moments left to right
    grid = np.concatenate([np.concatenate([np.pad(frames[r][t], ((3, 3), (3, 3), (0, 0)), constant_values=255)
                                           for t in ts], 1) for r in rows], 0)
    out = Path('scratch/head_views.png')
    out.parent.mkdir(exist_ok=True)
    imageio.imwrite(out, grid)
    print(f'{Path(env.obj_xml).parent.name}, seed {seed}, pick {"OK" if ok else "FAIL"}; rows: {rows}; steps {list(ts)}')


if __name__ == '__main__':
    main()
