"""Record rung 0 demos: the scripted apple picker, with touch, as raw episodes.

Runs in envs/robocasa/.venv. Writes one .npz per kept episode; rung0_to_lerobot.py (main venv)
turns them into a LeRobot v3.0 dataset. The split exists because this venv's LeRobot is 0.3.3
and writes an older format than the main venv's 0.5.1 reads.

Per step (what the robot saw, then what it did):
    state  [8]   7 joints + gripper opening, as every Panda dataset here
    action [8]   absolute joint targets + gripper command
    front, wrist [H, W, 3] uint8
    touch  [8]   finger_left, finger_right   normal force on each finger, N (sum over contacts)
                 fx fy fz tx ty tz           wrist force/torque at the gripper mount, N and N·m,
                                             tared at the episode's first step (as a real F/T is)

Touch is the robot's own sensing; the picker still aims from sim state. Only successful
episodes are kept.

    MUJOCO_GL=egl envs/robocasa/.venv/bin/python experiments/pick_shapes/scripts/rung0_record.py --episodes 3
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rung0_scripted_pick import PickKitchen, episode, make_skills  # noqa: E402


TOUCH_NAMES = ['finger_left', 'finger_right', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']
FINGERS = ('gripper0_right_leftfinger', 'gripper0_right_rightfinger')
CAMERAS = ('front', 'wrist')


class Touch:
    """Reads the 8 touch numbers from the live sim. Call ``reset`` after every env reset."""

    def __init__(self, env: PickKitchen):
        self.env = env

    def reset(self) -> None:
        model, _ = self.env.mujoco
        # every body under each finger (pads hang below the finger body)
        self.finger_bodies = []
        for name in FINGERS:
            under = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)}
            for b in range(model.nbody):
                p = b
                while p > 0 and p not in under:
                    p = model.body_parentid[p]
                if p in under:
                    under.add(b)
            self.finger_bodies.append(under)
        self.tare = None

    def read(self) -> np.ndarray:
        model, data = self.env.mujoco
        fingers = [0.0, 0.0]
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            for k, bodies in enumerate(self.finger_bodies):
                if (b1 in bodies) != (b2 in bodies):  # finger against something else
                    mujoco.mj_contactForce(model, data, i, f6)
                    fingers[k] += abs(f6[0])
        robot = self.env._env.robots[0]
        ft = np.concatenate([robot.ee_force['right'], robot.ee_torque['right']])
        if self.tare is None:
            self.tare = ft.copy()
        return np.array(fingers + list(ft - self.tare), dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--seed-start', type=int, default=1000)  # clear of the 0-19 picker check
    ap.add_argument('--name', default='pick_apple_rung0')
    args = ap.parse_args()
    raw = Path('data/robocasa') / f'{args.name}_n{args.episodes}_raw'
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)

    env = PickKitchen()
    skills = make_skills(env)
    touch = Touch(env)
    kept, log = 0, []
    seed = args.seed_start
    while kept < args.episodes and seed < args.seed_start + 3 * args.episodes:
        buf = []

        def on_step(obs, action):
            if not buf:
                touch.reset()
            buf.append((obs, np.asarray(action, np.float32), touch.read()))

        ok, _, info = episode(env, skills, seed, on_step=on_step)
        info['steps'] = len(buf)
        # bump: largest finger force while the gripper is still commanded open (should be 0)
        info['bump_N'] = round(float(max((t[:2].max() for _, a, t in buf if a[7] > 0.5), default=0.0)), 2)
        info['kept'] = ok
        log.append(info)
        print('KEEP' if ok else 'DROP', info, flush=True)
        if ok:
            np.savez_compressed(
                raw / f'ep{kept:04d}_seed{seed}.npz',
                state=np.stack([o.state for o, _, _ in buf]).astype(np.float32),
                action=np.stack([a for _, a, _ in buf]),
                touch=np.stack([t for _, _, t in buf]),
                **{cam: np.stack([o.images[cam] for o, _, _ in buf]) for cam in CAMERAS},
                task=env.prompt, seed=seed,
            )
            kept += 1
        seed += 1
    meta = {'fps': env.control_hz, 'cameras': CAMERAS, 'touch_names': TOUCH_NAMES,
            'robot_type': 'panda_omron', 'episodes': log}
    (raw / 'recording_log.json').write_text(json.dumps(meta, indent=1))
    print(f'kept {kept} of {len(log)} -> {raw}')


if __name__ == '__main__':
    main()
