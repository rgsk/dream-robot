"""Rung 0 data: a scripted top-down picker lifting one fixed apple mesh in one fixed kitchen.

The picker reads the apple's position from the sim (it only makes training data; it never
drives at test time). Only the apple's spot on the counter changes with the seed; the
kitchen (layout 1, style 1) and the robot's spawn stay put.

Recipe: open, go 15 cm above the apple, down to its centre, close, lift 15 cm.
Success = apple 10 cm above where it started and the gripper still holding.

    MUJOCO_GL=egl envs/robocasa/.venv/bin/python experiments/pick_shapes/scripts/rung0_scripted_pick.py --seeds 0 1 2 3 4
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
from robocasa.environments.kitchen.kitchen import FixtureType, Kitchen as RCKitchen

from dream_robot.sims.robocasa.kitchen import Kitchen
from dream_robot.sims.robosuite.skills import ArmSkills

ASSETS = Path('envs/robocasa/src/robocasa/robocasa/models/assets/objects').resolve()
APPLE = str(ASSETS / 'objaverse/apple/apple_0/model.xml')
OUT = Path('experiments/pick_shapes/rung0')
ABOVE, LIFT, SUCCESS_LIFT = 0.15, 0.15, 0.10
# Straight-down descent. The apple is 6.8 cm wide and the open fingers 8 cm apart: ~6 mm a
# side. One move_to from above to the centre curved (the IK goes straight in joint space)
# and came down 2-3 cm off-centre -- a finger clipped the apple before the close. So: align
# above to ALIGN_TOL, then drop through waypoints every DESCENT_STEP, each held to ALIGN_TOL.
ALIGN_TOL, DESCENT_STEP = 0.004, 0.02


def make_skills(env) -> ArmSkills:
    from dataclasses import replace
    from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertConfig
    return ArmSkills(env, replace(ExpertConfig.load(), fine_tolerance=ALIGN_TOL))


def descend(skills, start, obs, step):
    # move_to through points on the vertical line over the object, top to bottom
    res = None
    for h in np.arange(ABOVE - DESCENT_STEP, -1e-9, -DESCENT_STEP):
        res = skills.move_to(start + [0, 0, max(h, 0.0)], obs, step)
        obs = skills._last_obs
        if not res.ok:
            break
    return res


class PickOne(RCKitchen):
    """One object on a counter in front of the robot, nothing else. Defining the class
    registers it with robosuite, so create_env finds it by name."""

    def __init__(self, obj_groups=APPLE, *args, **kwargs):
        self.obj_groups = obj_groups
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        # Rung 0 pins the spot: layout 1's main counter, its long clear top (x 1.7-2.75 m,
        # the stretch nearest the stove). The short top sits by the coffee machine, the right
        # counter by the toaster -- an appliance knocked the apple off in seed 2.
        self.counter = self.register_fixture_ref('counter', dict(id='counter_main_main_group'))
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta['lang'] = f'pick up the {self.get_obj_lang()}'
        return meta

    def _get_obj_cfgs(self):
        return [dict(name='obj', obj_groups=self.obj_groups, graspable=True, init_robot_here=True,
                     placement=dict(fixture=self.counter, size=(0.4, 0.3), pos=(0, -1.0),
                                    sample_region_kwargs=dict(ref='stove_main_group', loc='nn')))]

    def _check_success(self):
        return False  # judged by the script (lift height), not here


class PickKitchen(Kitchen):
    """Our wrapper, building PickOne in a fixed kitchen with the robot spawned without jitter.
    ``obj_xml``: the object's model.xml; set it before a seeded reset to swap the mesh."""

    obj_xml = APPLE

    def _build(self, seed: int):
        import robocasa.utils.env_utils as env_utils
        from dream_robot.sims.robocasa.kitchen import CAMERAS, _joint_position_arm

        with _joint_position_arm(env_utils):
            return env_utils.create_env(
                'PickOne', seed=seed, obj_groups=self.obj_xml, camera_names=list(CAMERAS),
                camera_heights=self._camera_hw[0], camera_widths=self._camera_hw[1],
                layout_and_style_ids=[(1, 1)], robot_spawn_deviation_pos_x=0.0,
                robot_spawn_deviation_pos_y=0.0,
            )

    def obj_pos(self) -> np.ndarray:
        rc = self._env
        return np.array(rc.sim.data.body_xpos[rc.obj_body_id['obj']])


def episode(env: PickKitchen, skills: ArmSkills, seed: int, on_step=None) -> tuple[bool, list, dict]:
    # on_step(obs, action): called before every control step with what the robot saw and did
    obs = env.reset(seed=seed)
    skills.on_reset()
    frames = []
    last = [obs]

    def step(a):
        if on_step is not None:
            on_step(last[0], a)
        r = env.step(a)
        last[0] = r.observation
        frames.append(np.concatenate([env.frame('scene', (256, 256)), env.frame('wrist', (256, 256))], 1))
        return r.observation, False

    start = env.obj_pos()
    log = {'seed': seed, 'apple': start.round(3).tolist()}
    plan = [
        ('above', lambda o: skills.move_to(start + [0, 0, ABOVE], o, step)),
        ('down', lambda o: descend(skills, start, o, step)),
        ('close', lambda o: skills.gripper(True, o, step)),
        ('lift', lambda o: skills.move_by(np.array([0, 0, LIFT]), o, step)),
    ]
    for name, act in plan:
        res = act(obs)
        obs = skills._last_obs
        log[name] = res.phase
        if not res.ok and name != 'down':  # 'down' may stall on the counter: fine, close anyway
            break
    lifted = env.obj_pos()[2] - start[2]
    log['lifted_m'] = round(float(lifted), 3)
    log['holding'] = skills.holding(obs)
    return bool(lifted > SUCCESS_LIFT and log['holding']), frames, log


# Rung 2 training pool (objaverse): apple_0 + apple_11..22 (apple_1..10 are rung 1's test),
# every orange, peach, lemon. Test categories tomato and onion stay out.
def pool(name: str) -> list[str]:
    if name == 'apple0':
        return [APPLE]
    if name == 'rung2':
        obj = ASSETS / 'objaverse'
        apples = [obj / 'apple' / f'apple_{i}' for i in [0, *range(11, 23)]]
        others = [d for c in ('orange', 'peach', 'lemon') for d in sorted((obj / c).iterdir())]
        return [str(d / 'model.xml') for d in apples + others if (d / 'model.xml').exists()]
    raise ValueError(name)


def mesh_for(meshes: list[str], seed: int) -> str:
    # the mesh a seed gets: fixed per seed, so a recording and its check agree
    return meshes[np.random.default_rng(seed).integers(len(meshes))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--pool', default='apple0', help='apple0 (rung 0) or rung2')
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meshes = pool(args.pool)
    env = PickKitchen()
    skills = make_skills(env)
    wins = 0
    for seed in args.seeds:
        env.obj_xml = mesh_for(meshes, seed)
        ok, frames, log = episode(env, skills, seed)
        log['mesh'] = Path(env.obj_xml).parent.name
        log['size_m'] = np.round(env._env.objects['obj'].size, 3).tolist()
        wins += ok
        imageio.mimsave(args.out / f'seed{seed}.mp4', frames, fps=20)
        print('OK ' if ok else 'FAIL', log, flush=True)
    print(f'{wins}/{len(args.seeds)} lifted')


if __name__ == '__main__':
    main()
