"""Roll out a trained FM checkpoint on rung 0: same kitchen, apple at new spots (unseen seeds).

Runs in envs/robocasa/.venv. If the checkpoint was trained with observation.touch in its
state, touch is read live (same Touch as the recorder, read before each step, as recorded)
and appended. Success = apple 10 cm above its start at the end of the episode.

    MUJOCO_GL=egl envs/robocasa/.venv/bin/python experiments/pick_shapes/scripts/rung0_eval.py \
        experiments/pick_shapes/rung0/fm_touch/checkpoint.pt --seeds 2000-2019
"""

import argparse
import json
import sys
from pathlib import Path

import imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rung0_record import Touch  # noqa: E402
from rung0_scripted_pick import ASSETS, SUCCESS_LIFT, PickKitchen  # noqa: E402

from dream_robot.core.schema import Observation  # noqa: E402
from dream_robot.policies.flow_matching.policy import FlowPolicy  # noqa: E402

HORIZON = 200      # control steps (10 s); the scripted demos take ~90
VIDEOS = 5         # first few episodes get a video


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--seeds', default='2000-2019')
    ap.add_argument('--meshes', nargs='*', default=[],
                    help='objaverse apple meshes, e.g. apple_1 apple_2; episode i uses mesh i % n')
    ap.add_argument('--out', default='eval')
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split('-'))
    policy = FlowPolicy.load(args.checkpoint)
    extra = policy._net.config.extra_state
    assert extra in ((), ('observation.touch',)), extra
    out = args.checkpoint.parent / args.out
    out.mkdir(exist_ok=True)

    env = PickKitchen()
    touch = Touch(env)
    results = []
    for k, seed in enumerate(range(lo, hi + 1)):
        mesh = args.meshes[k % len(args.meshes)] if args.meshes else 'apple_0'
        env.obj_xml = str(ASSETS / 'objaverse/apple' / mesh / 'model.xml')
        obs = env.reset(seed=seed)
        size = np.round(env._env.objects['obj'].size, 3).tolist()
        touch.reset()
        policy.reset()
        start = env.obj_pos()
        frames = []
        for _ in range(HORIZON):
            state = obs.state
            if extra:
                state = np.concatenate([state, touch.read()]).astype(np.float32)
            action = policy(Observation(state=state, images=obs.images))
            obs = env.step(action).observation
            if k < VIDEOS:
                frames.append(np.concatenate([env.frame('scene', (256, 256)), env.frame('wrist', (256, 256))], 1))
        lifted = float(env.obj_pos()[2] - start[2])
        ok = lifted > SUCCESS_LIFT
        results.append({'seed': seed, 'mesh': mesh, 'size_m': size, 'ok': ok, 'lifted_m': round(lifted, 3)})
        print('OK ' if ok else 'FAIL', results[-1], flush=True)
        if frames:
            imageio.mimsave(out / f'seed{seed}.mp4', frames, fps=20)
    wins = sum(r['ok'] for r in results)
    print(f'{wins}/{len(results)} lifted')
    (out / 'results.json').write_text(json.dumps({'wins': wins, 'n': len(results), 'episodes': results}, indent=1))


if __name__ == '__main__':
    main()
