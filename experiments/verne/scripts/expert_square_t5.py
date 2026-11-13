# T5 (instructed_pick_place) expert with square_up. Checks:
#  1. square_up off still gives the recorded clean demos' actions (T2's own __call__ was dropped)
#  2. env.cube_yaw (the named cube, from body_xmat) == T1's quaternion formula when the named cube is slot 0
#  3. square_up off / on over n seeds: placed, failure modes, fingers off square at the first close
# usage: python experiments/verne/scripts/expert_square_t5.py 50
import logging
import os
import sys
from collections import Counter

os.environ.setdefault('MUJOCO_GL', 'egl')
logging.disable(logging.WARNING)
import robosuite  # noqa: F401,E402

logging.disable(logging.NOTSET)
logging.getLogger('robosuite_logs').setLevel(logging.ERROR)
import numpy as np  # noqa: E402

from dream_robot.policies.verne.data import load_table  # noqa: E402
from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import (  # noqa: E402
    InstructedPickPlace,
    TaskConfig,
)
from dream_robot.sims.robosuite.tasks.instructed_pick_place.expert import ExpertPolicy  # noqa: E402

n = int(sys.argv[1])
env = InstructedPickPlace(TaskConfig.load(), pairs='seen', wording='template')

# 1. regression against the recorded clean demos (episode i = seed i: 150/150 kept)
_, action, episode = load_table('instructed_pick_place_clean_n150')
policy = ExpertPolicy(env)
worst = 0.0
for seed in range(3):
    obs = env.reset(seed=seed)
    policy.reset()
    rec = action[episode == seed]
    for t in range(len(rec)):
        a, _ = policy(obs)
        worst = max(worst, float(np.abs(a - rec[t]).max()))
        r = env.step(a)
        obs = r.observation
        if r.terminated or r.truncated:
            break
print(f'1. square_up off vs recorded demos, seeds 0-2: max |diff| {worst:.2e}', flush=True)

# 2. the named cube's yaw
diffs = []
for seed in range(20):
    env.reset(seed=seed)
    if env.instruction.target_slot == 0:
        x, y, z, w = np.asarray(env._raw['cube_quat'], float)
        t1 = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        diffs.append(abs((env.cube_yaw - t1 + np.pi) % (2 * np.pi) - np.pi))
print(f'2. cube_yaw vs T1 formula on {len(diffs)} slot-0 seeds: max diff {max(diffs):.2e} rad', flush=True)


def misalign_deg(yaw):
    model, data = env.mujoco
    ax = np.asarray(data.xaxis[model.joint('gripper0_right_finger_joint1').id], float)
    phi = np.arctan2(ax[1], ax[0]) - np.pi / 2
    return abs(np.degrees((phi - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4))


# 3. off / on
for square in (False, True):
    policy, ok, angles, modes = ExpertPolicy(env, square_up=square), 0, [], Counter()
    for seed in range(n):
        obs = env.reset(seed=seed)
        policy.reset()
        yaw, closed, was_open = env.cube_yaw, False, False
        while True:
            a, _ = policy(obs)
            was_open |= a[7] >= 0.5
            if a[7] < 0.5 and was_open and not closed:
                angles.append(misalign_deg(yaw))
                closed = True
            r = env.step(a)
            obs = r.observation
            if r.terminated or r.truncated:
                break
        ok += bool(r.success)
        modes[str(r.failure_mode.value)] += not r.success
    a = np.array(angles)
    print(f'3. square_up={square}: {ok}/{n} placed, failures {dict(+modes)}; '
          f'off square at close: mean {a.mean():.1f} deg, max {a.max():.1f}', flush=True)
env.close()
