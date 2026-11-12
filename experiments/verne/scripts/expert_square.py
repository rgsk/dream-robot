# The scripted expert with and without square_up on T1-wide spawns: success rate, and the angle between
# the fingers' closing axis and the nearest cube face normal at the first close (0 = square).
# Films 3 square_up episodes with the most diagonal cubes to scratch/verne_films/.
# usage: python experiments/verne/scripts/expert_square.py 100
import sys

import numpy as np

from dream_robot.core.video import write_video
from dream_robot.policies.verne.rollouts import make_env
from dream_robot.sims.robosuite.tasks.pick_place_cube.demo import rollout_frames
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertPolicy

n = int(sys.argv[1])
env = make_env(wide=True)


def misalign_deg(yaw):
    model, data = env.mujoco  # live after every reset (robosuite rebuilds its sim on reset)
    ax = np.asarray(data.xaxis[model.joint('gripper0_right_finger_joint1').id], float)
    phi = np.arctan2(ax[1], ax[0]) - np.pi / 2  # finger axis is along y at reset: compare to the cube's y face
    return abs(np.degrees((phi - yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4))


yaws = {}
for square in (False, True):
    policy, ok, angles = ExpertPolicy(env, square_up=square), 0, []
    for seed in range(n):
        obs = env.reset(seed=seed)
        policy.reset()
        yaw, closed, was_open = env.cube_yaw, False, False
        yaws[seed] = abs(np.degrees((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4))
        for _ in range(env._cfg.horizon_steps):
            action, _ = policy(obs)
            was_open |= action[7] >= 0.5
            if action[7] < 0.5 and was_open and not closed:  # a close after the gripper has opened
                angles.append(misalign_deg(yaw))
                closed = True
            r = env.step(action)
            obs = r.observation
            if r.terminated or r.truncated:
                break
        ok += bool(r.success)
    a = np.array(angles)
    print(f'square_up={square}: {ok}/{n} placed; fingers off square at close: mean {a.mean():.1f} deg, max {a.max():.1f}', flush=True)
policy = ExpertPolicy(env, square_up=True)
for seed in [] if len(sys.argv) > 2 else sorted(yaws, key=yaws.get)[-3:]:
    frames, placed, why = rollout_frames(env, policy, seed, hold=15)
    path = write_video(frames, f'scratch/verne_films/expert_square_seed{seed}_{yaws[seed]:.0f}deg_{why.replace(" ", "_")}.mp4', fps=env.control_hz)
    print(f'film seed {seed} (cube {yaws[seed]:.0f} deg off square): {why} -> {path}', flush=True)
print('------ FINISHED ------')
