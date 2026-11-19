"""Step 20a checks: forward kinematics (hand.Kinematics) against the sim, the relative <-> absolute round trip, and
whether one IK step from a recorded state toward the hand pose of the recorded joint target reaches it.

    python experiments/verne/scripts/hand_check.py
"""

import numpy as np

from dream_robot.policies.verne import hand
from dream_robot.policies.verne.data import load_table
from dream_robot.policies.verne.rollouts import make_env
from dream_robot.sims.robosuite.ik import DifferentialIK

state, action, episode = load_table('instructed_pick_place_square_noise005_n150')
print('state', state.shape, 'action', action.shape)

# 1. FK equals the sim: a T5 env, the arm moved by real actions; our FK of its joints vs its own grip site
env = make_env('t5')
env.reset(seed=3)
kin = hand.Kinematics(env.mujoco[0], env.arm_qpos_idx)
errs = []
for t in range(60):
    model, data = env.mujoco
    q = np.asarray(data.qpos[env.arm_qpos_idx])
    p, R = kin.pose(q[None])
    site = mujoco_site = data.site_xpos[kin.site]
    errs.append(np.linalg.norm(p[0] - site))
    env.step(np.concatenate([q + 0.01 * np.sin(t / 5), [1.0]]))
print(f'1. FK vs the sim\'s grip site over 60 steps: max {max(errs) * 1000:.4f} mm')

# 2. hand poses of the demos: states and joint targets
ps, Rs = kin.pose(state[:, :7])
pa, Ra = kin.pose(action[:, :7])
step = np.diff(ps, axis=0)[np.diff(episode) == 0]
print(f'2. hand speed per control step: median {np.median(np.linalg.norm(step, axis=1)) * 1000:.1f} mm, '
      f'max {np.linalg.norm(step, axis=1).max() * 1000:.1f} mm')

# 3. relative -> absolute round trip (target 15 steps ahead, relative to the hand now)
i = np.arange(len(state) - 15)
i = i[episode[i] == episode[i + 15]]
rel = hand.relative(ps[i], Rs[i], pa[i + 15], Ra[i + 15])
p_back, R_back = hand.absolute(ps[i], Rs[i], rel)
print(f'3. round trip: position {np.abs(p_back - pa[i + 15]).max() * 1e6:.3f} um, '
      f'rotation {hand.rotation_error_deg(R_back, Ra[i + 15]).max():.2e} deg; '
      f'relative move 15 steps ahead: median {np.median(np.linalg.norm(rel[:, :3], axis=1)) * 100:.1f} cm, '
      f'max {np.linalg.norm(rel[:, :3], axis=1).max() * 100:.1f} cm')

# 4. one IK step from the recorded state toward the hand pose of its recorded joint target
rng = np.random.default_rng(0)
sample = rng.choice(len(state), 2000, replace=False)
model = env.mujoco[0]
import mujoco

data = mujoco.MjData(model)
ik = DifferentialIK(model, data, site_name=hand.SITE, qpos_idx=env.arm_qpos_idx, damping=0.15, max_joint_step=0.06)
pos_err, rot_err, joint_err = [], [], []
for j in sample:
    data.qpos[env.arm_qpos_idx] = state[j, :7]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    q_ik = ik.solve(pa[j], Ra[j])
    p_ik, R_ik = kin.pose(q_ik[None])
    pos_err.append(np.linalg.norm(p_ik[0] - pa[j]))
    rot_err.append(hand.rotation_error_deg(R_ik[0], Ra[j]))
    joint_err.append(np.abs(q_ik - action[j, :7]).max())
pos_err, rot_err, joint_err = map(np.array, (pos_err, rot_err, joint_err))
print(f'4. one IK step to the recorded target\'s hand pose (2000 samples): position median '
      f'{np.median(pos_err) * 1000:.2f} mm, 95% {np.percentile(pos_err, 95) * 1000:.2f} mm; rotation median '
      f'{np.median(rot_err):.3f} deg, 95% {np.percentile(rot_err, 95):.3f} deg; joints vs recorded: median '
      f'{np.degrees(np.median(joint_err)):.3f} deg, 95% {np.degrees(np.percentile(joint_err, 95)):.3f} deg')
print('------ FINISHED ------')
