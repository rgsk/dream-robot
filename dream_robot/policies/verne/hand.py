"""Step 20 (roadmap box 06): the hand's pose from joint angles (forward kinematics), and actions as hand moves
relative to the hand.

Verne (§10, §12) predicts gripper poses relative to the current hand, plus progress; IK outside the model turns
poses into joints. The T5 demos record absolute joint targets, so the targets are rebuilt here: the hand pose of
every recorded state and of every recorded joint target, by MuJoCo forward kinematics on the T5 arm (the
expert's grip site), cached once per dataset.

A chunk's targets are relative to the hand at the chunk's start, (p₀, R₀):

    position   R₀ᵀ (p_k − p₀)                  the move in the hand's own frame, metres
    rotation   R₀ᵀ R_k, first two columns      6 numbers (Zhou et al.: continuous, unlike Euler angles or quaternions)
    jaw        the recorded gripper command
    progress   step / (episode length − 1)     0 at the start, 1 at the end

Back to the world at run time: p_k = p₀ + R₀ · position,  R_k = R₀ · rotation (the 6 numbers completed to a
rotation by Gram-Schmidt and a cross product), then IK (sims/robosuite/ik.py) gives the joint targets.
"""

import mujoco
import numpy as np
from numpy.typing import NDArray

SITE = 'gripper0_right_grip_site'  # the expert's grip site (pick_place_cube/expert.py EEF_SITE)


class Kinematics:
    # forward kinematics of the T5 arm on a private MjData (never touches a running env's state)
    def __init__(self, model, arm_qpos_idx: NDArray):
        self.model, self.data = model, mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE)
        assert self.site >= 0, SITE
        self.idx = np.asarray(arm_qpos_idx)

    def pose(self, joints: NDArray) -> tuple[NDArray, NDArray]:
        # joints [N, 7] -> hand position [N, 3] (world, metres), rotation [N, 3, 3]
        pos, rot = np.empty((len(joints), 3)), np.empty((len(joints), 3, 3))
        for i, q in enumerate(joints):
            self.data.qpos[self.idx] = q
            mujoco.mj_kinematics(self.model, self.data)
            pos[i] = self.data.site_xpos[self.site]
            rot[i] = self.data.site_xmat[self.site].reshape(3, 3)
        return pos, rot


def t5_kinematics() -> Kinematics:
    # a T5 env, only for its arm model
    from .rollouts import make_env

    env = make_env('t5')
    env.reset(seed=0)
    model, _ = env.mujoco
    return Kinematics(model, env.arm_qpos_idx)


def relative(p0: NDArray, R0: NDArray, p: NDArray, R: NDArray) -> NDArray:
    # reference (p0 [..., 3], R0 [..., 3, 3]) and targets (p, R) -> [..., 9]: position in the reference's frame,
    # then the relative rotation's first two columns
    R0t = np.swapaxes(R0, -1, -2)
    pos = np.einsum('...ij,...j->...i', R0t, p - p0)
    rel = R0t @ R
    return np.concatenate([pos, rel[..., :, 0], rel[..., :, 1]], -1)


def absolute(p0: NDArray, R0: NDArray, rel9: NDArray) -> tuple[NDArray, NDArray]:
    # the inverse of relative: [..., 9] -> world position [..., 3], rotation [..., 3, 3]
    a, b = rel9[..., 3:6], rel9[..., 6:9]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    rel = np.stack([a, b, np.cross(a, b)], -1)
    p = p0 + np.einsum('...ij,...j->...i', R0, rel9[..., :3])
    return p, R0 @ rel


def rotation_error_deg(R1: NDArray, R2: NDArray) -> NDArray:
    # [..., 3, 3] x 2 -> the angle of R1ᵀ R2, degrees
    tr = np.einsum('...ii->...', np.swapaxes(R1, -1, -2) @ R2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def relative_t(p0, R0, p, R):
    # torch twin of relative (training batches on the GPU)
    import torch

    R0t = R0.transpose(-1, -2)
    pos = (R0t @ (p - p0)[..., None])[..., 0]
    rel = R0t @ R
    return torch.cat([pos, rel[..., :, 0], rel[..., :, 1]], -1)


class Driver:
    # run time: a predicted chunk of hand moves (raw: 9 relative + jaw + progress) -> joint actions. The chunk is
    # turned into world poses once, from the hand at its start; each executed step then solves IK from the arm's
    # current joints to that pose (10 damped steps on a private MjData: 0.00 mm in hand_check). Build after every
    # env.reset (robosuite rebuilds the model)
    def __init__(self, env, iters: int = 10, damping: float = 0.15, max_joint_step: float = 0.06):
        from dream_robot.sims.robosuite.ik import DifferentialIK

        model, _ = env.mujoco
        self.kin = Kinematics(model, env.arm_qpos_idx)
        self.data = mujoco.MjData(model)
        self.ik = DifferentialIK(
            model,
            self.data,
            site_name=SITE,
            qpos_idx=env.arm_qpos_idx,
            damping=damping,
            max_joint_step=max_joint_step,
        )
        self.idx, self.iters = np.asarray(env.arm_qpos_idx), iters

    def plan(self, q_now: NDArray, chunk: NDArray) -> list:
        # q_now [7], chunk [n, 11] raw -> n targets (position [3], rotation [3, 3], jaw)
        p0, R0 = self.kin.pose(q_now[None])
        p, R = absolute(p0, R0, chunk[:, :9])
        jaw = np.clip(chunk[:, 9], 0.0, 1.0)
        return list(zip(p, R, jaw, strict=True))

    def action(self, q_now: NDArray, target) -> NDArray:
        # the arm's joints now [7], one target -> the env action [8]: joint targets + gripper command
        p, R, jaw = target
        self.data.qpos[self.idx] = q_now
        for _ in range(self.iters):
            mujoco.mj_kinematics(self.ik._model, self.data)
            mujoco.mj_comPos(self.ik._model, self.data)
            self.data.qpos[self.idx] = self.ik.solve(p, R)
        return np.concatenate([self.data.qpos[self.idx], [jaw]])
