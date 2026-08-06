"""Differential inverse kinematics for MuJoCo-backed robosuite tasks.

Shared by every robosuite task, not owned by one of them. It exists so scripted
experts can plan where it is natural to think -- end-effector space -- while the
dataset still records what spec.md requires: absolute joint targets.

Generating an action and recording an action are different jobs. This is the
converter between them, and it lives on the sim side of the seam.
"""

from __future__ import annotations

import mujoco
import numpy as np


class DifferentialIK:
    '''Damped least-squares IK against a MuJoCo site, one control step at a time.

    Let e be the 6-D task-space error between where the end effector is and
    where we want it,

        e = [ x_des − x_now ; ω(R_des R_nowᵀ) ]

    where ω(·) is the rotation-vector (axis times angle) of the residual
    rotation. With J the 6 x n site Jacobian, the joint step that best satisfies
    J·dq = e is the pseudo-inverse solution dq = J⁺e. That blows up near a
    singularity, where J loses rank, so use the damped (Levenberg-Marquardt)
    form instead — the minimiser of

        ‖J·dq − e‖²  +  λ²‖dq‖²

    which has the closed form

        dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e

    λ trades tracking accuracy for stability: λ→0 recovers the exact
    pseudo-inverse and its singularity blow-up, larger λ gives smaller, safer
    steps that converge more slowly.

    dq is then clipped per joint. That clip is the smoothness knob — it bounds
    joint velocity to max_joint_step x control_hz rad/s, and keeps the commanded
    target inside what the position controller can actually follow in one step.
    '''

    def __init__(
        self,
        model,
        data,
        *,
        site_name: str,
        qpos_idx,
        damping: float,
        max_joint_step: float,
    ):
        self._model = model
        self._data = data
        self._site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self._site_id < 0:
            raise ValueError(f"no site named {site_name!r} in this model")
        self._qpos_idx = np.asarray(qpos_idx)
        # Jacobian columns are indexed by DOF, not by qpos. They coincide for
        # hinge joints but not in general: a free-floating object is 7 qpos and
        # 6 dof, so anything declared after one would be silently misaligned.
        qpos_adr = model.jnt_qposadr.tolist()
        self._dof_idx = np.array([model.jnt_dofadr[qpos_adr.index(i)] for i in self._qpos_idx])
        self._damping = float(damping)
        self._max_joint_step = float(max_joint_step)

    @property
    def site_pos(self) -> np.ndarray:
        return np.asarray(self._data.site_xpos[self._site_id]).copy()

    @property
    def site_mat(self) -> np.ndarray:
        return np.asarray(self._data.site_xmat[self._site_id]).reshape(3, 3).copy()

    def solve(self, x_des, R_des) -> np.ndarray:
        """Absolute joint targets that move the site one step toward the goal."""
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._site_id)
        J = np.vstack([jacp[:, self._dof_idx], jacr[:, self._dof_idx]])

        quat = np.zeros(4)
        omega = np.zeros(3)
        mujoco.mju_mat2Quat(quat, (np.asarray(R_des) @ self.site_mat.T).flatten())
        mujoco.mju_quat2Vel(omega, quat, 1.0)

        err = np.concatenate([np.asarray(x_des, dtype=float) - self.site_pos, omega])
        dq = J.T @ np.linalg.solve(J @ J.T + self._damping**2 * np.eye(6), err)

        q_now = np.asarray(self._data.qpos[self._qpos_idx], dtype=float)
        return q_now + np.clip(dq, -self._max_joint_step, self._max_joint_step)
