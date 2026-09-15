"""Step-1 style diagnosis for any BC checkpoint: arrival xy error at grasp height, freeze, gripper closure.
    python experiments/noisy_expert/scripts/diagnose_bc.py <checkpoint.pt>"""
import os, sys, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np, mujoco
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import EEF_SITE
from dream_robot.policies.bc.policy import BCPolicy
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

LOW = 3.0
env = PickPlaceCube(); bc = BCPolicy.load(sys.argv[1])
kinds, arrivals = {}, []
for seed in range(1000, 1020):
    obs = env.reset(seed=seed); bc.reset(); rows, Q, cmd = [], [], []
    m, d = env.mujoco; sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    while True:
        g = d.site_xpos[sid] - env.cube_pos
        rows.append((np.linalg.norm(g[:2]) * 100, g[2] * 100, obs.state[7])); Q.append(obs.state[:7].copy())
        a = bc(obs); cmd.append(a[7])
        r = env.step(a); obs = r.observation
        if r.terminated or r.truncated: break
    rows, Q, cmd = np.array(rows), np.array(Q), np.array(cmd)
    low = np.nonzero(rows[:, 1] < LOW)[0]
    if len(low) == 0:
        kind, xy, a = "never_reached_grasp_height", np.nan, None
        print(f"seed {seed}  {kind}  lowest dz {rows[:,1].min():.1f} cm  min gripper cmd {cmd.min():.2f}", flush=True)
    else:
        a = int(low[0]); xy = rows[a, 0]
        moving = np.nonzero(np.abs(np.diff(Q, axis=0)).max(1) > 1e-3)[0]; last = int(moving[-1]) if len(moving) else 0
        if r.success: kind = "success"
        elif rows[a:, 2].min() < 0.1: kind = "closed_on_air"
        elif rows[a:, 2].min() < 0.9: kind = "closed_then_lost"
        elif last < len(Q) - 100: kind = "frozen_open"
        else: kind = "hovering_open"
        print(f"seed {seed}  {kind:16} arrival xy {xy:5.2f} cm at t={a:3}  last motion t={last:3}  "
              f"min gripper cmd after arrival {cmd[a:].min():.2f}", flush=True)
    kinds[kind] = kinds.get(kind, 0) + 1; arrivals.append(xy)
print(f"\nkinds {kinds}\narrival xy median {np.nanmedian(arrivals):.2f} cm, range {np.nanmin(arrivals):.2f}..{np.nanmax(arrivals):.2f}")
env.close()
