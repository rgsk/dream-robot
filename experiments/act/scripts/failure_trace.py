"""Replay ACT failures and trace what the cube actually did, to check the failure label.
    python experiments/act/scripts/failure_trace.py <checkpoint> <seed> [<seed> ...]"""
import os, sys, logging
os.environ["MUJOCO_GL"] = "egl"
logging.disable(logging.WARNING)
import numpy as np, mujoco
from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import EEF_SITE
from dream_robot.policies.act.policy import ACTPolicy
logging.disable(logging.NOTSET); logging.getLogger("robosuite_logs").setLevel(logging.ERROR)

env = PickPlaceCube(); policy = ACTPolicy.load(sys.argv[1])
for seed in map(int, sys.argv[2:]):
    obs = env.reset(seed=seed); policy.reset()
    m, d = env.mujoco; sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    z0 = env.cube_pos[2]; bin_c = env.bin_center; rows = []
    while True:
        cube = env.cube_pos
        rows.append((np.linalg.norm(d.site_xpos[sid] - cube) * 100, (cube[2] - z0) * 100,
                     np.linalg.norm(cube[:2] - bin_c[:2]) * 100, obs.state[7]))
        r = env.step(policy(obs)); obs = r.observation
        if r.terminated or r.truncated: break
    rows = np.array(rows); t_peak = int(rows[:, 1].argmax())
    held = rows[:, 0] < 3.0                                  # gripper within 3 cm of cube centre
    print(f"seed {seed}: success={r.success} label={r.failure_mode} steps={len(rows)}")
    print(f"  cube max rise {rows[:, 1].max():.2f} cm at t={t_peak} (label rule: >3 cm = 'lifted')")
    print(f"  steps gripper within 3 cm of cube: {held.sum()} | cube start->bin {rows[0, 2]:.1f} cm, "
          f"closest to bin {rows[:, 2].min():.1f} cm, final {rows[-1, 2]:.1f} cm (label rule: >15 cm = dropped)")
    for t in range(0, len(rows), 40):
        g, z, b, o = rows[t]
        print(f"    t={t:3} grip-cube {g:5.1f} cm | cube rise {z:+5.2f} cm | cube-bin {b:5.1f} cm | opening {o:.2f}")
env.close()
