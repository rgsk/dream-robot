"""What does a handle grasp require geometrically, that a cube grasp did not?

    MUJOCO_GL=egl uv run python experiments/t6/scripts/grasp_geometry.py

T1's expert held the gripper's reset rotation for the whole episode, because a
cube is symmetric about z and any yaw grasps it. A handle bar is not: the
fingers have to close ACROSS it, and the tray's yaw is the scene's random
variable. Before writing a phase machine, measure:

1. which axis of the grip site the fingers close along (a model constant),
2. how the bar is oriented in world, per seed, and how far the yaw wanders,
3. whether handle0 always ends up on arm0's side of the table.

NOTE on reading the simulator: robosuite runs with ``hard_reset=True``, so
``env._env.sim`` is a DIFFERENT object after every reset. Caching it gives you
the construction-time scene forever -- which reads as "the tray never moves",
and cost an hour the first time.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from dream_robot.sims.robosuite.tasks.two_arm_lift.env import (
    EEF_SITES, HANDLE_SITES, N_ARMS, TwoArmLiftTask,
)

np.set_printoptions(precision=3, suppress=True)
SEEDS = range(1000, 1012)


def main():
    env = TwoArmLiftTask()
    rows = []
    for seed in SEEDS:
        env.reset(seed=seed)
        sim = env._env.sim                      # fresh handle, every reset
        tray_rot = np.array(sim.data.body_xmat[env._env.pot_body_id]).reshape(3, 3)
        yaw = np.degrees(np.arctan2(tray_rot[1, 0], tray_rot[0, 0]))
        centre = env.pot_center
        handles = env.handle_pos
        row = {"seed": seed, "yaw": yaw}
        for arm in range(N_ARMS):
            sid = sim.model.site_name2id(EEF_SITES[arm])
            grip = np.array(sim.data.site_xpos[sid], dtype=float)
            R = np.array(sim.data.site_xmat[sid]).reshape(3, 3)
            lf = np.array(sim.data.body_xpos[
                sim.model.body_name2id(f"gripper{arm}_right_leftfinger")])
            rf = np.array(sim.data.body_xpos[
                sim.model.body_name2id(f"gripper{arm}_right_rightfinger")])
            close_axis = (rf - lf) / np.linalg.norm(rf - lf)
            base = np.array(sim.data.body_xpos[
                sim.model.body_name2id(f"robot{arm}_base")], dtype=float)
            out = handles[arm] - centre                     # tray centre -> handle
            out = out / np.linalg.norm(out)
            row[f"a{arm}"] = dict(
                dist=np.linalg.norm(handles[arm] - grip),
                reach=np.linalg.norm(handles[arm] - base),
                out=out,
                close_in_site=R.T @ close_axis,
                # Sign convention check: is handle i on arm i's side of y = 0?
                side=np.sign(handles[arm][1] - centre[1]),
            )
        rows.append(row)

    print(f"{'seed':>5} {'yaw':>7} | {'h0 reach':>8} {'h0 side':>7} "
          f"| {'h1 reach':>8} {'h1 side':>7} | handle0 outward dir")
    for r in rows:
        print(f"{r['seed']:>5} {r['yaw']:>+7.1f} | {r['a0']['dist']:>8.3f} "
              f"{r['a0']['side']:>+7.0f} | {r['a1']['dist']:>8.3f} "
              f"{r['a1']['side']:>+7.0f} | {r['a0']['out']}")

    yaws = np.array([r["yaw"] for r in rows])
    print(f"\nyaw over {len(rows)} seeds: {yaws.min():+.1f} .. {yaws.max():+.1f} deg")
    print("finger closing axis in grip-site frame (arm0, arm1):")
    print("  ", rows[0]["a0"]["close_in_site"], rows[0]["a1"]["close_in_site"])
    print("reach from own base: arm0 %.3f..%.3f  arm1 %.3f..%.3f" % (
        min(r["a0"]["reach"] for r in rows), max(r["a0"]["reach"] for r in rows),
        min(r["a1"]["reach"] for r in rows), max(r["a1"]["reach"] for r in rows)))
    env.close()


if __name__ == "__main__":
    main()
