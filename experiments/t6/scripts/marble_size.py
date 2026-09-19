"""Do smaller marbles make the spill a real failure signal?

    MUJOCO_GL=egl uv run python experiments/t6/scripts/marble_size.py [r_mm] [u]

Step 2b found the marble count flagged none of the unsynchronised expert's bad
lifts -- including a tray hanging at 71 degrees with all 9 aboard. The review
question was whether that is a size effect: nine 18 mm balls span the 12 cm
cavity only 3.3 diameters across, which is the arching regime, and smaller ones
would flow instead of wedging.

The static rig (scripts/t6_spill_curve.py scale) says yes: at a FIXED lip ratio
the escape angle is scale free in theory and falls ~10 deg per halving in
practice. This asks the same question of the real scene and the real expert.

**Mass is scaled with r^3 on purpose.** Keeping the per-marble mass fixed while
shrinking the marbles is the confound: 16 x 20 g is a 0.32 kg payload against
today's 9 x 20 g = 0.18 kg, so the arms would be lifting nearly twice the load
and any change in the result could be that instead of the spill. Constant
density keeps the comparison about geometry.
"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")

from dataclasses import replace

from dream_robot.sims.robosuite.tasks.two_arm_lift.env import (
    MarbleSpec, TaskConfig, TwoArmLiftTask,
)
from dream_robot.sims.robosuite.tasks.two_arm_lift.expert import ExpertConfig, ExpertPolicy

BASE_R, BASE_MASS, BASE_N = 0.018, 0.02, 9
HOLD_SECONDS = 1.0              # overridable: argv[3]
GIVE_UP = 260
# The eleven seeds whose unsynchronised lift reached the height with 18 mm
# marbles. Rerunning only these skips nine 600-step timeouts.
LIFTED_WITHOUT_BARRIER = (1000, 1004, 1005, 1010, 1011, 1012, 1013, 1015, 1016, 1017, 1019)


def main():
    radius = float(sys.argv[1]) / 1000.0 if len(sys.argv) > 1 else 0.012
    u = float(sys.argv[2]) if len(sys.argv) > 2 else 0.85
    hold_steps = int(round((float(sys.argv[3]) if len(sys.argv) > 3 else HOLD_SECONDS) * 30))
    # Lattice columns the env will actually fit, so the count never spills into
    # a second layer -- on a 10 mm lip a stacked layer rolls off at rest.
    inner = 0.06
    per_row = max(1, int((2 * (inner - radius)) // (2.1 * radius)) + 1)
    count = per_row * per_row
    mass = BASE_MASS * (radius / BASE_R) ** 3

    base = TaskConfig.load()
    cfg = replace(
        base,
        marbles=MarbleSpec(count=count, radius=radius, mass=mass, keep_fraction=0.75),
        tray=replace(base.tray, lip_over_radius=u),
    )
    env = TwoArmLiftTask(cfg)
    print(f"r={radius*1000:.0f} mm  n={count} ({per_row}x{per_row})  lip={u}r="
          f"{u*radius*1000:.1f} mm  mass={mass*1000:.1f} g each, "
          f"{count*mass*1000:.0f} g payload (today: {BASE_N*BASE_MASS*1000:.0f} g)")
    print(f"need {cfg.marbles.required_inside} of {count} aboard, "
          f"held {hold_steps / 30:.1f} s past the height")

    for barrier, seeds in ((False, LIFTED_WITHOUT_BARRIER), (True, range(1000, 1020))):
        ecfg = ExpertConfig.load()
        if not barrier:
            ecfg = replace(ecfg, sync_barrier=False)
        policy = ExpertPolicy(env, ecfg)
        print(f"\n--- barrier={barrier}")
        print(f"{'seed':>5} {'tilt':>6} {'aboard':>7} {'height':>7} {'verdict':>8}")
        passed = 0
        for seed in seeds:
            obs = env.reset(seed=seed)
            policy.reset()
            lifted = None
            for t in range(env._cfg.horizon_steps):
                action, _ = policy(obs)
                result = env.step(action)
                obs = result.observation
                if lifted is None and env._pot_lift() > cfg.lift_height:
                    lifted = t
                if lifted is not None and t - lifted >= hold_steps:
                    break
                if lifted is None and t > GIVE_UP:
                    break
            if lifted is None:
                print(f"{seed:>5} {'-':>6} {'-':>7} {'-':>7} {'no lift':>8}")
                continue
            aboard, tilt, height = env.marbles_inside, env._pot_tilt_deg(), env._pot_lift()
            good = height > cfg.lift_height and aboard >= cfg.marbles.required_inside
            passed += good
            print(f"{seed:>5} {tilt:>6.0f} {aboard:>7} {height:>7.3f} "
                  f"{'PASS' if good else 'FAIL':>8}")
        print(f"{passed}/{len(list(seeds))} pass")
    env.close()


if __name__ == "__main__":
    main()
