"""Film a sampled policy several times from ONE identical start, varying only its own noise seed.

    MUJOCO_GL=egl uv run python experiments/t2_fm/scripts/film_samples.py \
        --checkpoint experiments/t2_fixed_fm_half100/checkpoint.pt --start 1010 --policy-seeds 0 1 2 \
        --out experiments/t2_fm/videos/fm_start1010_seeds012.mp4

Episodes play back to back in one video. core.run cannot do this: it ties one env seed to each episode.
"""

import argparse
from pathlib import Path

from dream_robot.core.registry import make_env, make_policy
from dream_robot.core.video import observation_panel, write_video

p = argparse.ArgumentParser()
p.add_argument("--env", default="robosuite/pick_place_two_bins_fixed")
p.add_argument("--policy", default="flow_matching")
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--start", type=int, default=1010)
p.add_argument("--policy-seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()

env = make_env(args.env)
policy = make_policy(args.policy, env, checkpoint=args.checkpoint)
frames = []
for k in args.policy_seeds:
    obs = env.reset(seed=args.start)
    policy.reset()
    policy.seed(k)
    for _ in range(env._cfg.horizon_steps):
        frames.append(observation_panel(env.render(), obs.images))
        result = env.step(policy(obs))
        obs = result.observation
        if result.terminated or result.truncated:
            break
    placed = env.which_bin(env.cube_pos)
    where = env.bin_names[placed] if placed is not None else "not in a bin"
    print(f"start {args.start}, policy seed {k}: {where}, {len(frames)} frames so far")
env.close()
print(f"-> {write_video(frames, args.out, fps=env.control_hz)}")
