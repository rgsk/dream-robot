"""T2 step 1: does the scene render both bins, and can the expert reach each one?

    MUJOCO_GL=glfw uv run python experiments/t2/scripts/scene_check.py

Writes scratch/t2/scene_seed0.png (front render | top camera | wrist camera) and
runs the expert on seeds 0-5 with the coin forced to each bin in turn.
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_two_bins.env import PickPlaceTwoBins
from dream_robot.sims.robosuite.tasks.pick_place_two_bins.expert import ExpertPolicy

OUT = Path("scratch/t2")


def up(img, k=2):
    return np.kron(img, np.ones((k, k, 1), dtype=np.uint8))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = PickPlaceTwoBins()
    expert = ExpertPolicy(env)
    obs = env.reset(seed=0)
    imageio.imwrite(
        OUT / "scene_seed0.png",
        np.concatenate([env.render(), up(obs.images["top"]), up(obs.images["wrist"])], axis=1),
    )
    print("bins:", dict(zip(env.bin_names, (c.round(3).tolist() for c in env.bin_centers))))

    for seed in range(6):
        forced = seed % 2
        obs = env.reset(seed=seed)
        expert.reset()
        coin = expert.target_bin
        expert.target_bin = forced
        for t in range(env._cfg.horizon_steps):
            action, _ = expert(obs)
            r = env.step(action)
            obs = r.observation
            if r.terminated or r.truncated:
                break
        placed = env.which_bin(env.cube_pos)
        print(
            f"seed {seed}: coin={env.bin_names[coin]} forced={env.bin_names[forced]} "
            f"success={r.success} in_bin={None if placed is None else env.bin_names[placed]} "
            f"steps={t + 1} failure={r.failure_mode}"
        )
    env.close()


if __name__ == "__main__":
    main()
