# rough/v0.py
import imageio.v3 as iio
import numpy as np

from dream_robot.sims.robosuite.tasks.pick_place_cube.env import PickPlaceCube
from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import ExpertPolicy

env = PickPlaceCube()
policy = ExpertPolicy(env)

obs = env.reset(seed=0)
policy.reset()

def upscale_nearest(frame: np.ndarray, factor: int) -> np.ndarray:
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if factor == 1:
        return frame
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)

frames = []
for _ in range(600):
    action, command = policy(obs)
    wide = env.render()
    order = ("top", "wrist")
    present = [name for name in order if name in obs.images]
    if not present:
        raise ValueError(f"none of {list(order)} in images {sorted(obs.images)}")
    if wide.shape[0] % len(present):
        raise ValueError(
            f"render height {wide.shape[0]}px is not divisible by {len(present)} cameras "
            f"{present}; each panel would be {wide.shape[0] / len(present)}px. "
            f"Set render.height in task.yaml to a multiple of {len(present)}."
        )

    target_image_height = wide.shape[0] // len(present)
    column = []
    for image_name in present:
        image = obs.images[image_name]
        if target_image_height % image.shape[0]:
            raise ValueError(
                f"camera {image_name} is {image.shape[0]}px tall, which does not divide the "
                f"{target_image_height}px panel height. Integer scaling only -- see upscale_nearest."
            )
        column.append(upscale_nearest(image, target_image_height // image.shape[0]))

    stacked = np.concatenate(column, axis=0)
    panel = np.concatenate([wide, stacked], axis=1)
    frames.append(panel)
    result = env.step(action)
    obs = result.observation
    if result.success:
        break

iio.imwrite("rough/v0.mp4", np.stack(frames), fps=30, codec="libx264")
print(len(frames), "frames")
