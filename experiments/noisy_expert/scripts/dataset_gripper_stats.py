"""Compare clean vs shaky datasets on what the recorded labels teach about the gripper."""
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

for name, root in [("clean", "data/robosuite/pick_place_cube"), ("shaky", "data/robosuite/pick_place_cube_noise005")]:
    ds = LeRobotDataset("dream_robot/pick_place_cube", root=root)
    a = np.asarray(ds.hf_dataset["action"], dtype=np.float32)
    s = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float32)
    ep = np.asarray(ds.hf_dataset["episode_index"])
    n_ep = len(np.unique(ep))
    closed = a[:, 7] < 0.5
    # "open command while arm is nearly still" = hovering frames
    still = np.r_[np.abs(np.diff(s[:, :7], axis=0)).max(1) < 2e-3, False]
    steps_before_close = []
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]; c = np.nonzero(closed[idx])[0]
        steps_before_close.append(c[0] if len(c) else len(idx))
    print(f"{name}: {n_ep} eps, {len(a)} frames | close-command frames {closed.mean():.1%} | "
          f"open & arm still frames {np.mean(still & ~closed):.1%} | steps until first close: median {np.median(steps_before_close):.0f}, "
          f"range {min(steps_before_close)}..{max(steps_before_close)} | arm action std {a[:, :7].std(0).mean():.3f}")
