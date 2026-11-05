"""Raw rung 0 episodes (rung0_record.py) -> a LeRobot v3.0 dataset, in the main venv.

Features: core/dataset.py's (observation.state, action, observation.images.front / .wrist)
plus observation.touch [8]. Policies that don't know touch simply never read that column.

    .venv/bin/python experiments/pick_shapes/scripts/rung0_to_lerobot.py data/robocasa/pick_apple_rung0_n3_raw
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dream_robot.core.dataset import ACTION_FEATURE, STATE_FEATURE, dataset_features, image_feature
from dream_robot.core.schema import PANDA

TOUCH_FEATURE = 'observation.touch'


def main(raw: Path) -> None:
    meta = json.loads((raw / 'recording_log.json').read_text())
    root = raw.with_name(raw.name.removesuffix('_raw'))
    if root.exists():
        shutil.rmtree(root)
    episodes = sorted(raw.glob('ep*.npz'))
    first = np.load(episodes[0])
    cams = meta['cameras']
    features = dataset_features(embodiment=PANDA, cameras=cams, image_hw=first[cams[0]].shape[1:3])
    features[TOUCH_FEATURE] = {'dtype': 'float32', 'shape': (8,), 'names': meta['touch_names']}
    ds = LeRobotDataset.create(repo_id=f'dream_robot/{root.name}', fps=int(meta['fps']),
                               features=features, root=root, robot_type=meta['robot_type'],
                               use_videos=True)
    for path in episodes:
        ep = np.load(path)
        task = str(ep['task'])
        for t in range(len(ep['action'])):
            frame = {STATE_FEATURE: ep['state'][t], ACTION_FEATURE: ep['action'][t],
                     TOUCH_FEATURE: ep['touch'][t], 'task': task}
            for cam in cams:
                frame[image_feature(cam)] = ep[cam][t]
            ds.add_frame(frame)
        ds.save_episode(parallel_encoding=False)
    ds.finalize()
    print(f'{len(episodes)} episodes -> {root}')


if __name__ == '__main__':
    main(Path(sys.argv[1]))
