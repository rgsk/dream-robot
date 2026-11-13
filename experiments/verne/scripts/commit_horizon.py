# When does the arm visibly commit to a cube? Real T5 demo frames (top camera, decoded/top.npy), held-out
# episodes, at steps 0, 16, 32, 48, 64, 80 from the episode start; each row one episode, named colour + slot in
# commit_horizon.txt. -> experiments/verne/runs/10_future_video/commit_horizon.png
import json

import imageio.v3 as iio
import numpy as np

from dream_robot.policies.verne.data import REPO, load_cameras, load_table
from dream_robot.policies.verne.run import split

name = 'instructed_pick_place_square_noise005_n150'
top = load_cameras(name)[0]
episode = load_table(name)[2]
kept = [a for a in json.loads((REPO / 'data/robosuite' / name / 'instructions.json').read_text())['attempts'] if a['kept']]
val = split('t5', 150)[1]
offsets = (0, 16, 32, 48, 64, 80)
rows, lines = [], []
for ep in val[:8]:
    first, last = np.searchsorted(episode, ep), np.searchsorted(episode, ep, side='right') - 1
    rows.append(np.concatenate([top[min(first + o, last)].transpose(1, 2, 0) for o in offsets], 1))
    a = kept[ep]
    close = first + np.argmax(load_table(name)[1][first : last + 1, 7] < 0.5)
    lines.append(f'ep {ep}: {a["prompt"]}  named slot {a["target_slot"]}  first close at step {close - first}')
out = REPO / 'experiments/verne/runs/10_future_video'
iio.imwrite(out / 'commit_horizon.png', np.concatenate(rows, 0))
(out / 'commit_horizon.txt').write_text('columns: steps ' + str(offsets) + '\n' + '\n'.join(lines) + '\n')
print('\n'.join(lines))
