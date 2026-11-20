# Every frame position pretraining will see, with every label drawn: one film per task, all its demos back to
# back, top | wrist at 2x. Colours: stove_knob yellow, stove (cook region) blue, moka_pot red, frying_pan green;
# a dot only where that name is in view in that camera. Demo number in the corner (a white bar per 10 demos).
#   python experiments/libero/scripts/film_labels.py data/libero/npz/*.npz   (either venv: numpy + imageio)
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

OUT = Path('scratch_nov/libero/pretrain_frames')
COLOURS = {'stove_knob': (255, 220, 0), 'stove': (0, 80, 255), 'moka_pot': (255, 0, 0), 'frying_pan': (0, 200, 0)}


def main(paths: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in map(Path, paths):
        z = np.load(p)
        names = [str(n) for n in z['names']]
        top, wrist, pix, seen, ep = z['top'], z['wrist'], z['pix'], z['seen'], z['episode']
        frames = np.concatenate([top, wrist], axis=2)  # [N, 128, 256, 3]
        for j, n in enumerate(names):
            if n not in COLOURS:
                continue
            for k in range(2):
                rows = np.flatnonzero(seen[:, j, k])
                r = np.round(pix[rows, j, k, 0]).astype(int)
                c = np.round(pix[rows, j, k, 1]).astype(int) + 128 * k
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        rr, cc = np.clip(r + dr, 0, 127), np.clip(c + dc, 128 * k, 128 * k + 127)
                        frames[rows, rr, cc] = COLOURS[n]
        for i in range(ep.max() + 1):  # demo counter: one 3 px white tick per demo, top-left
            frames[ep == i, 0:3, 0:min(3 * (i + 1), 120)] = 255
        big = frames.repeat(2, axis=1).repeat(2, axis=2)
        iio.imwrite(OUT / f'{p.stem}_all_demos.mp4', big, fps=40)  # 2x speed: ~2-4 min per task
        print(p.stem, len(frames), 'frames', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
