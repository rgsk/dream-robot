# Verne's canvas VAE ([top | wrist] 3 x 128 x 256 -> 4 x 16 x 32) on LIBERO frames (the T5 one has only seen the
# tabletop cubes). Cosine lr after a 250-step warmup, peak 5e-4, grad clip 1.0 (every T1/T5 WSD run spiked in its
# constant-lr phase); checkpoints every 1000 steps (ckpt/step<N>.pt) to rewind past a spike.
# Data: data/libero/npz/*.npz, every 2nd frame (20 Hz neighbours are near copies); 10% of each task's demos held
# out (episodes 0-4) for val loss and recon.png.
#   python experiments/libero/scripts/train_vae.py experiments/libero/runs/vae [steps]   (main venv)
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from dream_robot.policies.verne import vae
from dream_robot.policies.verne.data import canvas

REPO = Path(__file__).resolve().parents[3]
VAL_EPISODES = 5  # of 50 per task


def frames() -> tuple[np.ndarray, np.ndarray]:
    train, val = [], []
    for p in sorted((REPO / 'data/libero/npz').glob('*.npz')):
        z = np.load(p)
        ep = z['episode']
        c = canvas(z['top'].transpose(0, 3, 1, 2), z['wrist'].transpose(0, 3, 1, 2))  # [N, 3, 128, 256]
        train.append(c[np.flatnonzero(ep >= VAL_EPISODES)[::2]])
        val.append(c[np.flatnonzero(ep < VAL_EPISODES)[::20]])
        print(f'{p.stem}: {len(train[-1])} train, {len(val[-1])} val', flush=True)
    return np.concatenate(train), np.concatenate(val)


def main(out: Path, steps: int = 10000) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'weights.pt').exists():
        sys.exit(f'{out} already has weights')
    (out / 'meta.json').write_text(json.dumps({
        'gpu': torch.cuda.get_device_name(), 'steps': steps, 'schedule': 'cosine', 'lr': 5e-4, 'warmup': 250,
        'every': 2, 'val_episodes': VAL_EPISODES, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
    train, val = frames()
    print(f'{len(train)} training frames, {len(val)} val frames', flush=True)
    run = vae.train(train, val, steps=steps, ckpt=out / 'ckpt', schedule='cosine')
    (out / 'log.json').write_text(json.dumps(run['log']))
    torch.save(run['weights'], out / 'weights.pt')
    net = vae.VAE().cuda().eval()
    net.load_state_dict(run['weights'])
    pick = val[np.linspace(0, len(val) - 1, 6).astype(int)]
    rec = vae.reconstruct(net, pick)
    grid = np.concatenate([np.concatenate(list(pick), 2), np.concatenate(list(rec), 2)], 1)
    iio.imwrite(out / 'recon.png', grid.transpose(1, 2, 0))
    print(f'done: {out}\n------ FINISHED ------', flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 10000)
