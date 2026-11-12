# Step 04: the close-point probe (n_exec 8) for chosen sampler arms on step 03's T1-wide policies ->
# experiments/verne/runs/04_sampler/<arm>/t1-wide/seed<k>/probe.json (skipped if there).
# usage: python experiments/verne/scripts/sampler_probe.py 10:1.0 4:1.5
import sys

import torch

from dream_robot.policies.verne import rollouts, vae
from dream_robot.policies.verne.data import load_table, to_gpu
from dream_robot.policies.verne.run import RUNS, TASKS, split, write_json

if __name__ == '__main__':  # spawned workers import this file; only the parent runs it
    arms = [(int(a.split(':')[0]), float(a.split(':')[1])) for a in sys.argv[1:]]
    name, wide = TASKS['t1-wide']
    state, action, episode = load_table(name)
    d = to_gpu(vae.load_latents(name, vae.WEIGHTS), state, action, episode, split('t1-wide')[0])
    for steps, shift in arms:
        for seed in (0, 1, 2):
            out = RUNS / '04_sampler' / f's{steps}_shift{shift}' / 't1-wide' / f'seed{seed}'
            if (out / 'probe.json').exists():
                continue
            weights = torch.load(RUNS / '03_vae_latents/t1-wide' / f'seed{seed}' / 'weights.pt')
            torch.cuda.empty_cache()
            print(f'steps {steps} shift {shift} seed {seed}: ', end='', flush=True)
            points = rollouts.close_points(weights, d, wide, 8, workers=4, sampler=(steps, shift))
            fit = rollouts.close_fit(points)
            write_json(
                out / 'probe.json',
                {
                    'n_exec': 8,
                    'fit': {k: float(v) for k, v in fit.items()},
                    'points': {
                        str(s): [c.tolist(), None if g is None else g.tolist()]
                        for s, (c, g) in points.items()
                    },
                },
            )
    print('------ FINISHED ------')
