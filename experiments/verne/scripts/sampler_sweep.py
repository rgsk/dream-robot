# Step 04, the sampler (box 05): step 03's trained T1-wide policies, unchanged, swept with other samplers.
# Writes experiments/verne/runs/04_sampler/<arm>/t1-wide/seed<k>/{sweep_8,sweep_16,meta}.json, skipping files
# already there. arm = s<steps>_shift<shift>, e.g. s4_shift1.5 (Verne's test config) or s10_shift1.0 (step 03's).
# usage: python experiments/verne/scripts/sampler_sweep.py 10:1.0 4:1.0 4:1.5 10:1.5   (steps:shift arms)
import sys
import time

import torch

from dream_robot.policies.verne import rollouts, vae
from dream_robot.policies.verne.data import load_table, to_gpu
from dream_robot.policies.verne.run import RUNS, TASKS, package_hash, source, split, write_json

if __name__ == '__main__':  # the sweep's spawned workers import this file; only the parent runs it
    arms = [(int(a.split(':')[0]), float(a.split(':')[1])) for a in sys.argv[1:]]
    name, wide = TASKS['t1-wide']
    train_ids, _ = split('t1-wide')
    state, action, episode = load_table(name)
    d = to_gpu(vae.load_latents(name, vae.WEIGHTS), state, action, episode, train_ids)
    for steps, shift in arms:
        for seed in (0, 1, 2):
            weights = torch.load(RUNS / '03_vae_latents/t1-wide' / f'seed{seed}' / 'weights.pt')
            out = RUNS / '04_sampler' / f's{steps}_shift{shift}' / 't1-wide' / f'seed{seed}'
            out.mkdir(parents=True, exist_ok=True)
            if not (out / 'meta.json').exists():
                write_json(
                    out / 'meta.json',
                    source()
                    | {
                        'package_hash': package_hash(), 'gpu': torch.cuda.get_device_name(),
                        'weights': f'03_vae_latents/t1-wide/seed{seed}', 'steps': steps,
                        'shift': shift, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
                    },
                )
            for n_exec in (8, 16):
                path = out / f'sweep_{n_exec}.json'
                if path.exists():
                    continue
                print(
                    f'{time.strftime("%H:%M:%S")} steps {steps} shift {shift} seed {seed}: ',
                    end='',
                    flush=True,
                )
                torch.cuda.empty_cache()  # the 8 GB card: free the parent's cache before workers start
                res = rollouts.sweep(weights, d, wide, n_exec, workers=4, sampler=(steps, shift))
                write_json(path, {str(s): list(r) for s, r in res.items()})
    print('------ FINISHED ------')
