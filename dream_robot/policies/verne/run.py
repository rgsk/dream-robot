"""Runs as jobs: one command trains a Verne on one task with one seed, sweeps it in the sim and
probes it, and writes everything to experiments/verne/runs/<step>/<task>/seed<k>/. Notebooks
only read those files.

    python -m dream_robot.policies.verne.run job 02_cosine_lr t1-wide 0
    python -m dream_robot.policies.verne.run queue 02_cosine_lr --tasks t1 t1-wide --seeds 0 1 2

    python -m dream_robot.policies.verne.run vae 03_vae_latents   (the canvas VAE, both tasks)

`queue` runs its jobs one after another, each in its own process (so each holds only its own
task's frames on the GPU), and writes each job's output to <run dir>/job.log. Detached, so a
VS Code reload or a dropped ssh doesn't kill it:

    setsid nohup python -m dream_robot.policies.verne.run queue 02_cosine_lr \\
        --tasks t1 t1-wide --seeds 0 1 2 > experiments/verne/runs/02_cosine_lr/queue.log 2>&1 &

A run folder holds:
    meta.json        commit, uncommitted changes?, package hash, GPU, task, seed, steps
    weights.pt       the trained weights (gitignored)
    log.json         [step, train loss, val loss] every 1000 steps
    sweep_8.json     rollout seed -> [success, failure mode, steps], n_exec 8 (and _16)
    probe.json       T1-wide only: rollout seed -> [cube xy, gripper xy at first close or null],
                     plus the fit (slopes, miss)
Each file is written when its part finishes, so a rerun skips what's there. A rerun under a
different package hash stops: the folder's numbers would mix two versions of the code.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import rollouts, vae
from .data import REPO, canvas, load_cameras, load_table, to_gpu
from .train import train

RUNS = REPO / 'experiments/verne/runs'
STEPS = 20000
N_EXECS = (8, 16)
# task -> dataset, wide spawn?; the splits are the notebooks' (val = 10 demos as dreamzero /
# uwm2; T1 trains on 10 labelled of the other 90, T1-wide on all 90)
TASKS = {
    't1': ('pick_place_cube_n100', False),
    't1-wide': ('pick_place_cube_wide_n100', True),
    # the expert squares the gripper up to the cube (record.py --square-up); rollouts as t1-wide
    't1-wide-square': ('pick_place_cube_wide_square_n100', True),
    # the same expert with joint noise sigma 0.05 on the executed actions, clean labels (--noise-sigma)
    't1-wide-square-noise': ('pick_place_cube_wide_square_noise005_n100', True),
    # the squaring expert executes a per-episode wrist bias in +-45 deg until the lift, clean labels
    't1-wide-square-bias': ('pick_place_cube_wide_square_bias45_n100', True),
    # both: the +-45 deg wrist bias and joint noise sigma 0.05, clean labels
    't1-wide-square-bias-noise': ('pick_place_cube_wide_square_bias45_noise005_n100', True),
}


def split(task: str) -> tuple[np.ndarray, np.ndarray]:
    # -> train ids, val ids (episode indices)
    val = np.random.default_rng(0).permutation(100)[:10]
    rest = np.setdiff1d(np.arange(100), val)
    if task == 't1':
        return np.random.default_rng(1).choice(rest, 10, replace=False), val
    return rest, val


def load_task(task: str):
    # 'base' or 'base+extra1+extra2' (extra = dataset names under data/robosuite, e.g. DAgger rounds;
    # 'extra:fail' keeps only that dataset's failed episodes, per its recording_summary.json)
    # -> state, action, episode, latent, train ids, val ids, wide. Extra datasets' episodes are
    # renumbered after the base's (1000 * k + i) and all go to training; validation stays the base's
    # held-out demos, so losses and scores compare with the base task's runs.
    base, *extra = task.split('+')
    name, wide = TASKS[base]
    train_ids, val_ids = split(base)
    parts = [(*load_table(name), vae.load_latents(name, vae.WEIGHTS))]
    for k, extra_name in enumerate(extra, 1):
        extra_name, _, only = extra_name.partition(':')
        s, a, e = load_table(extra_name)
        z = vae.load_latents(extra_name, vae.WEIGHTS)
        if only == 'fail':
            summary = json.loads(
                (REPO / 'data/robosuite' / extra_name / 'recording_summary.json').read_text()
            )
            failed = [i for i, ep in enumerate(summary['episodes']) if not ep['success']]
            keep = np.isin(e, failed)
            s, a, e, z = s[keep], a[keep], e[keep], z[keep]
        parts.append((s, a, e + 1000 * k, z))
        train_ids = np.concatenate([train_ids, np.unique(e) + 1000 * k])
    state, action, episode, latent = (np.concatenate(x) for x in zip(*parts))
    return state, action, episode, latent, train_ids, val_ids, wide


def package_hash() -> str:
    # sha256 over the package's .py files: names the code a run was made with, committed or not
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()[:12]


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def source() -> dict[str, str | bool]:
    # -> commit, dirty (uncommitted changes in the package). A pod gets the code by rsync with no
    # .git: there the sync writes synced_from.json at the repo root with the local machine's
    # answer at sync time (`python -m dream_robot.policies.verne.run synced-from > ...`)
    if commit := git('rev-parse', '--short', 'HEAD'):
        return {
            'commit': commit,
            'dirty': bool(git('status', '--porcelain', '--', 'dream_robot/policies/verne')),
        }
    return json.loads((REPO / 'synced_from.json').read_text())


def write_json(path: Path, obj) -> None:
    path.with_suffix('.partial').write_text(json.dumps(obj, indent=1))
    path.with_suffix('.partial').rename(path)  # never a half-written file


def job(step: str, task: str, seed: int, workers: int) -> None:
    out = RUNS / step / task / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    meta = source() | {
        'package_hash': package_hash(),
        'gpu': torch.cuda.get_device_name(),
        'task': task,
        'seed': seed,
        'steps': STEPS,
    }
    if (out / 'meta.json').exists():
        old = json.loads((out / 'meta.json').read_text())
        if old['package_hash'] != meta['package_hash']:
            sys.exit(
                f'{out} was made with package {old["package_hash"]}, now {meta["package_hash"]}'
            )
    else:
        write_json(out / 'meta.json', meta | {'started': time.strftime('%Y-%m-%d %H:%M:%S')})

    state, action, episode, latent, train_ids, val_ids, wide = load_task(task)
    d = to_gpu(latent, state, action, episode, train_ids)

    if (out / 'weights.pt').exists():
        weights = torch.load(out / 'weights.pt')
    else:
        run = train(d, train_ids, val_ids, STEPS, seed=seed)
        weights = run['weights']
        write_json(out / 'log.json', run['log'])
        torch.save(weights, out / 'weights.pt')

    for n_exec in N_EXECS:
        path = out / f'sweep_{n_exec}.json'
        if not path.exists():
            res = rollouts.sweep(weights, d, wide, n_exec, workers)
            write_json(path, {str(s): list(r) for s, r in res.items()})

    if wide and not (out / 'probe.json').exists():
        points = rollouts.close_points(weights, d, wide, 8, workers)
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
    print(f'done: {out}')


def train_vae(step: str, seed: int = 0) -> None:
    # the canvas VAE on the train demos of both tasks -> RUNS/<step>/vae/: weights.pt (with the
    # latent stats), log.json [step, recon, KL, val recon, KL], meta.json, recon.png (val frames,
    # real above reconstructed)
    out = RUNS / step / 'vae'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'weights.pt').exists():
        sys.exit(f'{out} already has weights')
    write_json(
        out / 'meta.json',
        source()
        | {
            'package_hash': package_hash(), 'gpu': torch.cuda.get_device_name(), 'seed': seed,
            'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
    )
    frames, val = [], []
    for task, (name, _) in TASKS.items():
        _, val_ids = split(task)
        episode = load_table(name)[2]
        c = canvas(*load_cameras(name))
        frames.append(c[~np.isin(episode, val_ids)])
        val.append(c[np.flatnonzero(np.isin(episode, val_ids))[::40]])
        del c
    frames, val = np.concatenate(frames), np.concatenate(val)
    print(f'{len(frames)} training frames, {len(val)} val frames', flush=True)
    run = vae.train(frames, val, seed=seed)
    write_json(out / 'log.json', run['log'])
    torch.save(run['weights'], out / 'weights.pt')
    net = vae.VAE().cuda().eval()
    net.load_state_dict(run['weights'])
    pick = val[np.linspace(0, len(val) - 1, 6).astype(int)]
    rec = vae.reconstruct(net, pick)
    import imageio.v3 as iio

    grid = np.concatenate([np.concatenate(list(pick), 2), np.concatenate(list(rec), 2)], 1)
    iio.imwrite(out / 'recon.png', grid.transpose(1, 2, 0))
    print(f'done: {out}')


def queue(step: str, tasks: list[str], seeds: list[int], workers: int) -> None:
    # every (task, seed), one at a time, each a fresh process; a failed job doesn't stop the rest
    for task in tasks:
        for seed in seeds:
            out = RUNS / step / task / f'seed{seed}'
            out.mkdir(parents=True, exist_ok=True)
            print(f'{time.strftime("%H:%M:%S")} {task} seed {seed} -> {out}/job.log', flush=True)
            with open(out / 'job.log', 'a') as log:
                cmd = [sys.executable, '-m', __spec__.name, 'job', step, task, str(seed)]
                code = subprocess.run(
                    [*cmd, '--workers', str(workers)],
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                ).returncode
            print(f'{time.strftime("%H:%M:%S")} {task} seed {seed}: exit {code}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    j = sub.add_parser('job')
    j.add_argument('step')
    j.add_argument('task', help="a TASKS name, or 'base+extra[:fail]+...' (see load_task)")
    j.add_argument('seed', type=int)
    q = sub.add_parser('queue')
    q.add_argument('step')
    q.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    q.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    for s in (j, q):
        s.add_argument('--workers', type=int, default=8)
    sub.add_parser('synced-from')
    g = sub.add_parser('dagger')  # collect one DAgger round: the policy drives, StateExpert labels
    g.add_argument(
        'policy_run', help='run folder under RUNS, e.g. 05_square/t1-wide-square-noise/seed0'
    )
    g.add_argument('base_task', help='the task the policy was trained on (its stats)')
    g.add_argument('out_name', help='dataset to write under data/robosuite')
    g.add_argument('--episodes', type=int, default=100)
    g.add_argument('--seed-start', type=int, default=2000)
    v = sub.add_parser('vae')
    v.add_argument('step')
    a = p.parse_args()
    if a.cmd == 'dagger':
        from .dagger import collect

        s = collect(a.policy_run, a.base_task, a.out_name, a.episodes, a.seed_start)
        print(f'{s.kept} episodes / {s.total_frames} frames, policy placed {s.success_rate:.0%}')
    elif a.cmd == 'vae':
        train_vae(a.step)
    elif a.cmd == 'synced-from':
        print(json.dumps(source()))
    elif a.cmd == 'job':
        job(a.step, a.task, a.seed, a.workers)
    else:
        queue(a.step, a.tasks, a.seeds, a.workers)
