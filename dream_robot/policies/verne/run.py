"""Runs as jobs: one command trains a Verne on one task with one seed, sweeps it in the sim and
probes it, and writes everything to experiments/verne/runs/<step>/<task>/seed<k>/. Notebooks
only read those files.

    python -m dream_robot.policies.verne.run job 02_cosine_lr t1-wide 0
    python -m dream_robot.policies.verne.run queue 02_cosine_lr --tasks t1 t1-wide --seeds 0 1 2

    python -m dream_robot.policies.verne.run vae 07_instruction   (the canvas VAE on VAE_DATA)
    python -m dream_robot.policies.verne.run pretrain 15_erase_pretrain 0   (step 15 stage 1, pretrain.py)

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
    sweep_8_position.json, sweep_8_both.json   T5 only: the prompt names the cube by place / both
    probe.json       T1-wide and T5 (the named cube): rollout seed -> [cube xy, gripper xy at first
                     close or null],
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
from .data import (
    REPO,
    canvas,
    load_cameras,
    load_instructions,
    load_slots,
    load_table,
    to_gpu,
)
from .model import Verne
from .train import train

RUNS = REPO / 'experiments/verne/runs'
STEPS = 10000  # stage 2 (2026-11-20; 20000 before): val loss bottoms out ~10k, step 20 s0 scored the same at 10k
N_EXECS = (8, 16)
# task -> dataset, scene (rollouts.make_env); the splits are the notebooks' (val = a tenth of the
# demos, as dreamzero / uwm2; T1 trains on 10 labelled of the other 90, the rest on all of them)
TASKS = {
    't1': ('pick_place_cube_n100', 't1'),
    't1-wide': ('pick_place_cube_wide_n100', 't1-wide'),
    # the expert squares the gripper up to the cube (record.py --square-up); rollouts as t1-wide
    't1-wide-square': ('pick_place_cube_wide_square_n100', 't1-wide'),
    # the same expert with joint noise sigma 0.05 on the executed actions, clean labels (--noise-sigma)
    't1-wide-square-noise': ('pick_place_cube_wide_square_noise005_n100', 't1-wide'),
    # the squaring expert executes a per-episode wrist bias in +-45 deg until the lift, clean labels
    't1-wide-square-bias': ('pick_place_cube_wide_square_bias45_n100', 't1-wide'),
    # both: the +-45 deg wrist bias and joint noise sigma 0.05, clean labels
    't1-wide-square-bias-noise': ('pick_place_cube_wide_square_bias45_noise005_n100', 't1-wide'),
    # step 07: the expert grasps 0.5 cm below the cube centre (task.yaml grasp_dz), square + shake
    't1-wide-low': ('pick_place_cube_wide_square_low_noise005_n100', 't1-wide'),
    # T5: two cubes, two bins, 'put the <colour> cube in the <side> box'; square + shake, low grasp
    't5': ('instructed_pick_place_square_noise005_n150', 't5'),
}


def split(task: str, n: int = 100) -> tuple[np.ndarray, np.ndarray]:
    # -> train ids, val ids (episode indices) of a task's n demos; a tenth held out (n = 100: the
    # same 10 as always)
    val = np.random.default_rng(0).permutation(n)[: n // 10]
    rest = np.setdiff1d(np.arange(n), val)
    if task == 't1':
        return np.random.default_rng(1).choice(rest, 10, replace=False), val
    return rest, val


def load_task(task: str):
    # 'base' or 'base+extra1+extra2' (extra = dataset names under data/robosuite, e.g. DAgger rounds;
    # 'extra:fail' keeps only that dataset's failed episodes, per its recording_summary.json)
    # -> state, action, episode, latent, instruction, slot, train ids, val ids, scene. Extra datasets'
    # episodes are renumbered after the base's (1000 * k + i) and all go to training; validation
    # stays the base's held-out demos, so losses and scores compare with the base task's runs.
    base, *extra = task.split('+')
    name, scene = TASKS[base]
    s, a, e = load_table(name)
    train_ids, val_ids = split(base, int(e.max()) + 1)
    parts = [
        (s, a, e, vae.load_latents(name, vae.WEIGHTS), load_instructions(name), load_slots(name))
    ]
    for k, extra_name in enumerate(extra, 1):
        extra_name, _, only = extra_name.partition(':')
        s, a, e = load_table(extra_name)
        z = vae.load_latents(extra_name, vae.WEIGHTS)
        ins, slot = load_instructions(extra_name), load_slots(extra_name)
        if only == 'fail':
            summary = json.loads(
                (REPO / 'data/robosuite' / extra_name / 'recording_summary.json').read_text()
            )
            failed = [i for i, ep in enumerate(summary['episodes']) if not ep['success']]
            keep = np.isin(e, failed)
            s, a, e, z, ins, slot = s[keep], a[keep], e[keep], z[keep], ins[keep], slot[keep]
        parts.append((s, a, e + 1000 * k, z, ins, slot))
        train_ids = np.concatenate([train_ids, np.unique(e) + 1000 * k])
    state, action, episode, latent, instruction, slot = (np.concatenate(x) for x in zip(*parts))
    return state, action, episode, latent, instruction, slot, train_ids, val_ids, scene


def package_hash() -> str:
    # sha256 over the package's .py files: names the code a run was made with, committed or not
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob('*.py')):
        h.update(path.name.encode() + path.read_bytes())
    return h.hexdigest()[:12]


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ['git', *args], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    except FileNotFoundError:  # no git at all (a Modal image): source() reads synced_from.json
        return ''


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


def evaluate(out: Path, weights: dict, d, scene: str, workers: int) -> None:
    # a trained Verne's results into `out`: T5's binding and locate probes, sweeps at N_EXECS, the
    # close-point probe; each skipped if its file is there (a rerun finishes what a crash left)
    if scene == 't5' and not (out / 'patch_binding_probe.json').exists():
        # does the binding survive (or appear in) the demo training? as pretrain_job's probe; before the
        # sweeps, whose rollout workers leave the GPU too full for it (steps 15, 16 ran out of memory)
        from . import pretrain as pt
        from .data import load_scenes

        net = Verne().cuda()
        net.load_state_dict(weights)
        sc, val = load_scenes('train', vae.WEIGHTS), load_scenes('val', vae.WEIGHTS)
        write_json(
            out / 'patch_binding_probe.json', pt.binding_probe(net, sc, val, d.s_mean, d.s_std)
        )
        if not (out / 'patch_locate_probe.json').exists():
            from .data import load_erased

            er, ev = load_erased('train', vae.WEIGHTS), load_erased('val', vae.WEIGHTS)
            write_json(
                out / 'patch_locate_probe.json',
                pt.locate_probe(net, sc, val, er, ev, d.s_mean, d.s_std),
            )
        del net, sc, val
        torch.cuda.empty_cache()

    # the recorded prompts only (step 16: training never sees near/far, so place / both sweeps of steps
    # 13-15 would mean nothing)
    for n_exec in N_EXECS:
        path = out / f'sweep_{n_exec}.json'
        if not path.exists():
            res = rollouts.sweep(weights, d, scene, n_exec, workers)
            write_json(path, {str(s): list(r) for s, r in res.items()})

    if scene != 't1' and not (out / 'probe.json').exists():  # T1's cube barely moves
        points = rollouts.close_points(weights, d, scene, 8, workers)
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


def job(
    step: str,
    task: str,
    seed: int,
    workers: int,
    init: str = '',
    cotrain: float = 0.0,
    future_by: str = 'all',
    paired: bool = False,
    steps: int = STEPS,
    early: float = 0.0,
    erase: float = 0.0,
    position: float = 0.0,
    readout: str = 'patch',
) -> None:
    # position > 0 (step 22): position.py's loss on reset scenes joins the action loss, weighted position, B 64,
    # with readout ('patch' / 'free'; the init's head)
    # erase > 0 (step 21a): step 15's erase loss on reset scenes joins the action loss, weighted erase, B 64
    # init: a run folder under RUNS whose weights.pt training starts from (step 15: the stage 1 run,
    # '15_erase_pretrain/pretrain/seed0'), else a fresh Verne; cotrain > 0 (step 19g): the clip loss
    # (future_by as the clip job's) joins the action loss, weighted cotrain, with both probes every 4k steps
    out = RUNS / step / task / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    meta = source() | {
        'package_hash': package_hash(),
        'gpu': torch.cuda.get_device_name(),
        'task': task,
        'seed': seed,
        'steps': steps,
        'vae': str(vae.WEIGHTS.relative_to(REPO)),
        'init': init,
        'cotrain': cotrain,
        'future_by': future_by,
        'paired': paired,
        'early': early,
        'erase': erase,
        'position': position,
        'readout': readout,
    }
    if (out / 'meta.json').exists():
        old = json.loads((out / 'meta.json').read_text())
        if old['package_hash'] != meta['package_hash']:
            sys.exit(
                f'{out} was made with package {old["package_hash"]}, now {meta["package_hash"]}'
            )
    else:
        write_json(out / 'meta.json', meta | {'started': time.strftime('%Y-%m-%d %H:%M:%S')})

    state, action, episode, latent, instruction, slot, train_ids, val_ids, scene = load_task(task)
    d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)

    if (out / 'weights.pt').exists():
        weights = torch.load(out / 'weights.pt')
    else:
        start = torch.load(RUNS / init / 'weights.pt') if init else None
        probe = None
        if (cotrain > 0 or erase > 0 or position > 0) and scene == 't5':
            from . import position as pos
            from . import pretrain as pt
            from .data import load_erased, load_scenes

            sc, sv = load_scenes('train', vae.WEIGHTS), load_scenes('val', vae.WEIGHTS)
            er, ev = load_erased('train', vae.WEIGHTS), load_erased('val', vae.WEIGHTS)
            probe = lambda net: {  # noqa: E731
                'binding': pt.binding_probe(net, sc, sv, d.s_mean, d.s_std),
                'locate': pt.locate_probe(net, sc, sv, er, ev, d.s_mean, d.s_std),
            }
        run = train(
            d,
            train_ids,
            val_ids,
            steps,
            seed=seed,
            ckpt=out / 'ckpt',
            init=start,
            cotrain=cotrain,
            future_by=future_by,
            probe=probe,
            paired=paired,
            early=early,
            erase=erase,
            erase_data=(sc, er, 64) if erase > 0 else None,
            position=position,
            position_data=(sc, pos.centres(sc, er), readout, 64) if position > 0 else None,
        )
        weights = run['weights']
        write_json(out / 'log.json', run['log'])
        torch.save(weights, out / 'weights.pt')

    evaluate(out, weights, d, scene, workers)
    print(f'done: {out}')


def eval_ckpt(step: str, task: str, seed: int, at: int, workers: int) -> None:
    # an earlier checkpoint of a run (ckpt/step<at>.pt, saved every 5000 steps) evaluated like the run's end
    # -> RUNS/<step>/<task>/seed<k>/at<at>/. Mid-run weights sit mid cosine schedule (lr not yet decayed)
    run_dir = RUNS / step / task / f'seed{seed}'
    out = run_dir / f'at{at}'
    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / 'meta.json', source() | {'package_hash': package_hash(), 'ckpt': f'step{at}.pt'}
    )
    weights = torch.load(run_dir / 'ckpt' / f'step{at}.pt', weights_only=False)['net']
    state, action, episode, latent, instruction, slot, train_ids, val_ids, scene = load_task(task)
    d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)
    evaluate(out, weights, d, scene, workers)
    print(f'done: {out}')


# the VAE's frames: dataset -> episodes. Each keeps its first-tenth held out (the same draw as split's)
VAE_DATA = {
    'pick_place_cube_n100': 100,  # T1
    'pick_place_cube_wide_square_low_noise005_n100': 100,  # T1-wide square + shake, low grasp
    'pick_place_cube_wide_square_bias45_n100': 100,  # T1-wide, wrist +-45 deg crooked until lift
    'instructed_pick_place_square_noise005_n150': 150,  # T5 square + shake, low grasp
}


def vae_frames(every: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # VAE_DATA's training canvases (every `every`-th frame of the train demos) and a few held-out ones
    frames, val = [], []
    for name, n in VAE_DATA.items():
        val_ids = np.random.default_rng(0).permutation(n)[: n // 10]
        episode = load_table(name)[2]
        c = canvas(*load_cameras(name))
        frames.append(c[np.flatnonzero(~np.isin(episode, val_ids))[::every]])
        val.append(c[np.flatnonzero(np.isin(episode, val_ids))[::40]])
        del c
    return np.concatenate(frames), np.concatenate(val)


def train_vae(
    step: str,
    seed: int = 0,
    every: int = 2,
    steps: int = 10000,
    name: str = 'vae',
    extend_from: str | None = None,
) -> None:
    # the canvas VAE on VAE_DATA's train demos, every `every`-th frame (30 Hz neighbours are near
    # copies) -> RUNS/<step>/vae/: weights.pt (with the
    # latent stats), log.json [step, recon, KL, val recon, KL], meta.json, recon.png (val frames,
    # real above reconstructed), ckpt/ (step<N>.pt: a rerun resumes from the latest; stable.pt: extend_from for a
    # longer run, as '<step>/<name>')
    out = RUNS / step / name
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'weights.pt').exists():
        sys.exit(f'{out} already has weights')
    write_json(
        out / 'meta.json',
        source()
        | {
            'package_hash': package_hash(),
            'gpu': torch.cuda.get_device_name(),
            'seed': seed,
            'data': VAE_DATA,
            'every': every,
            'steps': steps,
            'extend_from': extend_from,
            'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        },
    )
    frames, val = vae_frames(every)
    print(f'{len(frames)} training frames, {len(val)} val frames', flush=True)
    run = vae.train(
        frames,
        val,
        steps=steps,
        seed=seed,
        ckpt=out / 'ckpt',
        extend_from=RUNS / extend_from / 'ckpt' if extend_from else None,
    )
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


PRETRAIN_STEPS = 5000  # step 15: the probe passes after 300; step 14 overfit past ~6k


def pretrain_job(step: str, seed: int) -> None:
    # step 15 stage 1 (erase the named cube) -> RUNS/<step>/pretrain/seed<k>/: meta.json, log.json,
    # weights.pt (a whole Verne, its action side untrained), patch_binding_probe.json (this model and an
    # untrained one), edit.png
    from . import pretrain as pt
    from .data import load_erased, load_scenes

    out = RUNS / step / 'pretrain' / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    meta = source() | {
        'package_hash': package_hash(),
        'gpu': torch.cuda.get_device_name(),
        'seed': seed,
        'steps': PRETRAIN_STEPS,
        'vae': str(vae.WEIGHTS.relative_to(REPO)),
    }
    if (out / 'meta.json').exists():
        old = json.loads((out / 'meta.json').read_text())
        if old['package_hash'] != meta['package_hash']:
            sys.exit(
                f'{out} was made with package {old["package_hash"]}, now {meta["package_hash"]}'
            )
    else:
        write_json(out / 'meta.json', meta | {'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    sc, val = load_scenes('train', vae.WEIGHTS), load_scenes('val', vae.WEIGHTS)
    erased, val_erased = load_erased('train', vae.WEIGHTS), load_erased('val', vae.WEIGHTS)
    # the T5 demos' state stats, which stage 2 will use too
    state, _, episode = load_table(TASKS['t5'][0])
    train_ids, _ = split('t5', int(episode.max()) + 1)
    s = torch.tensor(state[np.isin(episode, train_ids)], device='cuda')
    s_mean, s_std = s.mean(0), s.std(0)

    if (out / 'weights.pt').exists():
        weights = torch.load(out / 'weights.pt')
    else:
        run = pt.pretrain(
            sc, erased, val, val_erased, s_mean, s_std, PRETRAIN_STEPS, seed=seed, ckpt=out / 'ckpt'
        )
        weights = run['weights']
        write_json(out / 'log.json', run['log'])
        torch.save(weights, out / 'weights.pt')
    net = Verne().cuda()
    net.load_state_dict(weights)
    if not (out / 'patch_binding_probe.json').exists():
        torch.manual_seed(0)
        print('probe, untrained Verne:', flush=True)
        floor = pt.binding_probe(Verne().cuda(), sc, val, s_mean, s_std)
        print('probe, pretrained:', flush=True)
        write_json(
            out / 'patch_binding_probe.json',
            {'pretrained': pt.binding_probe(net, sc, val, s_mean, s_std), 'untrained': floor},
        )
    if not (out / 'edit.png').exists():
        import imageio.v3 as iio

        grid = pt.edit_picture(net, vae.load(vae.WEIGHTS), val, val_erased, s_mean, s_std)
        iio.imwrite(out / 'edit.png', grid.transpose(1, 2, 0))
    print(f'done: {out}')


def position_job(step: str, seed: int, readout: str, steps: int = PRETRAIN_STEPS) -> None:
    # step 22 stage 1 (the named cube's centre per camera, position.py) -> RUNS/<step>/pretrain_<readout>/seed<k>/:
    # meta.json, log.json, weights.pt, patch_locate_probe.json, patch_binding_probe.json, position.png
    from . import position as pos
    from . import pretrain as pt
    from .data import load_erased, load_scenes

    out = RUNS / step / f'pretrain_{readout}' / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    meta = source() | {
        'package_hash': package_hash(),
        'gpu': torch.cuda.get_device_name(),
        'seed': seed,
        'steps': steps,
        'readout': readout,
        'vae': str(vae.WEIGHTS.relative_to(REPO)),
    }
    if (out / 'meta.json').exists():
        old = json.loads((out / 'meta.json').read_text())
        if old['package_hash'] != meta['package_hash']:
            sys.exit(
                f'{out} was made with package {old["package_hash"]}, now {meta["package_hash"]}'
            )
    else:
        write_json(out / 'meta.json', meta | {'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    sc, val = load_scenes('train', vae.WEIGHTS), load_scenes('val', vae.WEIGHTS)
    erased, val_erased = load_erased('train', vae.WEIGHTS), load_erased('val', vae.WEIGHTS)
    cen, val_cen = pos.centres(sc, erased), pos.centres(val, val_erased)
    state, _, episode = load_table(TASKS['t5'][0])  # the T5 demos' state stats, as stage 2's
    train_ids, _ = split('t5', int(episode.max()) + 1)
    s = torch.tensor(state[np.isin(episode, train_ids)], device='cuda')
    s_mean, s_std = s.mean(0), s.std(0)

    if (out / 'weights.pt').exists():
        weights = torch.load(out / 'weights.pt')
    else:
        run = pos.pretrain(
            sc, cen, val, val_cen, s_mean, s_std, steps, readout, seed=seed, ckpt=out / 'ckpt'
        )
        weights = run['weights']
        write_json(out / 'log.json', run['log'])
        torch.save(weights, out / 'weights.pt')
    net = Verne().cuda()
    net.load_state_dict(weights)
    if not (out / 'patch_locate_probe.json').exists():
        write_json(
            out / 'patch_locate_probe.json',
            pt.locate_probe(net, sc, val, erased, val_erased, s_mean, s_std),
        )
    if not (out / 'patch_binding_probe.json').exists():
        write_json(out / 'patch_binding_probe.json', pt.binding_probe(net, sc, val, s_mean, s_std))
    if not (out / 'position.png').exists():
        import imageio.v3 as iio

        img = pos.picture(net, vae.load(vae.WEIGHTS), val, val_cen, s_mean, s_std, readout)
        iio.imwrite(out / 'position.png', img)
    print(f'done: {out}')


def clips_job(step: str, seed: int, steps: int, future_by: str = 'all', early: float = 0.0) -> None:
    # future_by: how the future is predicted (clips.velocity): 'all' / 'present' (steps 19, 19b), 'heads' (19d),
    # 'change' (19e: each present patch's own change, films.png is then the change picture), 'change_top' (19f: the
    # same, loss on the top camera's patches only)
    # step 19 stage 1 (clip denoising on the T5 demos' clips, video only) -> RUNS/<step>/pretrain/seed<k>/:
    # meta.json, log.json (binding + locate probes every 4000 steps), weights.pt, patch_binding_probe.json,
    # patch_locate_probe.json, films.png (binding_films on val reset scenes)
    from . import clips
    from . import pretrain as pt
    from .data import load_erased, load_scenes

    out = RUNS / step / 'pretrain' / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    meta = source() | {
        'package_hash': package_hash(),
        'gpu': torch.cuda.get_device_name(),
        'seed': seed,
        'steps': steps,
        'future_by': future_by,
        'early': early,
        'vae': str(vae.WEIGHTS.relative_to(REPO)),
    }
    if (out / 'meta.json').exists():
        old = json.loads((out / 'meta.json').read_text())
        if old['package_hash'] != meta['package_hash']:
            sys.exit(
                f'{out} was made with package {old["package_hash"]}, now {meta["package_hash"]}'
            )
    else:
        write_json(out / 'meta.json', meta | {'started': time.strftime('%Y-%m-%d %H:%M:%S')})
    state, action, episode, latent, instruction, slot, train_ids, val_ids, _ = load_task('t5')
    d = to_gpu(latent, state, instruction, slot, action, episode, train_ids)

    sc, val = load_scenes('train', vae.WEIGHTS), load_scenes('val', vae.WEIGHTS)
    erased, val_erased = load_erased('train', vae.WEIGHTS), load_erased('val', vae.WEIGHTS)
    if (out / 'weights.pt').exists():
        weights = torch.load(out / 'weights.pt')
    else:
        run = clips.pretrain_clips(
            d,
            train_ids,
            val_ids,
            steps,
            seed=seed,
            ckpt=out / 'ckpt',
            probe=lambda net: {
                'binding': pt.binding_probe(net, sc, val, d.s_mean, d.s_std),
                'locate': pt.locate_probe(net, sc, val, erased, val_erased, d.s_mean, d.s_std),
            },
            future_by=future_by,
            early=early,
        )
        weights = run['weights']
        write_json(out / 'log.json', run['log'])
        torch.save(weights, out / 'weights.pt')
    net = Verne().cuda()
    net.load_state_dict(weights)
    if not (out / 'patch_binding_probe.json').exists():
        write_json(
            out / 'patch_binding_probe.json', pt.binding_probe(net, sc, val, d.s_mean, d.s_std)
        )
    if not (out / 'patch_locate_probe.json').exists():
        write_json(
            out / 'patch_locate_probe.json',
            pt.locate_probe(net, sc, val, erased, val_erased, d.s_mean, d.s_std),
        )
    if not (out / 'films.png').exists():
        import imageio.v3 as iio

        picture = (
            clips.change_picture(net, vae.load(vae.WEIGHTS), val, d.s_mean, d.s_std)
            if future_by in ('change', 'change_top')
            else clips.binding_films(
                net, vae.load(vae.WEIGHTS), val, d.s_mean, d.s_std, future_by=future_by
            )
        )
        iio.imwrite(out / 'films.png', picture)
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
    j.add_argument(
        '--init',
        default='',
        help='a run folder under RUNS to start from, e.g. 15_erase_pretrain/pretrain/seed0',
    )
    j.add_argument('--cotrain', type=float, default=0.0, help='weight of the clip loss (step 19g)')
    j.add_argument(
        '--paired', action='store_true', help='clips = the moments of the demo batch (step 19i)'
    )
    j.add_argument('--steps', type=int, default=STEPS, help='training steps (cosine over them)')
    j.add_argument(
        '--early', type=float, default=0.0, help='share of samples from the decision window (19k)'
    )
    j.add_argument(
        '--future-by', choices=('all', 'present', 'heads', 'change', 'change_top'), default='all'
    )
    j.add_argument('--erase', type=float, default=0.0, help='weight of the erase loss (step 21a)')
    j.add_argument(
        '--position', type=float, default=0.0, help='weight of the position loss (step 22)'
    )
    j.add_argument(
        '--readout',
        choices=('patch', 'free', 'spread'),
        default='patch',
        help='position head (step 22)',
    )
    ev = sub.add_parser('eval', help="a run's ckpt/step<N>.pt evaluated like its end")
    ev.add_argument('step')
    ev.add_argument('task')
    ev.add_argument('seed', type=int)
    ev.add_argument('at', type=int, help='the checkpoint step, a multiple of 5000')
    ev.add_argument('--workers', type=int, default=8)
    pre = sub.add_parser('pretrain')
    pre.add_argument('step')
    pre.add_argument('seed', type=int)
    po = sub.add_parser('position')  # step 22 stage 1
    po.add_argument('step')
    po.add_argument('seed', type=int)
    po.add_argument('--readout', choices=('patch', 'free', 'spread'), default='patch')
    po.add_argument('--steps', type=int, default=PRETRAIN_STEPS)
    cl = sub.add_parser('clips')
    cl.add_argument('step')
    cl.add_argument('seed', type=int)
    cl.add_argument('--steps', type=int, default=20000)
    cl.add_argument(
        '--early', type=float, default=0.0, help='share of clips from the decision window (19j)'
    )
    cl.add_argument(
        '--future-by', choices=('all', 'present', 'heads', 'change', 'change_top'), default='all'
    )
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
    v.add_argument('--steps', type=int, default=10000)
    v.add_argument('--name', default='vae', help='folder under RUNS/<step>')
    v.add_argument('--extend-from', help="a shorter run's '<step>/<name>': continue its stable.pt")
    a = p.parse_args()
    if a.cmd == 'dagger':
        from .dagger import collect

        s = collect(a.policy_run, a.base_task, a.out_name, a.episodes, a.seed_start)
        print(f'{s.kept} episodes / {s.total_frames} frames, policy placed {s.success_rate:.0%}')
    elif a.cmd == 'vae':
        train_vae(a.step, steps=a.steps, name=a.name, extend_from=a.extend_from)
    elif a.cmd == 'synced-from':
        print(json.dumps(source()))
    elif a.cmd == 'job':
        job(
            a.step,
            a.task,
            a.seed,
            a.workers,
            a.init,
            a.cotrain,
            a.future_by,
            a.paired,
            a.steps,
            a.early,
            a.erase,
            a.position,
            a.readout,
        )
    elif a.cmd == 'eval':
        eval_ckpt(a.step, a.task, a.seed, a.at, a.workers)
    elif a.cmd == 'pretrain':
        pretrain_job(a.step, a.seed)
    elif a.cmd == 'position':
        position_job(a.step, a.seed, a.readout, a.steps)
    elif a.cmd == 'clips':
        clips_job(a.step, a.seed, a.steps, a.future_by, a.early)
    else:
        queue(a.step, a.tasks, a.seeds, a.workers)
