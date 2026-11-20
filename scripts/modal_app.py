"""dream-robot jobs on Modal: a GPU per call, billed per second, nothing while idle.

From the repo root:

    uv run modal run scripts/modal_app.py::check                     # GPU, torch, headless MuJoCo
    uv run modal run scripts/modal_app.py::job --step 07_instruction --task t5 --seed 0
    uv run modal run scripts/modal_app.py::pretrain --step 15_erase_pretrain --seed 0

The image is the project's uv.lock (exported without the dev group) plus the EGL loader and ffmpeg that MuJoCo and
the video decoder need (as scripts/runpod_setup.sh). The package code is copied in at each run, so
an edit needs no rebuild. Data and run folders live on two Modal Volumes mounted where the package
expects them (REPO = /root/dream-robot):

    dream-robot-data  -> /root/dream-robot/data                 (datasets; upload once)
    dream-robot-runs  -> /root/dream-robot/experiments/verne/runs  (run folders; pull with
                                                                    `modal volume get`)
"""

import subprocess

import modal

R = '/root/dream-robot'
REQS = 'scratch/modal_requirements.txt'
if modal.is_local():
    # the lock's runtime dependencies as a pip list: uv sync on the lock itself fails remotely, since
    # the dev group's ruffgrid is a path dependency on this machine
    subprocess.run(
        ['uv', 'export', '--no-dev', '--frozen', '--no-emit-project', '--no-hashes', '-o', REQS],
        check=True,
        capture_output=True,
    )
    # what run.py's meta.json records as the code version (the image has no .git)
    with open('scratch/modal_synced_from.json', 'w') as f:
        subprocess.run(
            ['uv', 'run', 'python', '-m', 'dream_robot.policies.verne.run', 'synced-from'],
            check=True,
            stdout=f,
        )
image = (
    modal.Image.debian_slim(python_version='3.14')
    .apt_install(
        'libegl1', 'libgl1', 'libglvnd0', 'ffmpeg', 'gcc'
    )  # gcc: torch.compile's triton builds its launcher
    .uv_pip_install(requirements=[REQS])
    .env(
        {
            'MUJOCO_GL': 'egl',
            'PYOPENGL_PLATFORM': 'egl',
            'PYTHONUNBUFFERED': '1',
            'HF_HUB_OFFLINE': '1',
        }
    )
    .add_local_dir('dream_robot', f'{R}/dream_robot', ignore=['**/__pycache__'])
    .add_local_dir('experiments/libero/scripts', f'{R}/experiments/libero/scripts', ignore=['**/__pycache__'])
    .add_local_file('scratch/modal_synced_from.json', f'{R}/synced_from.json')
)
# the first free one, in order (L40S ran out once, 2026-11-19); all are plenty for a 4.5M-param Verne
GPUS = ['L40S', 'A100-40GB', 'A10', 'L4']
# fast cards only, no slow fallback (2026-11-19: the A100 fallback ran ~3x slower than wanted)
FAST_GPUS = ['H100', 'L40S']
# jobs that render robosuite scenes (rollouts, EGL): no H100, whose graphics engine faulted (XID 31 / 109,
# 2026-11-19); L40S renders, the A100 rendered steps 17-18's seeds
# L40S only (user, 2026-11-19): the A100 fallback is slower
RENDER_GPUS = ['L40S']
data = modal.Volume.from_name('dream-robot-data', create_if_missing=True)
runs = modal.Volume.from_name('dream-robot-runs', create_if_missing=True)
app = modal.App('dream-robot', image=image)
VOLUMES = {f'{R}/data': data, f'{R}/experiments/verne/runs': runs}
# LIBERO (2026-11-20): its runs on their own Volume, mounted where experiments/libero/scripts write;
# data/libero/npz/ (to_npz.py's output) uploaded once to the data Volume at /libero/npz
libero_runs = modal.Volume.from_name('dream-robot-libero-runs', create_if_missing=True)
LIBERO_VOLUMES = {f'{R}/data': data, f'{R}/experiments/libero/runs': libero_runs}


@app.function(gpu=GPUS, timeout=600)
def check():
    # the GPU, torch on it, and one headless render of the T5 scene
    import os
    import subprocess
    import sys
    import time

    sys.path.insert(0, R)
    os.chdir(R)
    print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip())
    print('cpus', os.cpu_count())
    import torch

    x = torch.randn(4096, 4096, device='cuda')
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(50):
        x @ x
    torch.cuda.synchronize()
    print(f'torch {torch.__version__}, 50 matmuls 4096^2: {time.time() - t:.3f} s')
    from dream_robot.policies.verne import rollouts

    t = time.time()
    env = rollouts.make_env('t5')
    obs = env.reset(seed=1000)
    print(f'T5 env + reset {time.time() - t:.1f} s: "{env.prompt}", top {obs.images["top"].shape}')
    t = time.time()
    for _ in range(30):
        env.step(obs.state.copy())
    print(f'30 sim steps with rendering: {time.time() - t:.2f} s')
    env.close()


# cpu=8: the default container had 1 CPU (nproc) for the training loop and the rollout workers
@app.function(gpu=RENDER_GPUS, volumes=VOLUMES, timeout=4 * 3600, cpu=8)
def job(
    step: str,
    task: str,
    seed: int,
    workers: int = 8,
    vae: str = '',
    init: str = '',
    cotrain: float = 0.0,
    future_by: str = 'all',
    paired: bool = False,
    steps: int = 10000,  # = run.STEPS
    early: float = 0.0,
    erase: float = 0.0,
    position: float = 0.0,
    readout: str = 'patch',
):
    # run.py's job (train, sweeps at n_exec 8 and 16, probe) into the runs Volume; vae: another VAE's
    # weights under the runs Volume for this run only, e.g. '03_vae_latents/vae/weights.pt'; init: a run
    # folder on the runs Volume to start training from, e.g. '15_erase_pretrain/pretrain/seed0'
    import os
    import sys

    if vae:
        os.environ['VERNE_VAE_WEIGHTS'] = f'{R}/experiments/verne/runs/{vae}'

    sys.path.insert(0, R)
    os.chdir(R)
    from dream_robot.policies.verne import run

    try:
        run.job(
            step,
            task,
            seed,
            workers,
            init,
            cotrain,
            future_by,
            paired,
            steps,
            early,
            erase,
            position,
            readout,
        )
    finally:
        runs.commit()  # whatever finished, even if a later part failed


@app.function(gpu=FAST_GPUS, volumes=LIBERO_VOLUMES, timeout=4 * 3600)
def libero_vae(name: str = 'vae', steps: int = 10000):
    # experiments/libero/scripts/train_vae.py into the LIBERO runs Volume at <name>/ (weights.pt, log.json,
    # recon.png, ckpt/); pull: modal volume get --force dream-robot-libero-runs /<name> experiments/libero/runs/
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, R)
    sys.path.insert(0, f'{R}/experiments/libero/scripts')
    os.chdir(R)
    import train_vae

    try:
        train_vae.main(Path(f'{R}/experiments/libero/runs/{name}'), steps)
    finally:
        libero_runs.commit()


@app.function(gpu=GPUS, volumes=VOLUMES, timeout=4 * 3600)
def pretrain(step: str, seed: int = 0):
    # run.py's pretrain_job (step 15 stage 1: erase the cube a colour word names, on T5 reset scenes;
    # binding probe, edit.png) into the runs Volume
    import os
    import sys

    sys.path.insert(0, R)
    os.chdir(R)
    from dream_robot.policies.verne import run

    try:
        run.pretrain_job(step, seed)
    finally:
        runs.commit()


@app.function(gpu=GPUS, volumes=VOLUMES, timeout=4 * 3600)
def position(step: str, seed: int = 0, readout: str = 'patch', steps: int = 5000):
    # run.py's position_job (step 22 stage 1: the named cube's centre per camera; probes, position.png)
    import os
    import sys

    sys.path.insert(0, R)
    os.chdir(R)
    from dream_robot.policies.verne import run

    try:
        run.position_job(step, seed, readout, steps)
    finally:
        runs.commit()


# cpu=8: the default container had 1 CPU (nproc) for the training loop and the rollout workers
@app.function(gpu=FAST_GPUS, volumes=VOLUMES, timeout=4 * 3600, cpu=8)
def clips(step: str, seed: int = 0, steps: int = 20000, future_by: str = 'all', early: float = 0.0):
    # run.py's clips_job (step 19 stage 1: clip denoising on the T5 demos' clips; binding probe every 4000
    # steps and at the end, films.png) into the runs Volume
    import os
    import sys

    sys.path.insert(0, R)
    os.chdir(R)
    from dream_robot.policies.verne import run

    try:
        run.clips_job(step, seed, steps, future_by, early)
    finally:
        runs.commit()


@app.local_entrypoint()
def jobs(step: str, tasks: str, seeds: str = '0', workers: int = 8):
    # every (task, seed) at once, a GPU each:
    #   uv run modal run --detach scripts/modal_app.py::jobs --step 07_instruction --tasks t5,t1-wide-low
    pairs = [(step, t, int(s), workers) for t in tasks.split(',') for s in seeds.split(',')]
    for (_, t, s, _), _ in zip(pairs, job.starmap(pairs, return_exceptions=True)):
        print(f'{t} seed {s}: finished')
