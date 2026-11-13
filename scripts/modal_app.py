"""dream-robot jobs on Modal: a GPU per call, billed per second, nothing while idle.

From the repo root:

    uv run modal run scripts/modal_app.py::check                     # GPU, torch, headless MuJoCo
    uv run modal run scripts/modal_app.py::job --step 07_instruction --task t5 --seed 0

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
        check=True, capture_output=True,
    )
    # what run.py's meta.json records as the code version (the image has no .git)
    with open('scratch/modal_synced_from.json', 'w') as f:
        subprocess.run(['uv', 'run', 'python', '-m', 'dream_robot.policies.verne.run', 'synced-from'],
                       check=True, stdout=f)
image = (
    modal.Image.debian_slim(python_version='3.14')
    .apt_install('libegl1', 'libgl1', 'libglvnd0', 'ffmpeg')
    .uv_pip_install(requirements=[REQS])
    .env({'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1'})
    .add_local_dir('dream_robot', f'{R}/dream_robot', ignore=['**/__pycache__'])
    .add_local_file('scratch/modal_synced_from.json', f'{R}/synced_from.json')
)
data = modal.Volume.from_name('dream-robot-data', create_if_missing=True)
runs = modal.Volume.from_name('dream-robot-runs', create_if_missing=True)
app = modal.App('dream-robot', image=image)
VOLUMES = {f'{R}/data': data, f'{R}/experiments/verne/runs': runs}


@app.function(gpu='L40S', timeout=600)
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


@app.function(gpu='L40S', volumes=VOLUMES, timeout=4 * 3600)
def job(step: str, task: str, seed: int, workers: int = 8, vae: str = ''):
    # run.py's job (train, sweeps at n_exec 8 and 16, probe) into the runs Volume; vae: another VAE's
    # weights under the runs Volume for this run only, e.g. '03_vae_latents/vae/weights.pt'
    import os
    import sys

    if vae:
        os.environ['VERNE_VAE_WEIGHTS'] = f'{R}/experiments/verne/runs/{vae}'

    sys.path.insert(0, R)
    os.chdir(R)
    from dream_robot.policies.verne import run

    try:
        run.job(step, task, seed, workers)
    finally:
        runs.commit()  # whatever finished, even if a later part failed


@app.local_entrypoint()
def jobs(step: str, tasks: str, seeds: str = '0', workers: int = 8):
    # every (task, seed) at once, a GPU each:
    #   uv run modal run --detach scripts/modal_app.py::jobs --step 07_instruction --tasks t5,t1-wide-low
    pairs = [(step, t, int(s), workers) for t in tasks.split(',') for s in seeds.split(',')]
    for (_, t, s, _), _ in zip(pairs, job.starmap(pairs, return_exceptions=True)):
        print(f'{t} seed {s}: finished')
