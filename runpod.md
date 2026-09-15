# RunPod for dream-robot

The shared pod/volume setup (network volume, SSH key, `/workspace/setup.sh`, wandb secrets) is
documented in `~/Documents/codes/projects/llm/runpod.md`. This file is what is different for
dream-robot, set up and measured on 2026-09-16 (log: `experiments/t2/notes.md`, steps 7–8).

## Should this run on a pod at all?

**Usually not.** Measured on an RTX 4000 Ada pod against the local machine (20 cores, RTX 4060):

| step | local | pod |
|---|---|---|
| record 100 half-shake demos | ~4 min | 6.3 min (pod-local disk) · ~35 min (network volume) |
| eval, 20 episodes | ~1 min | 45 s |
| BC, one epoch on 100 demos | ~9 s | ~26 s (60 epochs: 11.1 min) |

- Recording and BC are **CPU-bound** (MuJoCo, the scripted expert's IK, video decoding). The pod
  reports 48 cores but its container quota is **7.65 cores**
  (`/sys/fs/cgroup/cpu/cpu.cfs_quota_us` / `cfs_period_us`). Check that before choosing a pod.
- ACT is **GPU-bound** (99% GPU locally), but an RTX 4000 Ada is roughly 4060-class.

A pod pays off for: **several independent runs at once** (seeds, grids), a **much faster GPU** for
ACT / Diffusion Policy, or keeping the local machine free.

## New pod

1. Deploy from the network volume ("RunPod PyTorch" template, CUDA filter ≥ 12.8 — dream-robot's
   lock installs `torch 2.10.0+cu128`). Take `<ip>` and `<port>` from Connect → "SSH over exposed TCP".
2. **Sync code** (local, repo root). `--no-owner --no-group` is required: the volume refuses chown.
   ```bash
   rsync -az --no-owner --no-group --mkpath -e "ssh -i ~/.ssh/rgsk_github_ssh -p <port>" \
     --exclude .git --exclude .venv --exclude data --exclude scratch --exclude __pycache__ \
     --exclude '*.pt' --exclude '*.mp4' --exclude '*.log' --exclude MUJOCO_LOG.TXT \
     --exclude .pytest_cache --exclude .ruff_cache \
     ./ root@<ip>:/workspace/dream-robot/
   ```
3. **Set up** (installs uv, rsync, the EGL loader, ffmpeg; sets `MUJOCO_GL=egl`), then install:
   ```bash
   ssh -i ~/.ssh/rgsk_github_ssh -p <port> root@<ip> 'bash /workspace/dream-robot/scripts/runpod_setup.sh'
   ssh -i ~/.ssh/rgsk_github_ssh -p <port> root@<ip> 'bash -lc "cd /workspace/dream-robot && uv sync"'
   ```
4. **Smoke test** (~8 min: sim tests with headless rendering, 1 epoch BC + ACT, 2-episode eval):
   ```bash
   ssh ... 'bash -lc "bash /workspace/dream-robot/experiments/t2/scripts/pod_smoke.sh"'
   ```
   The smoke script expects the T2 dataset on the volume (step 5).

## Data

Datasets live under `data/` and are not synced with the code. Copy one up:
```bash
rsync -az --no-owner --no-group --mkpath -e "ssh -i ~/.ssh/rgsk_github_ssh -p <port>" \
  data/robosuite/<dataset>/ root@<ip>:/workspace/dream-robot/data/robosuite/<dataset>/
```

- **Record onto pod-local disk, never onto the volume.** The recorder writes a PNG per frame per
  camera before encoding; on the network volume that ran at ~21 s/episode instead of ~3.8 s.
  Record to `/root/data/...`, then `cp -r` to `/workspace/dream-robot/data/...` so it survives.
- **Train from pod-local disk** too (copy the dataset to `/root/data/...` first).

## Running a job

Launch detached on the pod, log to the volume, and pull the log from local while it runs.
Example: `experiments/t2/scripts/pod_fixed_bc.sh`.
```bash
ssh ... 'nohup bash -lc "bash /workspace/dream-robot/<script>.sh" \
    > /workspace/dream-robot/<log> 2>&1 < /dev/null &'
# local, every minute or so
rsync -az -e "ssh -i ~/.ssh/rgsk_github_ssh -p <port>" root@<ip>:/workspace/dream-robot/<log> <log>
```
Pull results with the same rsync on `experiments/<run>/`, `experiments/t2/paths/`, and each
dataset's `recording_summary.json` / `bin_choices.json`. **Terminate the pod when the job ends** — an
idle pod bills the same as a busy one.

## Pitfalls hit so far

- **Log filters must be line-buffered**, or the log stays empty until the run ends: on the pod use
  `grep --line-buffered`; locally in Claude's shell use `command grep --line-buffered` (plain `grep`
  there is a wrapper that ignores the flag). Also `PYTHONUNBUFFERED=1`.
- **`pkill -f <pattern>` over ssh kills its own ssh command** when the pattern appears in the
  command line, and the ssh exits with code 1 before killing the target. Kill by PID instead.
- **One venv per pod**: `/workspace/setup.sh` puts every project's venv at `/root/venv`. Running
  `uv sync` for llm on the same pod replaces dream-robot's.
- **The network volume is shared by every project.** Only the folder separates dream-robot from llm;
  space and `/workspace/.secrets` are shared.
- LeRobot's video encoder prints `Svt[info]` lines per episode; filter them out of logs.
