"""Copy Verne run folders from the Modal runs Volume, without their ckpt/ (mid-run checkpoints stay on the
Volume; fetch one later with `modal volume get dream-robot-runs /<run>/ckpt/step<N>.pt ...`).

    uv run python scripts/pull_runs.py 22_position_spread/t5/seed3 22_position/t5/seed4 ...
"""

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VOL, LOCAL = 'dream-robot-runs', Path('experiments/verne/runs')


def files(path: str) -> list[str]:
    # every file under a Volume folder, except ckpt/
    out = subprocess.run(
        ['modal', 'volume', 'ls', '--json', VOL, path], capture_output=True, text=True, check=True
    )
    found = []
    for e in json.loads(out.stdout):
        name = e['filename']
        if name.rstrip('/').split('/')[-1] == 'ckpt':
            continue
        found += files(name) if e['type'] == 'dir' else [name]
    return found


def get(name: str) -> str:
    # into the file's local folder; up to 3 tries (parallel modal CLI calls fail now and then)
    dest = LOCAL / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['modal', 'volume', 'get', '--force', VOL, '/' + name, str(dest.parent) + '/']
    for attempt in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return name
    raise RuntimeError(f'{name}: {r.stderr.strip()[-300:]}')


if __name__ == '__main__':
    todo = [f for run in sys.argv[1:] for f in files(run)]
    with ThreadPoolExecutor(4) as pool:
        for name in pool.map(get, todo):
            print(name)
