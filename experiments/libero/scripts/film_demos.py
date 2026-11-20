# LIBERO first look (2026-11-20): film recorded demos straight from the HDF5 files (agentview | wrist side by
# side, as recorded), and replay one demo per task in the installed sim from its first saved state with its
# recorded actions, to check the sim reproduces the demo (task success at the end).
# Runs in envs/libero/.venv:
#   MUJOCO_GL=egl envs/libero/.venv/bin/python experiments/libero/scripts/film_demos.py data/libero/libero_90/*.hdf5
import json
import os
import sys
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

# this repo's LIBERO config (envs/libero/setup.sh writes it), not ~/.libero
os.environ.setdefault('LIBERO_CONFIG_PATH', str(Path(__file__).resolve().parents[3] / 'envs/libero/config'))
OUT = Path('scratch_nov/libero/films')
N_FILM = 2  # recorded demos filmed per task
FPS = 20  # LIBERO's control rate


def describe(f) -> str:
    d = f['data']
    info = json.loads(d.attrs['problem_info'])
    lens = [d[k]['actions'].shape[0] for k in d]
    e = d['demo_0']
    obs = {k: e['obs'][k].shape for k in e['obs']}
    return (f'{info["language_instruction"]!r}: {len(d)} demos, steps {min(lens)}-{max(lens)} '
            f'(mean {np.mean(lens):.0f}), actions {e["actions"].shape[1:]}, obs {obs}')


def film_recorded(f, stem: str) -> None:
    d = f['data']
    for k in sorted(d, key=lambda s: int(s.split('_')[1]))[:N_FILM]:
        o = d[k]['obs']
        # LIBERO's images are stored upside down (OpenGL convention): flip rows
        frames = np.concatenate([o['agentview_rgb'][()][:, ::-1], o['eye_in_hand_rgb'][()][:, ::-1]], axis=2)
        iio.imwrite(OUT / f'{stem}_{k}_recorded.mp4', frames, fps=FPS)


def replay(path: Path, f, stem: str) -> str:
    # the demo's bddl scene, reset to its first recorded state, then its actions; frames from the sim
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    d = f['data']
    info = json.loads(d.attrs['problem_info'])
    # the HDF5 names the file under its old folder (libero_100); find it by name in the package's suites
    name = Path(json.loads(d.attrs['env_args'])['env_kwargs']['bddl_file_name']).name
    bddl = next(Path(get_libero_path('bddl_files')).glob(f'*/{name}'))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    e = d['demo_0']
    obs = env.set_init_state(e['states'][0])
    frames, done = [], False
    for a in e['actions'][()]:
        obs, _, done, _ = env.step(a)
        frames.append(np.concatenate([obs['agentview_image'][::-1], obs['robot0_eye_in_hand_image'][::-1]], axis=1))
    ok = bool(env.check_success())
    env.close()
    iio.imwrite(OUT / f'{stem}_demo_0_replayed_{"success" if ok else "fail"}.mp4', np.stack(frames), fps=FPS)
    return f'replay demo_0 ({info["language_instruction"]}): success {ok}'


def main(paths: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for p in map(Path, paths):
        stem = p.name.removesuffix('_demo.hdf5')
        with h5py.File(p) as f:
            print(stem, describe(f), flush=True)
            film_recorded(f, stem)
            print(' ', replay(p, f, stem), flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
