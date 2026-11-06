"""Roll out a camera-only checkpoint (uwm_cam.py, or uwm_dino.py) on rung 0's kitchen, unseen seeds.

Runs in envs/robocasa/.venv. Every n_exec steps: the last wrist frames (t-12, -8, -4, t) ->
a 16-step plan of arm changes + gripper; targets = q[t] + change, the first n_exec played.
The network sees only the wrist camera; q[t] is used to turn its changes back into targets.
Success = apple 10 cm above its start at the end.

Seeds run in --workers processes (each its own kitchen + net); episodes are independent and
3 workers on the 4060 take 56 s for 20 seeds (serial: ~4-5 min); see --workers. Each episode seeds its
plan noise with its kitchen seed, so any seed reproduces alone, in any order or worker.

    MUJOCO_GL=egl envs/robocasa/.venv/bin/python experiments/pick_shapes/scripts/uwm_cam_eval.py \
        experiments/pick_shapes/uwm_cam/l25_v0/step10000.pt --seeds 2000-2019
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

HORIZON, VIDEOS = 200, 5


def run_seeds(checkpoint: str, seeds: list[int], video_seeds: set[int], n_exec: int, out: str,
              device: str, pool_name: str = 'apple0') -> list[dict]:
    # one worker: its own kitchen and net; out: one result dict per seed
    import imageio
    import torch

    from rung0_scripted_pick import SUCCESS_LIFT, PickKitchen, mesh_for, pool
    from uwm_cam import PAST, UWMCam, to_128

    torch.set_num_threads(2)                           # workers share the cores
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    if blob.get('kind') in ('dino', 'dino_target'):    # uwm_dino(_target).py: frozen DINOv2 features in
        from dino_cache import Dino, pool8
        from uwm_dino import DinoAct
        if blob['kind'] == 'dino_target':              # future is an internal target: plans like A
            from uwm_dino_target import DinoTarget
            net = DinoTarget(torch.zeros(8), torch.ones(8), torch.zeros(384), torch.ones(384)).to(device).eval()
        else:
            net = DinoAct(torch.zeros(8), torch.ones(8),
                          mixed=blob['args'].get('features') == 'dino+vjepa_last').to(device).eval()
        dino = Dino(device)
        feats_kind = blob['args'].get('features', 'dino')
        if feats_kind == 'dino':
            # what one frame is remembered as: its 8x8 x 384 features [64, 384]
            see = lambda im: torch.from_numpy(pool8(dino(im[None])[0]).reshape(64, -1))
            to_input = lambda seen, now: torch.stack([seen[max(0, now + o)] for o in PAST])[None].to(device)
        elif feats_kind == 'dino+vjepa_last':          # DINOv2 per frame + V-JEPA newest moment per plan
            from vjepa_cache import CLIP, VJepa, project
            vj = VJepa(device)
            pca = np.load(Path(__file__).resolve().parents[3] / 'data/robocasa' / f"{blob['args']['data']}_vjepa.npz")
            mean, comp = pca['pca_mean'], pca['pca_comp']
            see = lambda im: (im, torch.from_numpy(pool8(dino(im[None])[0]).reshape(64, -1)))
            def to_input(seen, now):
                d4 = torch.stack([seen[max(0, now + o)][1] for o in PAST])                # [4, 64, 384]
                clip = np.stack([seen[max(0, now + o)][0] for o in CLIP])[None]
                v = torch.from_numpy(project(vj(clip), mean, comp))[0, -1:].float()       # [1, 64, 384]
                return torch.cat([d4, v])[None].to(device)
        else:                                          # vjepa(_last): one clip t-14..t per plan
            from vjepa_cache import CLIP, VJepa, project
            del dino
            vj = VJepa(device)
            pca = np.load(Path(__file__).resolve().parents[3] / 'data/robocasa' / f"{blob['args']['data']}_vjepa.npz")
            mean, comp = pca['pca_mean'], pca['pca_comp']
            see = lambda im: im                        # keep raw frames
            def to_input(seen, now):
                clip = np.stack([seen[max(0, now + o)] for o in CLIP])[None]           # [1, 8, 256, 256, 3]
                f = torch.from_numpy(project(vj(clip), mean, comp)).float().to(device)  # [1, 4, 64, 384]
                return f[:, -1:] if feats_kind == 'vjepa_last' else f
        future = lambda x: None
        if blob['args'].get('predictor'):              # arm B: frozen predictor's t+16 guess, fp32 as trained
            from dino_predict import Predictor
            pred = Predictor(torch.zeros(384), torch.ones(384)).to(device).eval()
            pred.load_state_dict(torch.load(blob['args']['predictor'], map_location=device,
                                            weights_only=False)['state_dict'])
            future = pred.predict
    else:
        future = lambda x: None
        net = UWMCam(torch.zeros(8), torch.ones(8)).to(device).eval()
        see = lambda im: to_128(im[None])[0]
        to_input = lambda seen, now: torch.from_numpy(np.stack([seen[max(0, now + o)] for o in PAST]))[None].to(device).float() / 255
    net.load_state_dict(blob['state_dict'])
    env = PickKitchen()
    meshes = pool(pool_name)
    results = []
    for seed in seeds:
        torch.manual_seed(seed)                        # plan noise tied to the episode
        t0 = time.time()
        env.obj_xml = mesh_for(meshes, seed)           # apple0: always the rung 0 apple
        try:
            obs = env.reset(seed=seed)
        except ValueError as e:                        # RoboCasa couldn't place this mesh: log, don't lose the worker
            results.append({'seed': seed, 'mesh': Path(env.obj_xml).parent.name, 'ok': None, 'error': repr(e)})
            print(f'live SKIP {results[-1]}', flush=True)
            continue
        t1 = time.time()
        start = env.obj_pos()
        seen = [see(obs.images['wrist'])]            # every wrist frame since reset, newest last
        plan, frames = [], []
        for _ in range(HORIZON):
            if not plan:
                now = len(seen) - 1
                x = to_input(seen, now)
                fut = future(x)
                with torch.autocast(device, dtype=torch.bfloat16):
                    chunk = (net.plan(x, fut) if fut is not None else net.plan(x))[0].float().cpu().numpy()                 # [16, 8]
                q = obs.state[:7]
                targets = np.concatenate([q + chunk[:, :7], np.clip(chunk[:, 7:], 0, 1)], 1)
                plan = list(targets[:n_exec].astype(np.float32))
            obs = env.step(plan.pop(0)).observation
            seen.append(see(obs.images['wrist']))
            if seed in video_seeds:
                frames.append(np.concatenate([env.frame('scene', (256, 256)), env.frame('wrist', (256, 256))], 1))
        lifted = float(env.obj_pos()[2] - start[2])
        results.append({'seed': seed, 'mesh': Path(env.obj_xml).parent.name, 'ok': lifted > SUCCESS_LIFT,
                        'lifted_m': round(lifted, 3),
                        'build_s': round(t1 - t0, 1), 'run_s': round(time.time() - t1, 1)})
        # progress as it happens (workers finish out of order); the sorted summary prints at the end
        print(f"live {'OK  ' if results[-1]['ok'] else 'FAIL'} {results[-1]}", flush=True)
        if frames:
            imageio.mimsave(Path(out) / f'seed{seed}.mp4', frames, fps=20)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint', type=Path)
    ap.add_argument('--seeds', default='2000-2019')
    ap.add_argument('--n-exec', type=int, default=8)
    ap.add_argument('--pool', default='apple0', help='objects: apple0 (rung 0 apple), fruit, unseen, rung2')
    ap.add_argument('--videos', type=int, default=VIDEOS, help='first N seeds get a video')
    # Measured 2026-11-05 (20 seeds): net on CPU 43 s per rollout (25 plans x 10 denoising passes);
    # on GPU 3.5 s. Each GPU worker holds ~1.34 GB (CUDA context + EGL), so the 8 GB 4060 fits 3
    # (4 and 5 ran out of memory). 3 workers: 56 s for 20 seeds; the old serial script ~4-5 min.
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--device', default='cuda', help='where the net runs (cpu is ~10x slower per rollout)')
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split('-'))
    seeds = list(range(lo, hi + 1))
    out = args.checkpoint.with_suffix('')            # step10000.pt -> step10000/ (step10000_fruit/ ...)
    if args.pool != 'apple0':
        out = out.with_name(f'{out.name}_{args.pool}')
    out.mkdir(exist_ok=True)
    video_seeds = set(seeds[:args.videos])
    n = max(1, min(args.workers, len(seeds)))
    shares = [seeds[i::n] for i in range(n)]         # interleaved: each worker gets a similar mix
    job = [(str(args.checkpoint), s, video_seeds, args.n_exec, str(out), args.device, args.pool) for s in shares]
    with mp.get_context('spawn').Pool(n) as pool:    # spawn: CUDA and MuJoCo don't survive fork
        results = sorted((r for part in pool.starmap(run_seeds, job) for r in part), key=lambda r: r['seed'])
    for r in results:
        print('SKIP' if r['ok'] is None else 'OK ' if r['ok'] else 'FAIL', r, flush=True)
    skipped = [r for r in results if r['ok'] is None]
    results = [r for r in results if r['ok'] is not None]
    wins = sum(r['ok'] for r in results)
    print(f'{wins}/{len(results)} lifted' + (f' ({len(skipped)} seeds skipped: mesh not placeable)' if skipped else ''))
    # per seed range, so a 60-seed re-score doesn't overwrite the 20-seed one
    (out / f'results_{args.seeds}.json').write_text(json.dumps({'wins': wins, 'n': len(results), 'episodes': results}, indent=1))


if __name__ == '__main__':
    main()
