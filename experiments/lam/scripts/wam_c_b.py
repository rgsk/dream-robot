"""Option b: WAM-C trunk dreams (frames + z), a jzf head reads the DREAM VALUES -> actions, trained together.

Trunk = wam_c.py's WAMC with its action tokens hidden (no one looks at them, no loss on them).
Head  = wam_c_heads.Head(now=0, z=1, f=1): joints + the trunk's clean guess of the chunk's dream
        x_hat = x + (1 - tau) v  (values, not hidden states, so the now frame can't leak in)
        + noisy actions at the same tau -> velocity. All 4 chunks, labelled samples only.
Loss  = trunk frames + z (labelled + video) + head actions (labelled).
--stop-grad: x_hat detached (b-stop, the control); default lets the action loss reach the trunk (b-grad).
Rollout: one 10-step loop, trunk refines the dream while the head refines the actions from it.
--dream-steps S (the clean end of the spectrum): the head instead reads the FINISHED dream. Training:
the trunk dreams all 4 chunks from noise in S steps (first S-1 without grad, the last with grad, so the
action loss still reaches the trunk); actions at a random tau read it. Rollout: 10 dream steps, then
10 action steps on the fixed dream.

    PYTHONUNBUFFERED=1 .venv/bin/python experiments/lam/scripts/wam_c_b.py --out experiments/lam/wamb_grad_s0
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lam import NAME, REPO, load, split
from wam_c import H, K, M, WAMC, at_acts, is_robot, mask, tokens, vae_dec, vae_enc
from wam_c_heads import Head

sys.path.insert(0, str(REPO / 'rough'))
dev = 'cuda'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labelled', type=int, default=3)
    ap.add_argument('--z', type=Path, default=REPO / 'experiments/lam/wamc_l3/z.pt')
    ap.add_argument('--stop-grad', action='store_true')
    ap.add_argument('--dream-steps', type=int, default=0, help='0 = dream at the action tau')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sweep-only', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f'dream_robot/{NAME}', root=REPO / 'data/robosuite' / NAME)
    action = torch.tensor(np.asarray(ds.hf_dataset['action'], dtype=np.float32), device=dev)
    state_np, episode, _ = load(NAME)
    state = torch.tensor(state_np, device=dev)
    eps = split()
    vl = torch.load(REPO / 'cache/lam/vae_lat.pt', weights_only=False)
    lat = vl['lat'].to(dev)
    first, lens = np.searchsorted(episode, np.arange(100)), np.bincount(episode)
    Z = torch.load(args.z).to(dev)  # same LAM z as wam_c (z8_det_rev)
    train_rows = ~np.isin(episode, eps['val'])
    Zn = (Z - Z[train_rows].mean(0)) / Z[train_rows].std(0)
    lab10 = torch.tensor(np.isin(episode, eps['labelled']), device=dev)
    a_mean, a_std = action[lab10].mean(0), action[lab10].std(0)
    s_mean, s_std = state[lab10].mean(0), state[lab10].std(0)

    def get_batch(ids, B, rng):
        e = rng.choice(ids, B)
        last = (first[e] + lens[e] - 1)[:, None]
        now = first[e] + rng.integers(0, lens[e])
        f = np.minimum(now[:, None] + H // K * np.arange(M * K + 1), last)
        a = np.minimum(now[:, None] + np.arange(M * H), last)
        return torch.tensor(f, device=dev), torch.tensor(a, device=dev)

    net = nn.ModuleDict(dict(trunk=WAMC(8), head=Head(0, 1, 1))).to(dev)
    is_act = torch.zeros(len(mask), dtype=torch.bool)
    is_act[at_acts.flatten()] = True
    masks = {'robot': ~mask.to(dev) | is_act.to(dev)[None, :],  # trunk: action tokens hidden
             'video': ~mask.to(dev) | is_robot.to(dev)[None, :]}

    def losses(f_rows, a_rows, labelled):
        B = len(f_rows)
        seen = tokens(lat[f_rows])
        clean_f = seen[:, 1:].reshape(B, M, K * 64, 16)
        clean_z = Zn[f_rows[:, :-1]].view(B, M, K, -1)
        st = (state[a_rows[:, ::H]] - s_mean) / s_std if labelled else torch.zeros(B, M, 8, device=dev)
        tau = torch.rand(B, M, device=dev)
        nf, nz = torch.randn_like(clean_f), torch.randn_like(clean_z)
        mix = lambda n, c: (1 - tau.view(B, M, 1, 1)) * n + tau.view(B, M, 1, 1) * c
        xf, xz = mix(nf, clean_f), mix(nz, clean_z)
        v_f, v_z, _ = net['trunk'](seen, st, xf, xz, torch.zeros(B, M, H, 8, device=dev), tau,
                                   masks['robot' if labelled else 'video'])
        lf, lz = F.mse_loss(v_f, clean_f - nf), F.mse_loss(v_z, clean_z - nz)
        if not labelled:
            return lf, lz, torch.zeros((), device=dev)
        if args.dream_steps:  # finished dream from noise, grad through the last step only
            S, no_a = args.dream_steps, torch.zeros(B, M, H, 8, device=dev)
            df, dz = torch.randn_like(clean_f), torch.randn_like(clean_z)
            with torch.no_grad():
                for i in range(S - 1):
                    wf, wz, _ = net['trunk'](seen, st, df, dz, no_a, torch.full((B, M), i / S, device=dev), masks['robot'])
                    df, dz = df + wf / S, dz + wz / S
            wf, wz, _ = net['trunk'](seen, st, df, dz, no_a, torch.full((B, M), (S - 1) / S, device=dev), masks['robot'])
            hat_f, hat_z = df + wf / S, dz + wz / S
        else:
            keep = 1 - tau.view(B, M, 1, 1)
            hat_f, hat_z = xf + keep * v_f, xz + keep * v_z  # trunk's guess of the clean dream
        if args.stop_grad:
            hat_f, hat_z = hat_f.detach(), hat_z.detach()
        clean_a = ((action[a_rows] - a_mean) / a_std).view(B * M, H, 8)
        na = torch.randn_like(clean_a)
        t = tau.flatten()
        xa = (1 - t[:, None, None]) * na + t[:, None, None] * clean_a
        v_a = net['head'](st.flatten(0, 1), None, hat_z.flatten(0, 1), hat_f.flatten(0, 1), xa, t)
        return lf, lz, F.mse_loss(v_a, clean_a - na)

    lab_ids = eps['labelled'][:args.labelled]
    print(f'b-{"stop" if args.stop_grad else "grad"} dream_steps {args.dream_steps} seed {args.seed}: labelled {lab_ids} + video {len(eps["video"])}; '
          f'{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params', flush=True)

    if not args.sweep_only:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        rng, log, t0 = np.random.default_rng(args.seed), [], time.time()
        for step in range(args.steps + 1):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lf, lz, la = losses(*get_batch(lab_ids, args.batch, rng), True)
                vf, vz, _ = losses(*get_batch(eps['video'], args.batch, rng), False)
                total = lf + lz + la + vf + vz
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % 500 == 0:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    torch.manual_seed(1234)
                    xf, xz, xa = losses(*get_batch(eps['val'], 32, np.random.default_rng(1)), True)
                    torch.manual_seed(args.seed * 100003 + step)
                log.append(dict(step=step, lab_f=lf.item(), lab_z=lz.item(), lab_a=la.item(), vid_f=vf.item(),
                                vid_z=vz.item(), val_f=xf.item(), val_z=xz.item(), val_a=xa.item()))
                print(f"{step:5d}  lab f {lf:.3f} z {lz:.3f} a {la:.3f}  video f {vf:.3f} z {vz:.3f}   "
                      f"val f {xf:.3f} z {xz:.3f} a {xa:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        torch.save({'state_dict': net.state_dict(), 'args': vars(args) | {'out': str(args.out), 'z': str(args.z)}},
                   args.out / 'model.pt')
        (args.out / 'history.json').write_text(json.dumps(log, indent=1))
    else:
        net.load_state_dict(torch.load(args.out / 'model.pt', weights_only=False)['state_dict'])
    net.eval()

    from uwm_parts import make_sim
    vae = nn.ModuleDict(dict(enc=vae_enc(), dec=vae_dec())).to(dev).eval()
    vae.load_state_dict(vl['weights'])
    env = make_sim()

    @torch.no_grad()
    def act(obs, steps=10):
        frame = torch.from_numpy(obs.images['top']).permute(2, 0, 1).to(dev)[None].float() / 255 * 2 - 1
        seen = torch.zeros(1, 1 + M * K, 64, 16, device=dev)
        seen[:, 0] = tokens(vae['enc'](frame).chunk(2, dim=1)[0])
        st = ((torch.from_numpy(obs.state).to(dev) - s_mean) / s_std).expand(1, M, -1)
        x_f = torch.randn(1, M, K * 64, 16, device=dev)
        x_z = torch.randn(1, M, K, 8, device=dev)
        x_a = torch.randn(1, H, 8, device=dev)
        tau = torch.zeros(1, M, device=dev)
        if args.dream_steps:  # dream first, then actions on the fixed dream
            for i in range(steps):
                tau[:, 0] = i / steps
                v_f, v_z, _ = net['trunk'](seen, st, x_f, x_z, torch.zeros(1, M, H, 8, device=dev), tau, masks['robot'])
                x_f[:, 0] += v_f[:, 0] / steps
                x_z[:, 0] += v_z[:, 0] / steps
            for i in range(steps):
                x_a += net['head'](st[:, 0], None, x_z[:, 0], x_f[:, 0], x_a, torch.full((1,), i / steps, device=dev)) / steps
        for i in range(0 if args.dream_steps else steps):
            tau[:, 0] = i / steps
            v_f, v_z, _ = net['trunk'](seen, st, x_f, x_z, torch.zeros(1, M, H, 8, device=dev), tau, masks['robot'])
            keep = 1 - i / steps
            v_a = net['head'](st[:, 0], None, x_z[:, 0] + keep * v_z[:, 0], x_f[:, 0] + keep * v_f[:, 0], x_a, tau[:, 0])
            x_f[:, 0] += v_f[:, 0] / steps
            x_z[:, 0] += v_z[:, 0] / steps
            x_a += v_a / steps
        chunk = (x_a[0] * a_std + a_mean).cpu().numpy()
        chunk[:, 7] = np.clip(chunk[:, 7], 0.0, 1.0)
        return chunk

    def rollout(seed, n_exec):
        obs, plan = env.reset(seed=seed), []
        for _ in range(600):
            if not plan:
                plan = list(act(obs)[:n_exec])
            r = env.step(plan.pop(0))
            obs = r.observation
            if r.terminated or r.truncated:
                break
        return bool(r.success), str(r.failure_mode)

    sweep, total = {}, 0
    for n_exec in [8, 16]:
        torch.manual_seed(n_exec)
        res = {seed: rollout(seed, n_exec) for seed in range(1000, 1020)}
        ok = sum(s for s, _ in res.values())
        total += ok
        fails = Counter(m for s, m in res.values() if not s)
        print(f'n_exec {n_exec:2d}: {ok:2d}/20   failures {dict(fails)}', flush=True)
        sweep[n_exec] = {'ok': ok, 'failures': dict(fails), 'per_seed': res}
    print(f'total {total}/40   (jzf head on frozen WAM-C: 22/17/16; WAM-C own 32/16/13; no-z 15/16/17)', flush=True)
    (args.out / 'sweep.json').write_text(json.dumps(sweep, indent=1))


if __name__ == '__main__':
    main()
