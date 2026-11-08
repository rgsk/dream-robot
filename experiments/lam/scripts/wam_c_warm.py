"""Warm-start fine-tune: trained WAM-C trunk + trained jzf head, then both trained together, action grad on.

Start: trunk = wamc_l3 (seed 0) model.pt, head = wamc_l3/heads/jzf.pt (22/40 on the frozen trunk).
Each step:
  trunk loss = wam_c.py's, unchanged (labelled: frames + z + its own actions; video: frames + z)
  head loss  = jzf's input, as in wam_c_heads.py: the trunk dreams chunk 1 (frames + z + its own actions,
               full WAM-C mask) from the now frame + joints in 10 steps; the first 9 without grad, the
               last with grad, so the head's action loss reaches the trunk. The head reads joints + the
               finished z + frames -> FM loss on the 16 actions (labelled only).
Every 1000 steps: reliance probe on 128 val rows (swap the dream for another row's, same noise:
mean squared action change arm / gripper; jzf frozen s0 = 0.385 / 0.658). Did the head stop reading it?
Then the sweeps: the head (dream -> head, as wam_c_heads.py) and the trunk's own actions (as wam_c.py).

    PYTHONUNBUFFERED=1 .venv/bin/python experiments/lam/scripts/wam_c_warm.py --out experiments/lam/wamwarm_s0
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
from wam_c import H, K, M, WAMC, is_robot, mask, tokens, vae_dec, vae_enc
from wam_c_heads import Head

sys.path.insert(0, str(REPO / 'rough'))
dev = 'cuda'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', type=Path, default=REPO / 'experiments/lam/wamc_l3')
    ap.add_argument('--labelled', type=int, default=3)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
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
    Z = torch.load(args.init / 'z.pt').to(dev)
    train_rows = ~np.isin(episode, eps['val'])
    Zn = (Z - Z[train_rows].mean(0)) / Z[train_rows].std(0)
    lab10 = torch.tensor(np.isin(episode, eps['labelled']), device=dev)
    a_mean, a_std = action[lab10].mean(0), action[lab10].std(0)
    s_mean, s_std = state[lab10].mean(0), state[lab10].std(0)

    net = nn.ModuleDict(dict(trunk=WAMC(8), head=Head(0, 1, 1))).to(dev)
    net['trunk'].load_state_dict(torch.load(args.init / 'model.pt', weights_only=False)['state_dict'])
    net['head'].load_state_dict(torch.load(args.init / 'heads/jzf.pt', weights_only=False)['state_dict'])
    masks = {'robot': ~mask.to(dev), 'video': ~mask.to(dev) | is_robot.to(dev)[None, :]}

    def get_batch(ids, B, rng):
        e = rng.choice(ids, B)
        last = (first[e] + lens[e] - 1)[:, None]
        now = first[e] + rng.integers(0, lens[e])
        f = np.minimum(now[:, None] + H // K * np.arange(M * K + 1), last)
        a = np.minimum(now[:, None] + np.arange(M * H), last)
        return torch.tensor(f, device=dev), torch.tensor(a, device=dev)

    def dream(now, st, grad_last=False, steps=10):
        # chunk 1 only, from the now frame + joints, as at rollout: now [B, 64, 16], st [B, 8]
        B = len(now)
        seen = torch.zeros(B, 1 + M * K, 64, 16, device=dev)
        seen[:, 0] = now
        s4 = st[:, None].expand(B, M, -1)
        x_f = torch.randn(B, M, K * 64, 16, device=dev)
        x_z = torch.randn(B, M, K, 8, device=dev)
        x_a = torch.randn(B, M, H, 8, device=dev)
        tau = torch.zeros(B, M, device=dev)
        with torch.no_grad():
            for i in range(steps - 1):
                tau[:, 0] = i / steps
                v_f, v_z, v_a = net['trunk'](seen, s4, x_f, x_z, x_a, tau, masks['robot'])
                x_f[:, 0] += v_f[:, 0] / steps
                x_z[:, 0] += v_z[:, 0] / steps
                x_a[:, 0] += v_a[:, 0] / steps
        tau[:, 0] = (steps - 1) / steps
        with torch.set_grad_enabled(grad_last and torch.is_grad_enabled()):
            v_f, v_z, v_a = net['trunk'](seen, s4, x_f, x_z, x_a, tau, masks['robot'])
        return x_f[:, 0] + v_f[:, 0] / steps, x_z[:, 0] + v_z[:, 0] / steps, x_a[:, 0] + v_a[:, 0] / steps

    def trunk_losses(f_rows, a_rows, labelled):  # wam_c.py's
        B = len(f_rows)
        seen = tokens(lat[f_rows])
        clean_f = seen[:, 1:].reshape(B, M, K * 64, 16)
        clean_z = Zn[f_rows[:, :-1]].view(B, M, K, -1)
        if labelled:
            clean_a = ((action[a_rows] - a_mean) / a_std).view(B, M, H, 8)
            st = (state[a_rows[:, ::H]] - s_mean) / s_std
        else:
            clean_a, st = torch.zeros(B, M, H, 8, device=dev), torch.zeros(B, M, 8, device=dev)
        tau = torch.rand(B, M, device=dev)
        nf, nz, na = torch.randn_like(clean_f), torch.randn_like(clean_z), torch.randn_like(clean_a)
        mix = lambda n, c: (1 - tau.view(B, M, 1, 1)) * n + tau.view(B, M, 1, 1) * c
        v_f, v_z, v_a = net['trunk'](seen, st, mix(nf, clean_f), mix(nz, clean_z), mix(na, clean_a), tau,
                                     masks['robot' if labelled else 'video'])
        return (F.mse_loss(v_f, clean_f - nf), F.mse_loss(v_z, clean_z - nz),
                F.mse_loss(v_a, clean_a - na) if labelled else torch.zeros((), device=dev))

    def head_loss(f_rows, a_rows):
        r = f_rows[:, 0]
        st = (state[r] - s_mean) / s_std
        hat_f, hat_z, _ = dream(tokens(lat[r]), st, grad_last=True)
        clean_a = (action[a_rows[:, :H]] - a_mean) / a_std
        na, t = torch.randn_like(clean_a), torch.rand(len(r), device=dev)
        xa = (1 - t[:, None, None]) * na + t[:, None, None] * clean_a
        return F.mse_loss(net['head'](st, None, hat_z, hat_f, xa, t), clean_a - na)

    val_rows = torch.tensor(np.flatnonzero(np.isin(episode, eps['val']))[::4], device=dev)
    last_row = torch.tensor(np.searchsorted(episode, episode, side='right') - 1, device=dev)

    @torch.no_grad()
    def reliance(rows):  # swap the dream (z + f) for another row's, same noise: action change arm / gripper
        torch.manual_seed(0)
        st = (state[rows] - s_mean) / s_std
        fz = [dream(tokens(lat[rows[s]]), st[s])[:2] for s in torch.arange(len(rows), device=dev).split(32)]
        f, z = torch.cat([a for a, _ in fz]), torch.cat([b for _, b in fz])
        perm = torch.randperm(len(rows), generator=torch.Generator().manual_seed(0)).to(dev)
        smp = lambda zz, ff: (torch.manual_seed(1), net['head'].sample(st, None, zz, ff))[1]
        own, sw = smp(z, f), smp(z[perm], f[perm])
        acts = (action[torch.minimum(rows[:, None] + torch.arange(H, device=dev), last_row[rows][:, None])] - a_mean) / a_std
        d, e = (sw - own).square(), (own - acts).square()
        return d[..., :7].mean().item(), d[..., 7].mean().item(), e[..., :7].mean().item(), e[..., 7].mean().item()

    lab_ids = eps['labelled'][:args.labelled]
    print(f'warm start from {args.init.name} + jzf, seed {args.seed}, lr {args.lr}: labelled {lab_ids} + video '
          f'{len(eps["video"])}; {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params', flush=True)

    if not args.sweep_only:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        rng, log, t0 = np.random.default_rng(args.seed), [], time.time()
        for step in range(args.steps + 1):
            if step % 1000 == 0:
                net.eval()
                ra, rg, ea, eg = reliance(val_rows[:128])
                torch.manual_seed(args.seed * 100003 + step)
                net.train()
                print(f'{step:5d}  reliance: swap dream arm {ra:.3f} gripper {rg:.3f}   (val err arm {ea:.3f} gripper {eg:.3f})', flush=True)
                log.append(dict(step=step, rel_arm=ra, rel_grip=rg, err_arm=ea, err_grip=eg))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                b = get_batch(lab_ids, args.batch, rng)
                lf, lz, la = trunk_losses(*b, True)
                lh = head_loss(*b)
                vf, vz, _ = trunk_losses(*get_batch(eps['video'], args.batch, rng), False)
                total = lf + lz + la + lh + vf + vz
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % 500 == 0:
                print(f"{step:5d}  lab f {lf:.3f} z {lz:.3f} a {la:.3f} head {lh:.3f}  video f {vf:.3f} z {vz:.3f}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
                log.append(dict(step=step, lab_f=lf.item(), lab_z=lz.item(), lab_a=la.item(), head=lh.item(),
                                vid_f=vf.item(), vid_z=vz.item()))
        net.eval()
        ra, rg, ea, eg = reliance(val_rows)
        print(f'final reliance ({len(val_rows)} val rows): swap dream arm {ra:.3f} gripper {rg:.3f}'
              f'   (val err arm {ea:.3f} gripper {eg:.3f})', flush=True)
        log.append(dict(step='final', rel_arm=ra, rel_grip=rg, err_arm=ea, err_grip=eg))
        torch.save({'state_dict': net.state_dict(), 'args': vars(args) | {'out': str(args.out), 'init': str(args.init)}},
                   args.out / 'model.pt')
        (args.out / 'history.json').write_text(json.dumps(log, indent=1))
    else:
        net.load_state_dict(torch.load(args.out / 'model.pt', weights_only=False)['state_dict'])
    net.eval()
    torch.cuda.empty_cache()

    from uwm_parts import make_sim
    vae = nn.ModuleDict(dict(enc=vae_enc(), dec=vae_dec())).to(dev).eval()
    vae.load_state_dict(vl['weights'])
    env = make_sim()

    @torch.no_grad()
    def act(obs, who):
        frame = torch.from_numpy(obs.images['top']).permute(2, 0, 1).to(dev)[None].float() / 255 * 2 - 1
        st = ((torch.from_numpy(obs.state).to(dev) - s_mean) / s_std)[None]
        f, z, a = dream(tokens(vae['enc'](frame).chunk(2, dim=1)[0]), st)
        a = net['head'].sample(st, None, z, f) if who == 'head' else a
        chunk = (a[0] * a_std + a_mean).cpu().numpy()
        chunk[:, 7] = np.clip(chunk[:, 7], 0.0, 1.0)
        return chunk

    def rollout(seed, n_exec, who):
        obs, plan = env.reset(seed=seed), []
        for _ in range(600):
            if not plan:
                plan = list(act(obs, who)[:n_exec])
            r = env.step(plan.pop(0))
            obs = r.observation
            if r.terminated or r.truncated:
                break
        return bool(r.success), str(r.failure_mode)

    results = {}
    for who in ['head', 'trunk']:
        sweep, total = {}, 0
        for n_exec in [8, 16]:
            torch.manual_seed(n_exec)
            res = {seed: rollout(seed, n_exec, who) for seed in range(1000, 1020)}
            ok = sum(s for s, _ in res.values())
            total += ok
            fails = Counter(m for s, m in res.values() if not s)
            print(f'{who} n_exec {n_exec:2d}: {ok:2d}/20   failures {dict(fails)}', flush=True)
            sweep[n_exec] = {'ok': ok, 'failures': dict(fails), 'per_seed': res}
        sweep['total'] = total
        results[who] = sweep
        print(f'{who} total {total}/40   (seed 0 refs: jzf on frozen WAM-C 22, WAM-C own 32, b-clean 15)', flush=True)
    (args.out / 'sweep.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
