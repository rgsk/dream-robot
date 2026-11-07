"""WAM-C: the latent WAM (rough/dreamzero.ipynb WAMz) + LAM motion notes z, co-trained.

Per chunk the WAM denoises [joints | 2 frames (64 latent tokens each) | 2 z tokens | 16 actions].
z token j of chunk k = the LAM's note for the 8 steps that end at that frame.

    labelled demos (first --labelled of the notebook's 10):  loss on frames + z + actions
    video-only demos (the 80):                              loss on frames + z; joints and
                                                            actions hidden (WAMv's mask)

Each step: --batch labelled + --batch video samples. --no-z: z tokens hidden and unscored,
everything else the same: the control (co-train at --batch + --batch).
Rollout dreams frames + z + actions of chunk 1 together, executes the actions; z never leaves.
Then the sweep: env seeds 1000-1019 x n_exec 8, 16 (the notebook's).

    PYTHONUNBUFFERED=1 .venv/bin/python experiments/lam/scripts/wam_c.py --out experiments/lam/wamc_l3
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
from einops import rearrange
from torch import nn

from lam import LAM, NAME, REPO, load, split

sys.path.insert(0, str(REPO / 'rough'))

K, H, M, D = 2, 16, 4, 128
dev = 'cuda'

# ---- token layout and mask (the notebook's, chunk = joints, frames, z, actions)
layout = [('C0', 64, 0, False)]
for k in range(1, M + 1):
    layout += [(f'chunk {k}', 1 + K * 64 + K + H, k, True)]
    if k < M:
        layout += [(f'C{k}', K * 64, k, False)]
t = torch.tensor([time for _, n, time, _ in layout for _ in range(n)])
noisy = torch.tensor([nz for _, n, _, nz in layout for _ in range(n)])
key_noisy, q_noisy = noisy[None, :], noisy[:, None]
same_chunk = t[None, :] == t[:, None]
clean_before = t[None, :] <= t[:, None] - q_noisy.long()
mask = torch.where(key_noisy, q_noisy & same_chunk, clean_before)  # True = may look
edges = np.cumsum([0] + [n for _, n, _, _ in layout])
starts = edges[1::2]
at_frames = torch.tensor(np.stack([s + 1 + np.arange(K * 64) for s in starts]))  # [4, 128]
at_z = torch.tensor(np.stack([s + 1 + K * 64 + np.arange(K) for s in starts]))  # [4, 2]
at_acts = torch.tensor(np.stack([s + 1 + K * 64 + K + np.arange(H) for s in starts]))  # [4, 16]
is_robot = torch.zeros(len(t), dtype=torch.bool)
is_robot[starts] = True  # joints
is_robot[at_acts.flatten()] = True
is_z = torch.zeros(len(t), dtype=torch.bool)
is_z[at_z.flatten()] = True


def tokens(lat):
    # [..., 4, 16, 16] -> [..., 64, 16], one row per 2 x 2 square
    return rearrange(lat, '... c (h p) (w q) -> ... (h w) (p q c)', p=2, q=2)


class WAMC(nn.Module):
    def __init__(self, z_dim=8):
        super().__init__()
        self.frame_in, self.z_in, self.action_in = nn.Linear(16, D), nn.Linear(z_dim, D), nn.Linear(8, D)
        self.state_in, self.tau_in = nn.Linear(8, D), nn.Linear(1, D)
        self.pos = nn.Embedding(len(t), D)
        layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=4 * D, dropout=0.0,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, num_layers=4, enable_nested_tensor=False)
        self.frame_out, self.z_out, self.action_out = nn.Linear(D, 16), nn.Linear(D, z_dim), nn.Linear(D, 8)

    def forward(self, seen, state, nf, nz, na, tau, blocked):
        # seen [B, 9, 64, 16]; state [B, M, 8]; noisy nf [B, M, 128, 16], nz [B, M, K, z], na [B, M, H, 8]
        f = self.frame_in(seen)
        blocks = [f[:, 0]]
        for k in range(M):
            chunk = [self.state_in(state[:, k, None]), self.frame_in(nf[:, k]), self.z_in(nz[:, k]),
                     self.action_in(na[:, k])]
            blocks.append(torch.cat(chunk, 1) + self.tau_in(tau[:, k, None])[:, None])
            if k < M - 1:
                blocks.append(f[:, 1 + K * k:1 + K * (k + 1)].flatten(1, 2))
        h = self.body(torch.cat(blocks, 1) + self.pos.weight, mask=blocked)
        return self.frame_out(h[:, at_frames]), self.z_out(h[:, at_z]), self.action_out(h[:, at_acts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labelled', type=int, default=3)
    ap.add_argument('--lam', type=Path, default=REPO / 'experiments/lam/z8_det_rev')
    ap.add_argument('--no-z', action='store_true', help='control: z hidden and unscored')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=32, help='per kind: labelled and video each')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sweep-only', action='store_true')
    ap.add_argument('--n-seeds', type=int, default=20, help='env seeds 1000.. per n_exec')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- data: joints, actions, VAE latents (the notebook's), LAM notes z
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f'dream_robot/{NAME}', root=REPO / 'data/robosuite' / NAME)
    action = torch.tensor(np.asarray(ds.hf_dataset['action'], dtype=np.float32), device=dev)
    state_np, episode, top = load(NAME)
    state = torch.tensor(state_np, device=dev)
    eps = split()
    vl = torch.load(REPO / 'cache/lam/vae_lat.pt', weights_only=False)
    lat = vl['lat'].to(dev)  # [N, 4, 16, 16]
    first, lens = np.searchsorted(episode, np.arange(100)), np.bincount(episode)
    last_of = np.searchsorted(episode, episode, side='right') - 1

    z_path = args.out / 'z.pt'
    if not z_path.exists():  # z for (row, row + 8 clipped to the demo), every row
        ck = torch.load(args.lam / 'lam.pt', weights_only=False)
        lam = LAM(ck['args']['z_dim']).to(dev).eval()
        lam.load_state_dict(ck['state_dict'])
        frames = lambda r: torch.from_numpy(np.stack([top[i] for i in r])).to(dev).float() / 255
        rows = np.arange(len(episode))
        with torch.no_grad():
            z = torch.cat([lam.encode(frames(r), frames(np.minimum(r + 8, last_of[r])))[0]
                           for r in np.array_split(rows, len(rows) // 256)])
        torch.save(z.cpu(), z_path)
    Z = torch.load(z_path).to(dev)
    train_rows = ~np.isin(episode, eps['val'])
    Zn = (Z - Z[train_rows].mean(0)) / Z[train_rows].std(0)  # z in std units

    lab10 = torch.tensor(np.isin(episode, eps['labelled']), device=dev)  # stats as the notebook
    a_mean, a_std = action[lab10].mean(0), action[lab10].std(0)
    s_mean, s_std = state[lab10].mean(0), state[lab10].std(0)

    def get_batch(ids, B, rng):
        # the notebook's get_batch(to_end=True): rows of now + 8 frames, 64 action rows
        e = rng.choice(ids, B)
        last = (first[e] + lens[e] - 1)[:, None]
        now = first[e] + rng.integers(0, lens[e])
        f = np.minimum(now[:, None] + H // K * np.arange(M * K + 1), last)  # [B, 9]
        a = np.minimum(now[:, None] + np.arange(M * H), last)  # [B, 64]
        return torch.tensor(f, device=dev), torch.tensor(a, device=dev)

    net = WAMC(Z.shape[1]).to(dev)
    hide = is_robot | is_z if args.no_z else is_robot
    masks = {'robot': ~mask.to(dev) | (is_z.to(dev)[None, :] if args.no_z else False),
             'video': ~mask.to(dev) | hide.to(dev)[None, :]}

    def losses(f_rows, a_rows, labelled):
        B = len(f_rows)
        seen = tokens(lat[f_rows])
        clean_f = seen[:, 1:].reshape(B, M, K * 64, 16)
        clean_z = Zn[f_rows[:, :-1]].view(B, M, K, -1)  # note for each 8-step move into a frame
        if labelled:
            clean_a = ((action[a_rows] - a_mean) / a_std).view(B, M, H, 8)
            st = (state[a_rows[:, ::H]] - s_mean) / s_std
        else:
            clean_a, st = torch.zeros(B, M, H, 8, device=dev), torch.zeros(B, M, 8, device=dev)
        tau = torch.rand(B, M, device=dev)
        nf, nz, na = torch.randn_like(clean_f), torch.randn_like(clean_z), torch.randn_like(clean_a)
        mix = lambda n, c: (1 - tau.view(B, M, 1, 1)) * n + tau.view(B, M, 1, 1) * c
        v_f, v_z, v_a = net(seen, st, mix(nf, clean_f), mix(nz, clean_z), mix(na, clean_a), tau,
                            masks['robot' if labelled else 'video'])
        return (F.mse_loss(v_f, clean_f - nf), F.mse_loss(v_z, clean_z - nz),
                F.mse_loss(v_a, clean_a - na) if labelled else torch.zeros((), device=dev))

    lab_ids = eps['labelled'][:args.labelled]
    print(f'labelled {lab_ids} + video {len(eps["video"])} demos; no_z={args.no_z}; '
          f'{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M params, {len(t)} tokens', flush=True)

    if not args.sweep_only:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
        rng, log, t0 = np.random.default_rng(args.seed), [], time.time()
        for step in range(args.steps + 1):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                lf, lz, la = losses(*get_batch(lab_ids, args.batch, rng), True)
                vf, vz, _ = losses(*get_batch(eps['video'], args.batch, rng), False)
                total = lf + la + vf + (0 if args.no_z else lz + vz)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            if step % 500 == 0:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    torch.manual_seed(1234)
                    xf, xz, xa = losses(*get_batch(eps['val'], 32, np.random.default_rng(1)), True)
                    torch.manual_seed(args.seed * 100003 + step)
                row = dict(step=step, lab_f=lf.item(), lab_z=lz.item(), lab_a=la.item(), vid_f=vf.item(),
                           vid_z=vz.item(), val_f=xf.item(), val_z=xz.item(), val_a=xa.item())
                log.append(row)
                print(f"{step:5d}  lab f {lf:.3f} z {lz:.3f} a {la:.3f}  video f {vf:.3f} z {vz:.3f}   "
                      f"val f {xf:.3f} z {xz:.3f} a {xa:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        torch.save({'state_dict': net.state_dict(), 'args': vars(args) | {'out': str(args.out), 'lam': str(args.lam)}},
                   args.out / 'model.pt')
        (args.out / 'history.json').write_text(json.dumps(log, indent=1))
    else:
        net.load_state_dict(torch.load(args.out / 'model.pt', weights_only=False)['state_dict'])
    net.eval()

    # ---- rollout: dream chunk 1 (frames + z + actions) from the camera frame + joints
    from uwm_parts import make_sim

    vae = nn.ModuleDict(dict(enc=vae_enc(), dec=vae_dec())).to(dev)
    vae.load_state_dict(vl['weights'])
    env = make_sim()

    @torch.no_grad()
    def act(obs, steps=10):
        frame = torch.from_numpy(obs.images['top']).permute(2, 0, 1).to(dev)[None].float() / 255 * 2 - 1
        seen = torch.zeros(1, 1 + M * K, 64, 16, device=dev)
        seen[:, 0] = tokens(vae['enc'](frame).chunk(2, dim=1)[0])
        st = ((torch.from_numpy(obs.state).to(dev) - s_mean) / s_std).expand(1, M, -1)
        x_f = torch.randn(1, M, K * 64, 16, device=dev)
        x_z = torch.randn(1, M, K, Z.shape[1], device=dev)
        x_a = torch.randn(1, M, H, 8, device=dev)
        tau = torch.zeros(1, M, device=dev)
        for i in range(steps):
            tau[:, 0] = i / steps
            v_f, v_z, v_a = net(seen, st, x_f, x_z, x_a, tau, masks['robot'])
            x_f[:, 0] += v_f[:, 0] / steps
            x_z[:, 0] += v_z[:, 0] / steps
            x_a[:, 0] += v_a[:, 0] / steps
        chunk = (x_a[0, 0] * a_std + a_mean).cpu().numpy()
        chunk[:, 7] = np.clip(chunk[:, 7], 0.0, 1.0)
        return chunk

    def rollout(seed, n_exec):
        obs, plan = env.reset(seed=seed), []
        for step in range(600):
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
        res = {seed: rollout(seed, n_exec) for seed in range(1000, 1000 + args.n_seeds)}
        ok = sum(s for s, _ in res.values())
        total += ok
        fails = Counter(m for s, m in res.values() if not s)
        print(f'n_exec {n_exec:2d}: {ok:2d}/{args.n_seeds}   failures {dict(fails)}', flush=True)
        sweep[n_exec] = {'ok': ok, 'failures': dict(fails), 'per_seed': res}
    print(f'total {total}/{2 * args.n_seeds}   (3 labelled, one seed: floor 11, fine-tune all 17, co-train 13, frozen 6)')
    (args.out / 'sweep.json').write_text(json.dumps(sweep, indent=1))


# ---- the notebook's VAE (step 16), to encode the live camera frame
def down(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 4, stride=2, padding=1), nn.GroupNorm(8, o), nn.SiLU())


def up(i, o):
    return nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())


def vae_enc():
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), down(32, 64), down(64, 128), down(128, 128),
                         nn.Conv2d(128, 8, 1))


def vae_dec():
    return nn.Sequential(nn.Conv2d(4, 128, 3, padding=1), up(128, 128), up(128, 64), up(64, 32),
                         nn.Conv2d(32, 3, 3, padding=1))


if __name__ == '__main__':
    main()
