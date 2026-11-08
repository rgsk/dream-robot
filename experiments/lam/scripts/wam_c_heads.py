"""8 action heads on top of a trained WAM-C, each reading a different slice of what the WAM dreams.

The WAM-C trunk is frozen (heads are gradient-cut, so this equals training them inside the WAM run).
For chunk 1 the WAM dreams 2 frames (t+8, t+16) and 2 z notes from the now frame + joints. Each head
is a small flow-matching transformer -> 16 actions, trained on the 3 labelled demos, fed the WAM's
DREAMS (not the real future), so train matches test.

    head   joints  now  z  frames
    j        x                       blind
    jn       x      x                bc
    jz       x         x
    jnz      x      x  x
    jf       x            x
    jnf      x      x     x
    jzf      x         x  x
    jnzf     x      x  x  x
    (row 9 = the WAM's own action tokens: sweep.json in the run dir)

--now-drop 0.5: blank the now frame in half the training samples (saved as e.g. jnzf_d50).
Then the sim sweep per head (env seeds 1000-1019 x n_exec 8, 16, as wam_c.py).

    PYTHONUNBUFFERED=1 .venv/bin/python experiments/lam/scripts/wam_c_heads.py experiments/lam/wamc_l3
"""

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
from wam_c import H, K, M, WAMC, mask, tokens, vae_dec, vae_enc

sys.path.insert(0, str(REPO / 'rough'))
dev = 'cuda'
D = 128
HEADS = {'j': (0, 0, 0), 'jn': (1, 0, 0), 'jz': (0, 1, 0), 'jnz': (1, 1, 0),
         'jf': (0, 0, 1), 'jnf': (1, 0, 1), 'jzf': (0, 1, 1), 'jnzf': (1, 1, 1)}  # (now, z, frames)


class Head(nn.Module):
    # tokens: [joints | now 64 | z 2 | frames 128 | 16 noisy actions], only the ones this head reads
    def __init__(self, now, z, f, layers=3):
        super().__init__()
        self.use = (now, z, f)
        self.state_in, self.frame_in, self.z_in = nn.Linear(8, D), nn.Linear(16, D), nn.Linear(8, D)
        self.action_in, self.tau_in = nn.Linear(8, D), nn.Linear(1, D)
        self.pos = nn.Embedding(1 + 64 * now + K * z + K * 64 * f + H, D)
        layer = nn.TransformerEncoderLayer(D, nhead=4, dim_feedforward=4 * D, dropout=0.0,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.out = nn.Linear(D, 8)

    def forward(self, st, now, z, f, x_a, tau):
        # st [B, 8]; now [B, 64, 16]; z [B, 2, 8]; f [B, 128, 16]; x_a [B, 16, 8]; tau [B]
        parts = [self.state_in(st)[:, None]]
        if self.use[0]:
            parts.append(self.frame_in(now))
        if self.use[1]:
            parts.append(self.z_in(z))
        if self.use[2]:
            parts.append(self.frame_in(f))
        parts.append(self.action_in(x_a) + self.tau_in(tau[:, None])[:, None])
        return self.out(self.body(torch.cat(parts, 1) + self.pos.weight)[:, -H:])

    @torch.no_grad()
    def sample(self, st, now, z, f, steps=10):
        x = torch.randn(len(st), H, 8, device=dev)
        for i in range(steps):
            x += self(st, now, z, f, x, torch.full((len(st),), i / steps, device=dev)) / steps
        return x


def main():
    run = Path(sys.argv[1])
    only = sys.argv[sys.argv.index('--heads') + 1].split(',') if '--heads' in sys.argv else list(HEADS)
    steps = int(sys.argv[sys.argv.index('--steps') + 1]) if '--steps' in sys.argv else 6000
    # --now-drop p: blank the now frame for a fraction p of training samples (test: always shown)
    drop = float(sys.argv[sys.argv.index('--now-drop') + 1]) if '--now-drop' in sys.argv else 0.0
    tag = f'_d{round(drop * 100)}' if drop else ''
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f'dream_robot/{NAME}', root=REPO / 'data/robosuite' / NAME)
    action = torch.tensor(np.asarray(ds.hf_dataset['action'], dtype=np.float32), device=dev)
    state_np, episode, _ = load(NAME)
    state = torch.tensor(state_np, device=dev)
    eps = split()
    lab10 = torch.tensor(np.isin(episode, eps['labelled']), device=dev)
    a_mean, a_std = action[lab10].mean(0), action[lab10].std(0)
    s_mean, s_std = state[lab10].mean(0), state[lab10].std(0)
    vl = torch.load(REPO / 'cache/lam/vae_lat.pt', weights_only=False)
    lat = vl['lat'].to(dev)
    last = torch.tensor(np.searchsorted(episode, episode, side='right') - 1, device=dev)

    net = WAMC(8).to(dev).eval()
    net.load_state_dict(torch.load(run / 'model.pt', weights_only=False)['state_dict'])
    blocked = ~mask.to(dev)

    @torch.no_grad()
    def dream(now, st, steps=10):  # now tokens [B, 64, 16], st [B, 8] -> frames [B, 128, 16], z [B, 2, 8]
        outs = []
        for sl in torch.arange(len(now), device=dev).split(32):
            B = len(sl)
            seen = torch.zeros(B, 1 + M * K, 64, 16, device=dev)
            seen[:, 0] = now[sl]
            x_f = torch.randn(B, M, K * 64, 16, device=dev)
            x_z = torch.randn(B, M, K, 8, device=dev)
            x_a = torch.randn(B, M, H, 8, device=dev)
            tau = torch.zeros(B, M, device=dev)
            for i in range(steps):
                tau[:, 0] = i / steps
                v_f, v_z, v_a = net(seen, st[sl][:, None].expand(B, M, -1), x_f, x_z, x_a, tau, blocked)
                x_f[:, 0] += v_f[:, 0] / steps
                x_z[:, 0] += v_z[:, 0] / steps
                x_a[:, 0] += v_a[:, 0] / steps
            outs.append((x_f[:, 0], x_z[:, 0]))
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    def rows_data(r, n_dreams):  # -> dict of tensors, each row repeated n_dreams times (fresh dream each)
        r = r.repeat(n_dreams)
        st = (state[r] - s_mean) / s_std
        now = tokens(lat[r])
        f, z = dream(now, st)
        acts = (action[torch.minimum(r[:, None] + torch.arange(H, device=dev), last[r][:, None])] - a_mean) / a_std
        return dict(st=st, now=now, z=z, f=f, a=acts)

    torch.manual_seed(0)
    t0 = time.time()
    lab_rows = torch.tensor(np.flatnonzero(np.isin(episode, eps['labelled'][:3])), device=dev)
    train = rows_data(lab_rows, 4)
    val = rows_data(torch.tensor(np.flatnonzero(np.isin(episode, eps['val']))[::4], device=dev), 1)
    print(f'{run.name}: {len(lab_rows)} labelled rows x 4 dreams, {len(val["a"])} val rows ({time.time() - t0:.0f}s)', flush=True)

    out_dir = run / 'heads'
    out_dir.mkdir(exist_ok=True)
    heads = {}
    for name in only:
        torch.manual_seed(0)
        head = Head(*HEADS[name]).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=1e-5)
        for step in range(1, steps + 1):
            i = torch.randint(0, len(train['a']), (64,), device=dev)
            b = {k: v[i] for k, v in train.items()}
            if drop:
                b['now'] = b['now'] * (torch.rand(64, 1, 1, device=dev) >= drop)
            tau = torch.rand(64, device=dev)
            noise = torch.randn_like(b['a'])
            x = (1 - tau[:, None, None]) * noise + tau[:, None, None] * b['a']
            loss = F.mse_loss(head(b['st'], b['now'], b['z'], b['f'], x, tau), b['a'] - noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        torch.manual_seed(0)
        err = torch.cat([(head.sample(*(val[k][s] for k in ['st', 'now', 'z', 'f'])) - val['a'][s]).square()
                         for s in torch.arange(len(val['a']), device=dev).split(256)])
        print(f'  head {name + tag:5s} train fm {loss.item():.3f}   val (offline, dreamed) arm {err[..., :7].mean():.3f}'
              f'  gripper {err[..., 7].mean():.3f}  ({time.time() - t0:.0f}s)', flush=True)
        torch.save({'state_dict': head.state_dict(), 'name': name}, out_dir / f'{name}{tag}.pt')
        heads[name + tag] = head

    from uwm_parts import make_sim
    vae = nn.ModuleDict(dict(enc=vae_enc(), dec=vae_dec())).to(dev).eval()
    vae.load_state_dict(vl['weights'])
    env = make_sim()
    res_path = run / 'sweep_heads.json'
    results = json.loads(res_path.read_text()) if res_path.exists() else {}

    for name, head in heads.items():
        @torch.no_grad()
        def act(obs):
            frame = torch.from_numpy(obs.images['top']).permute(2, 0, 1).to(dev)[None].float() / 255 * 2 - 1
            now = tokens(vae['enc'](frame).chunk(2, dim=1)[0])
            st = ((torch.from_numpy(obs.state).to(dev) - s_mean) / s_std)[None]
            f, z = dream(now, st) if any(head.use[1:]) else (None, None)
            chunk = (head.sample(st, now, z, f)[0] * a_std + a_mean).cpu().numpy()
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
            sweep[n_exec] = {'ok': ok, 'failures': dict(Counter(m for s, m in res.values() if not s)), 'per_seed': res}
        sweep['total'] = total
        results[name] = sweep
        res_path.write_text(json.dumps(results, indent=1))
        print(f'{run.name} head {name:5s}: n_exec 8 {sweep[8]["ok"]:2d}/20  16 {sweep[16]["ok"]:2d}/20  total {total}/40'
              f'  ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
