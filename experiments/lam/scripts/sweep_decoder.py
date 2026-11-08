"""Sim sweep where the actions come only from a decoder (--kind z | f | bc, default z).
z: the dreamed z; f: now frame + the 2 dreamed frames; bc: now frame only (no WAM used).
Default z: WAM-C dreams chunk 1 (frames + z +
its own actions, 10 steps), its 2 z notes + the joints go into decoders/z.pt -> 16 actions.
Same sweep as wam_c.py: env seeds 1000-1019 x n_exec 8, 16.

    .venv/bin/python experiments/lam/scripts/sweep_decoder.py experiments/lam/wamc_l3
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from decoder import Decoder
from lam import NAME, REPO, load, split
from wam_c import H, K, M, WAMC, mask, tokens, vae_dec, vae_enc

sys.path.insert(0, str(REPO / 'rough'))
dev = 'cuda'


def main():
    run = Path(sys.argv[1])
    kind = sys.argv[sys.argv.index('--kind') + 1] if '--kind' in sys.argv else 'z'
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
    vae = torch.nn.ModuleDict(dict(enc=vae_enc(), dec=vae_dec())).to(dev).eval()
    vae.load_state_dict(vl['weights'])
    net = WAMC(8).to(dev).eval()
    net.load_state_dict(torch.load(run / 'model.pt', weights_only=False)['state_dict'])
    dec = Decoder(kind).to(dev).eval()
    dec.load_state_dict(torch.load(REPO / 'experiments/lam/decoders' / f'{kind}.pt', weights_only=False)['state_dict'])
    from einops import rearrange
    to_lat = lambda t: rearrange(t, 'b (k h w) (p q c) -> b k c (h p) (w q)', k=K, h=8, p=2, q=2)
    blocked = ~mask.to(dev)

    @torch.no_grad()
    def act(obs, steps=10):
        frame = torch.from_numpy(obs.images['top']).permute(2, 0, 1).to(dev)[None].float() / 255 * 2 - 1
        seen = torch.zeros(1, 1 + M * K, 64, 16, device=dev)
        now = vae['enc'](frame).chunk(2, dim=1)[0]  # [1, 4, 16, 16]
        seen[:, 0] = tokens(now)
        st = (torch.from_numpy(obs.state).to(dev) - s_mean) / s_std
        if kind == 'bc':
            chunk = (dec(st[None], frames=now[:, None])[0] * a_std + a_mean).cpu().numpy()
            chunk[:, 7] = np.clip(chunk[:, 7], 0.0, 1.0)
            return chunk
        x_f = torch.randn(1, M, K * 64, 16, device=dev)
        x_z = torch.randn(1, M, K, 8, device=dev)
        x_a = torch.randn(1, M, H, 8, device=dev)
        tau = torch.zeros(1, M, device=dev)
        for i in range(steps):
            tau[:, 0] = i / steps
            v_f, v_z, v_a = net(seen, st.expand(1, M, -1), x_f, x_z, x_a, tau, blocked)
            x_f[:, 0] += v_f[:, 0] / steps
            x_z[:, 0] += v_z[:, 0] / steps
            x_a[:, 0] += v_a[:, 0] / steps
        if kind == 'z':
            out = dec(st[None], z=x_z[:, 0])
        else:  # f: now + dreamed t+8, t+16
            out = dec(st[None], frames=torch.cat([now[:, None], to_lat(x_f[:, 0])], 1))
        chunk = (out[0] * a_std + a_mean).cpu().numpy()
        chunk[:, 7] = np.clip(chunk[:, 7], 0.0, 1.0)
        return chunk

    from uwm_parts import make_sim
    env = make_sim()

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
        res = {seed: rollout(seed, n_exec) for seed in range(1000, 1020)}
        ok = sum(s for s, _ in res.values())
        total += ok
        fails = Counter(m for s, m in res.values() if not s)
        print(f'n_exec {n_exec:2d}: {ok:2d}/20   failures {dict(fails)}', flush=True)
        sweep[n_exec] = {'ok': ok, 'failures': dict(fails), 'per_seed': res}
    print(f'total {total}/40', flush=True)
    (run / f'sweep_decoder_{kind}.json').write_text(json.dumps(sweep, indent=1))


if __name__ == '__main__':
    main()
