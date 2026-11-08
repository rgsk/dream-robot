"""Action decoders that can only read the future (step 1-2 of "actions must read the future").

Trained on the 3 labelled demos, scored on the 10 val demos, fed the REAL future (upper bound):

    bc   joints + now frame                         -> 16 actions   (control: no future at all)
    z    joints + LAM z (t->t+8, t+8->t+16)         -> 16 actions   (LAPA-style)
    f    joints + now frame + frames t+8, t+16      -> 16 actions   (inverse dynamics)

Frames are the dreamzero VAE latents (4x16x16); z is the LAM's (z8_det_rev, std units).
Small nets, MSE on the clean chunk (std units), AdamW + weight decay, cosine; val every 500 steps.
Reference: WAM-C's own chunk-1 actions on val (zuse.log): arm 0.14-0.18, gripper 0.10-0.20.

    .venv/bin/python experiments/lam/scripts/decoder.py --out experiments/lam/decoders
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from lam import NAME, REPO, load, split

dev = 'cuda'
H = 16


class Decoder(nn.Module):
    def __init__(self, kind, z_dim=8, w=512):
        super().__init__()
        self.kind = kind
        n_frames = {'bc': 1, 'z': 0, 'f': 3}[kind]
        self.conv = nn.Sequential(  # latents [B, 4 * n, 16, 16] -> [B, 128]
            nn.Conv2d(4 * n_frames, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 128, 4, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        ) if n_frames else None
        d_in = 8 + (128 if n_frames else 0) + (2 * z_dim if kind == 'z' else 0)
        self.mlp = nn.Sequential(nn.Linear(d_in, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, H * 8))

    def forward(self, state, frames=None, z=None):
        parts = [state]
        if self.conv is not None:
            parts.append(self.conv(frames.flatten(1, 2)))
        if z is not None:
            parts.append(z.flatten(1))
        return self.mlp(torch.cat(parts, 1)).view(-1, H, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labelled', type=int, default=3)
    ap.add_argument('--z', type=Path, default=REPO / 'experiments/lam/wamc_l3/z.pt')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(f'dream_robot/{NAME}', root=REPO / 'data/robosuite' / NAME)
    action = torch.tensor(np.asarray(ds.hf_dataset['action'], dtype=np.float32), device=dev)
    state_np, episode, _ = load(NAME)
    state = torch.tensor(state_np, device=dev)
    eps = split()
    lab10 = torch.tensor(np.isin(episode, eps['labelled']), device=dev)
    a_mean, a_std = action[lab10].mean(0), action[lab10].std(0)
    s_mean, s_std = state[lab10].mean(0), state[lab10].std(0)
    lat = torch.load(REPO / 'cache/lam/vae_lat.pt', weights_only=False)['lat'].to(dev)
    Z = torch.load(args.z).to(dev)
    train_rows = ~np.isin(episode, eps['val'])
    Zn = (Z - Z[train_rows].mean(0)) / Z[train_rows].std(0)
    last = torch.tensor(np.searchsorted(episode, episode, side='right') - 1, device=dev)

    def inputs(r):  # rows [B] -> state, frames [B, 3, 4, 16, 16], z [B, 2, 8], target [B, 16, 8]
        ahead = lambda k: torch.minimum(r + k, last[r])
        frames = torch.stack([lat[r], lat[ahead(8)], lat[ahead(16)]], 1)
        z = torch.stack([Zn[r], Zn[ahead(8)]], 1)
        acts = (action[torch.minimum(r[:, None] + torch.arange(H, device=dev), last[r][:, None])] - a_mean) / a_std
        return (state[r] - s_mean) / s_std, frames, z, acts

    lab_rows = torch.tensor(np.flatnonzero(np.isin(episode, eps['labelled'][:args.labelled])), device=dev)
    val_rows = torch.tensor(np.flatnonzero(np.isin(episode, eps['val'])), device=dev)
    print(f'labelled {eps["labelled"][:args.labelled]} ({len(lab_rows)} rows), val {len(val_rows)} rows', flush=True)

    results = {}
    for kind in ['bc', 'z', 'f']:
        torch.manual_seed(args.seed)
        net = Decoder(kind).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=1e-5)
        feed = lambda s, f, z: net(s, f[:, :1] if kind == 'bc' else f if kind == 'f' else None, z if kind == 'z' else None)
        log = []
        for step in range(1, args.steps + 1):
            r = lab_rows[torch.randint(0, len(lab_rows), (64,), device=dev)]
            s, f, z, a = inputs(r)
            loss = (feed(s, f, z) - a).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 500 == 0:
                net.eval()
                with torch.no_grad():
                    s, f, z, a = inputs(val_rows)
                    err = (feed(s, f, z) - a).square()
                net.train()
                log.append([step, loss.item(), err[..., :7].mean().item(), err[..., 7].mean().item()])
                print(f'{kind:2s} step {step:5d}  train {loss.item():.4f}   val arm {log[-1][2]:.3f}  gripper {log[-1][3]:.3f}', flush=True)
        torch.save({'state_dict': net.state_dict(), 'kind': kind, 'args': vars(args) | {'out': str(args.out), 'z': str(args.z)}},
                   args.out / f'{kind}.pt')
        results[kind] = {'final_arm': log[-1][2], 'final_gripper': log[-1][3], 'log': log}
    (args.out / 'results.json').write_text(json.dumps(results, indent=1))
    print('\nval MSE (std units), real future as input, final step:')
    for k, v in results.items():
        print(f'  {k:2s}  arm {v["final_arm"]:.3f}  gripper {v["final_gripper"]:.3f}')


if __name__ == '__main__':
    main()
