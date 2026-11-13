# Does a run's video side bind the colour word to a cube? A linear probe, trained on frozen weights: the
# instruction token where it enters the last video layer (the keys and values the last action layer reads)
# -> the named cube's slot (0/1). T5 reset scenes, first frame (home pose: only picture + word decide):
# 1000 to fit (seeds 5000-5999), 200 to score (9000-9199); rendered once into
# data/robosuite/t5_scenes/<part>_<VAE tag>.npz. Chance ~50%. Writes <run>/binding_probe.json.
# usage: python experiments/verne/scripts/binding_probe.py experiments/verne/runs/<step>/t5/seed0 [...]
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.data import REPO, frame_of, one_hot, to_gpu
from dream_robot.policies.verne.model import Verne
from dream_robot.policies.verne.run import load_task

SEEDS = {'train': range(5000, 6000), 'val': range(9000, 9200)}


def scenes(part):
    tag = hashlib.sha256(vae_mod.WEIGHTS.read_bytes()).hexdigest()[:8]
    path = REPO / 'data/robosuite/t5_scenes' / f'{part}_{tag}.npz'
    if not path.exists():
        from dream_robot.policies.verne.rollouts import make_env

        enc, env, out = vae_mod.load(vae_mod.WEIGHTS), make_env('t5'), {}
        rows = []
        for seed in SEEDS[part]:
            obs = env.reset(seed=seed)
            rows.append((frame_of(obs, enc)[0].half().cpu().numpy(), np.asarray(obs.state, np.float32),
                         one_hot(env.prompt), env.instruction.target_slot))
        env.close()
        out = dict(zip(('latent', 'state', 'instruction', 'slot'), (np.stack(c) for c in zip(*rows))))
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **out)
    z = np.load(path)
    return {k: torch.tensor(z[k], device='cuda') for k in z.files}


state, action, episode, latent, instruction, train_ids, _, _ = load_task('t5')
d = to_gpu(latent[:1], state, instruction[:1], action, episode, train_ids)  # the demos' state stats
data = {p: scenes(p) for p in SEEDS}
for run in map(Path, sys.argv[1:]):
    net = Verne().cuda().eval()
    missing, _ = net.load_state_dict(torch.load(run / 'weights.pt'), strict=False)  # pre-step-10 runs: no future_*
    assert all(k.startswith('future_') for k in missing), missing
    feats = {}
    for p, sc in data.items():
        captured = []
        h = net.video[-1].norm1.register_forward_pre_hook(lambda m, a: captured.append(a[0][:, 1].float()))
        with torch.no_grad():
            for i in range(0, len(sc['slot']), 100):
                s = (sc['state'][i : i + 100] - d.s_mean) / d.s_std
                net.see(sc['latent'][i : i + 100].float(), s, sc['instruction'][i : i + 100])
        h.remove()
        feats[p] = torch.cat(captured)
    mu, sd = feats['train'].mean(0), feats['train'].std(0) + 1e-6
    x, y = (feats['train'] - mu) / sd, data['train']['slot']
    torch.manual_seed(0)
    head = torch.nn.Linear(x.shape[1], 2).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-2)
    for _ in range(2000):
        opt.zero_grad()
        F.cross_entropy(head(x), y).backward()
        opt.step()
    acc = {p: (head((feats[p] - mu) / sd).argmax(-1) == data[p]['slot']).float().mean().item() for p in SEEDS}
    (run / 'binding_probe.json').write_text(json.dumps(acc))
    print(f'{run}: probe train {acc["train"]:.3f}   val {acc["val"]:.3f}')
