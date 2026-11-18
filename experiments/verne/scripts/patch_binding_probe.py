# Step 14's check: does a frozen video side already know which cube a colour prompt names? A linear probe
# on the 128 patch tokens where they enter the last video layer (flattened, 128 x 256; the keys and values
# the last action layer reads) -> the named cube's slot (0 near / 1 far). Linear on purpose: matching the
# prompt's colour to a place is an 'and' a weighted sum can't do, so the probe only scores if the model
# already did the match. T5 reset scenes (home pose), 1000 to fit (seeds 5000-5999), 200 to score
# (9000-9199), from binding_probe.py's cache. Prompts: 'colour' = the scene's recorded prompt ('red left'),
# 'both' = plus the named cube's place word ('near red left', the ceiling: the answer is in the words).
# Chance ~50%. Writes <run>/patch_binding_probe.json; 'random' = an untrained Verne (torch seed 0).
# usage: python experiments/verne/scripts/patch_binding_probe.py random experiments/verne/runs/<step>/t5/seed0
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.data import PLACES, REPO, load_table
from dream_robot.policies.verne.model import Verne
from dream_robot.policies.verne.run import split

PARTS = ('train', 'val')


def scenes(part):
    # the cache holds the old 6-word one-hot (colours, sides): put the 2 place slots in front
    tag = hashlib.sha256(vae_mod.WEIGHTS.read_bytes()).hexdigest()[:8]
    z = np.load(REPO / 'data/robosuite/t5_scenes' / f'{part}_{tag}.npz')
    sc = {k: torch.tensor(z[k], device='cuda') for k in z.files}
    assert sc['instruction'].shape[1] == 6
    sc['instruction'] = F.pad(sc['instruction'], (2, 0))
    return sc


def prompts(sc, kind):
    ins = sc['instruction'].clone()
    if kind == 'both':
        ins[:, PLACES] = F.one_hot(sc['slot'], 2).float()
    return ins


# the demos' state stats, as training normalised them; scenes start at home, so previous = now
state, _, episode = load_table('instructed_pick_place_square_noise005_n150')
train_ids, _ = split('t5', int(episode.max()) + 1)
s = torch.tensor(state[np.isin(episode, train_ids)], device='cuda')
s_mean, s_std = s.mean(0).repeat(2), s.std(0).repeat(2)
data = {p: scenes(p) for p in PARTS}

for name in sys.argv[1:]:
    torch.manual_seed(0)
    net = Verne().cuda().eval()
    if name != 'random':
        net.load_state_dict(torch.load(Path(name) / 'weights.pt'))
    out = {}
    for kind in ('colour', 'both'):
        feats = {}
        for p, sc in data.items():
            captured = []
            h = net.video[-1].norm1.register_forward_pre_hook(
                lambda m, a: captured.append(a[0].flatten(1).float())
            )
            with torch.no_grad():
                for i in range(0, len(sc['slot']), 100):
                    st = (sc['state'][i : i + 100].repeat(1, 2) - s_mean) / s_std
                    net.see(sc['latent'][i : i + 100].float(), st, prompts(sc, kind)[i : i + 100])
            h.remove()
            feats[p] = torch.cat(captured)
        mu, sd = feats['train'].mean(0), feats['train'].std(0) + 1e-6
        x, y = (feats['train'] - mu) / sd, data['train']['slot']
        torch.manual_seed(0)
        head = torch.nn.Linear(x.shape[1], 2).cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-1)
        for _ in range(2000):
            opt.zero_grad()
            F.cross_entropy(head(x), y).backward()
            opt.step()
        out[kind] = {
            p: (head((feats[p] - mu) / sd).argmax(-1) == data[p]['slot']).float().mean().item()
            for p in PARTS
        }
        print(
            f'{name}: prompt {kind:6s}  probe train {out[kind]["train"]:.3f}   val {out[kind]["val"]:.3f}',
            flush=True,
        )
    dest = (
        REPO / 'scratch/patch_binding_probe_random.json'
        if name == 'random'
        else Path(name) / 'patch_binding_probe.json'
    )
    dest.write_text(json.dumps(out, indent=1))
print('val slot share far:', data['val']['slot'].float().mean().item())
print('------ FINISHED ------')
