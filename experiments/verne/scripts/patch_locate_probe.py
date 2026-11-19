"""Step 19 check: does the present frame mark WHERE the named cube is, patch by patch? (The whole-frame binding probe
asks only near / far; every run, good arm or frame-blind, scores 79%+ after demos.)

T5 reset scenes, each twice: the colour word of the near cube, then of the far one (plus the scene's side word).
Per patch (8 x 16 grid of the [top | wrist] canvas), the label comes from the erased renders: a patch belongs to a
cube if hiding that cube changes it (summed |latent change| over channels > 0.3 in any of its 2 x 2 cells). One
linear head, shared by all patches (tokens carry no position: RoPE lives in attention), on the 256-number token
entering the last video layer -> 'this patch shows the named cube', trained on 1000 scenes x 2, scored on 200 x 2:

    named > other   per val (scene, word): mean score on the named cube's patches > on the other cube's.
                    Chance 50%; a token that knows only 'I am a red cube' and not the word can't beat it.
    AUC             named-cube patches vs all other patches.

    python experiments/verne/scripts/patch_locate_probe.py name=path/to/weights.pt ...   ('untrained' for a fresh
    Verne). Writes scratch/patch_locate_probe.json.
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from dream_robot.policies.verne import vae as vae_mod
from dream_robot.policies.verne.data import COLOURS, load_erased, load_scenes, load_table
from dream_robot.policies.verne.model import Verne
from dream_robot.policies.verne.pretrain import scene_state
from dream_robot.policies.verne.run import TASKS, split

state, _, episode = load_table(TASKS['t5'][0])
train_ids, _ = split('t5', int(episode.max()) + 1)
s = torch.tensor(state[np.isin(episode, train_ids)], device='cuda')
s_mean, s_std = s.mean(0), s.std(0)
data = {}
for part in ('train', 'val'):
    sc, er = load_scenes(part, vae_mod.WEIGHTS), load_erased(part, vae_mod.WEIGHTS)
    change = (er.float() - sc.latent.float()[:, None]).abs().sum(2)  # [N, 2, 16, 32]
    cube = F.max_pool2d((change > 0.3).float(), 2).flatten(2).bool()  # [N, 2, 128] patch shows cube k
    data[part] = (sc, cube)
    print(part, 'patches per cube: mean', cube.float().sum(2).mean().item())


def features(net, sc):
    # -> tokens [N, 2, 128, 256] for (scene, named slot 0 / 1)
    out = []
    for which in (0, 1):
        captured = []
        h = net.video[-1].norm1.register_forward_pre_hook(lambda m, a: captured.append(a[0].float()))
        with torch.no_grad():
            for i in range(0, len(sc.slot), 100):
                ids = torch.arange(i, min(i + 100, len(sc.slot)), device='cuda')
                ins = sc.instruction[ids].clone()
                ins[:, COLOURS] = 0
                ins[torch.arange(len(ids)), COLOURS.start + sc.colours[ids, which]] = 1
                net.see(sc.latent[ids].float(), scene_state(sc, ids, s_mean, s_std), ins)
        h.remove()
        out.append(torch.cat(captured))
    return torch.stack(out, 1)


def auc(score, label):
    order = score.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(score) + 1, device=score.device, dtype=torch.float)
    pos = label.sum()
    return ((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos))).item()


results = {}
for arg in sys.argv[1:]:
    name, _, path = arg.partition('=')
    torch.manual_seed(0)
    net = Verne().cuda().eval()
    if name != 'untrained':
        missing, unexpected = net.load_state_dict(torch.load(path), strict=False)
        assert not unexpected and all(k.startswith(('time_video', 'gen_out', 'future_heads')) for k in missing), missing
    feats, labels, others = {}, {}, {}
    for part, (sc, cube) in data.items():
        f = features(net, sc)  # [N, 2, 128, 256]
        named = torch.stack([cube[:, 0], cube[:, 1]], 1)  # [N, 2(which), 128]
        other = torch.stack([cube[:, 1], cube[:, 0]], 1)
        feats[part], labels[part], others[part] = f, named, other
    mu, sd = feats['train'].flatten(0, 2).mean(0), feats['train'].flatten(0, 2).std(0) + 1e-6
    x = ((feats['train'] - mu) / sd).flatten(0, 2)
    y = labels['train'].flatten().float()
    torch.manual_seed(0)
    head = torch.nn.Linear(256, 1).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-2)
    pw = (1 - y).sum() / y.sum()
    for _ in range(500):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(head(x)[:, 0], y, pos_weight=pw).backward()
        opt.step()
    with torch.no_grad():
        sv = head((feats['val'] - mu) / sd)[..., 0]  # [N, 2, 128]
        n_, o_ = labels['val'].float(), others['val'].float()
        on_named = (sv * n_).sum(2) / n_.sum(2).clamp(min=1)
        on_other = (sv * o_).sum(2) / o_.sum(2).clamp(min=1)
        ok = (n_.sum(2) > 0) & (o_.sum(2) > 0)
        r = {
            'named_gt_other': (on_named > on_other)[ok].float().mean().item(),
            'auc': auc(sv.flatten(), labels['val'].flatten()),
        }
    results[name] = r
    print(f'{name:28s} named > other {r["named_gt_other"]:.3f}   auc {r["auc"]:.3f}', flush=True)
open('scratch/patch_locate_probe.json', 'w').write(json.dumps(results, indent=1))
print('------ FINISHED ------')
