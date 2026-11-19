"""Step 15, stage 1: the video side learns to erase the cube a colour word names.

A T5 reset scene's first frame (no demo) and one colour word ('red') -> the change that erases that
cube: latent(scene with the cube hidden) − latent(scene). Every scene comes twice, the same picture
with each cube's colour, so only the word says which cube goes: the model must find the named colour's
place (colour -> place), the lookup the arm needs. The change is read from the patches entering the
last video layer, the tokens whose keys and values the last action layer reads, so the lookup has to
happen where the arm can see it. (Step 14 painted from captions instead: place -> colour, probe 53%.)

Then binding_probe, frozen weights: a clean frame and the reset's colour prompt ('red left'), a linear
probe on those patches -> near / far. Untrained ~68-73% (32,768 features from 1000 scenes), step 13
71%, step 14 53%. Rule: >= 85% stage 2, <= 76% stop.

    loss = ‖edit(scene, colour) − (erased − scene)‖²
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from frozendict import frozendict
from torch import Tensor

from . import lr as lr_mod
from .data import COLOURS, PLACES, Scenes, colour_prompts
from .model import Verne


def scene_state(sc: Scenes, ids: Tensor, s_mean: Tensor, s_std: Tensor) -> Tensor:
    # [B, 16] std units, the demos' stats; a reset scene: previous = now
    return (sc.state[ids].repeat(1, 2) - s_mean.repeat(2)) / s_std.repeat(2)


def target(sc: Scenes, erased: Tensor, ids: Tensor, which: Tensor) -> Tensor:
    # [B, Z, 16, 32] the change that hides slot `which` in scenes `ids`
    return erased[ids, which].float() - sc.latent[ids].float()


@torch.no_grad()
def val_numbers(net: Verne, sc: Scenes, erased: Tensor, s_mean: Tensor, s_std: Tensor) -> dict:
    # every val scene x both cubes: loss with the right colour word and with the other cube's
    # (wrong - right = how much the word is used), and erase accuracy: the predicted change is larger
    # on the named cube's cells than on the other cube's (cells weighted by each cube's true change)
    net.eval()
    n = len(sc.slot)
    ids = torch.arange(n, device=sc.latent.device).repeat(2)
    which = torch.arange(2, device=ids.device).repeat_interleave(n)
    state = scene_state(sc, ids, s_mean, s_std)
    true = target(sc, erased, ids, which)
    other = target(sc, erased, ids, 1 - which)
    pred = net.edit(sc.latent[ids].float(), state, colour_prompts(sc, ids, which))
    wrong = net.edit(sc.latent[ids].float(), state, colour_prompts(sc, ids, 1 - which))
    size = lambda c: c.abs().sum(1)  # [B, 16, 32] per cell, over channels
    on_named = (size(pred) * size(true)).sum((1, 2))
    on_other = (size(pred) * size(other)).sum((1, 2))
    net.train()
    return {
        'right': F.mse_loss(pred, true).item(),
        'wrong': F.mse_loss(wrong, true).item(),
        'erase_acc': (on_named > on_other).float().mean().item(),
    }


def pretrain(
    sc: Scenes,
    erased: Tensor,
    val: Scenes,
    val_erased: Tensor,
    s_mean: Tensor,
    s_std: Tensor,
    steps: int,
    B: int = 64,
    lr: float = 3e-4,
    seed: int = 0,
    ckpt: Path | None = None,
) -> frozendict[str, Any]:
    # out: {'weights': state dict on the cpu (the whole Verne: its action side untouched), 'log': per
    # 1000 steps {step, train, right, wrong, erase_acc}}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = sc.latent.device
    net = Verne().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    start, log = lr_mod.resume(ckpt, None, steps, 0.2, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, 0.2, 2000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        ids = torch.tensor(rng.integers(0, len(sc.slot), B), device=dev)
        which = torch.tensor(rng.integers(0, 2, B), device=dev)  # which cube the word names
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = net.edit(
                sc.latent[ids].float(),
                scene_state(sc, ids, s_mean, s_std),
                colour_prompts(sc, ids, which),
            )
            loss = F.mse_loss(pred.float(), target(sc, erased, ids, which))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 1000 == 0:
            v = val_numbers(net, val, val_erased, s_mean, s_std)
            log.append({'step': step, 'train': loss.item()} | v)
            print(
                f'step {step:6d}   train {loss.item():.4f}   val right {v["right"]:.4f}   '
                f'wrong {v["wrong"]:.4f}   erase acc {v["erase_acc"]:.3f}',
                flush=True,
            )
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)


def binding_probe(
    net: Verne, sc: Scenes, val: Scenes, s_mean: Tensor, s_std: Tensor
) -> dict[str, dict[str, float]]:
    # -> {'colour' / 'both': {'train', 'val'} accuracy}. Prompts: the reset's own ('red left');
    # 'both' adds
    # the named cube's place word ('near red left': the ceiling, the answer is in the words)
    net.eval()

    def features(s: Scenes, kind: str) -> Tensor:
        ins = s.instruction.clone()
        if kind == 'both':
            ins[:, PLACES] = F.one_hot(s.slot, 2).float()
        captured = []
        hook = net.video[-1].norm1.register_forward_pre_hook(
            lambda m, a: captured.append(a[0].flatten(1).float())
        )
        with torch.no_grad():
            for i in range(0, len(s.slot), 100):
                ids = torch.arange(i, min(i + 100, len(s.slot)), device=s.latent.device)
                net.see(s.latent[ids].float(), scene_state(s, ids, s_mean, s_std), ins[ids])
        hook.remove()
        return torch.cat(captured)

    out = {}
    for kind in ('colour', 'both'):
        f = {'train': features(sc, kind), 'val': features(val, kind)}
        mu, sd = f['train'].mean(0), f['train'].std(0) + 1e-6
        torch.manual_seed(0)
        head = torch.nn.Linear(f['train'].shape[1], 2).to(mu.device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-1)
        for _ in range(2000):
            opt.zero_grad()
            F.cross_entropy(head((f['train'] - mu) / sd), sc.slot).backward()
            opt.step()
        with torch.no_grad():
            out[kind] = {
                p: (head((f[p] - mu) / sd).argmax(-1) == s.slot).float().mean().item()
                for p, s in (('train', sc), ('val', val))
            }
        print(
            f'probe, prompt {kind:6s}: train {out[kind]["train"]:.3f}   val {out[kind]["val"]:.3f}',
            flush=True,
        )
    return out


def locate_probe(
    net: Verne,
    sc: Scenes,
    val: Scenes,
    erased: Tensor,
    val_erased: Tensor,
    s_mean: Tensor,
    s_std: Tensor,
) -> dict[str, float]:
    # step 19: is the NAMED cube marked on its own patches? Each reset scene twice (the near cube's colour
    # word, then the far one's, plus the scene's side word); a patch shows cube k if hiding k changes it
    # (summed |latent change| > 0.3 in any of its 2 x 2 cells). One linear head shared by all patches, on the
    # token entering the last video layer -> 'named cube here'. 'named_gt_other': per val (scene, word), the
    # mean score on the named cube's patches beats the other cube's (chance 50%; erase pretraining 100%, its
    # stage 2 ~76%, clips / none ~53-55%). 'auc': named-cube patches vs all others.
    net.eval()

    def parts(s: Scenes, er: Tensor):
        change = (er.float() - s.latent.float()[:, None]).abs().sum(2)  # [N, 2, 16, 32]
        cube = F.max_pool2d((change > 0.3).float(), 2).flatten(2).bool()  # [N, 2, 128]
        out = []
        for which in (0, 1):
            captured = []
            hook = net.video[-1].norm1.register_forward_pre_hook(
                lambda m, a: captured.append(a[0].float())
            )
            with torch.no_grad():
                for i in range(0, len(s.slot), 100):
                    ids = torch.arange(i, min(i + 100, len(s.slot)), device=s.latent.device)
                    ins = s.instruction[ids].clone()
                    ins[:, COLOURS] = 0
                    ins[torch.arange(len(ids)), COLOURS.start + s.colours[ids, which]] = 1
                    net.see(s.latent[ids].float(), scene_state(s, ids, s_mean, s_std), ins)
            hook.remove()
            out.append(torch.cat(captured))
        named = torch.stack([cube[:, 0], cube[:, 1]], 1)
        other = torch.stack([cube[:, 1], cube[:, 0]], 1)
        return torch.stack(out, 1), named, other  # [N, 2, 128, 256], [N, 2, 128] x 2

    (ft, nt, _), (fv, nv, ov) = parts(sc, erased), parts(val, val_erased)
    mu, sd = ft.flatten(0, 2).mean(0), ft.flatten(0, 2).std(0) + 1e-6
    x, y = ((ft - mu) / sd).flatten(0, 2), nt.flatten().float()
    torch.manual_seed(0)
    head = torch.nn.Linear(ft.shape[-1], 1).to(x.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-2)
    pw = (1 - y).sum() / y.sum()
    for _ in range(500):
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(head(x)[:, 0], y, pos_weight=pw).backward()
        opt.step()
    with torch.no_grad():
        sv = head((fv - mu) / sd)[..., 0]  # [N, 2, 128]
        n_, o_ = nv.float(), ov.float()
        on_named = (sv * n_).sum(2) / n_.sum(2).clamp(min=1)
        on_other = (sv * o_).sum(2) / o_.sum(2).clamp(min=1)
        ok = (n_.sum(2) > 0) & (o_.sum(2) > 0)
        score, label = sv.flatten(), nv.flatten()
        ranks = torch.empty_like(score)
        ranks[score.argsort()] = torch.arange(
            1, len(score) + 1, device=score.device, dtype=torch.float
        )
        pos = label.sum()
        out = {
            'named_gt_other': (on_named > on_other)[ok].float().mean().item(),
            'auc': ((ranks[label].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos))).item(),
        }
    print(
        f'locate probe: named > other {out["named_gt_other"]:.3f}   auc {out["auc"]:.3f}',
        flush=True,
    )
    return out


@torch.no_grad()
def edit_picture(
    net: Verne, vae, val: Scenes, val_erased: Tensor, s_mean: Tensor, s_std: Tensor, n: int = 4
) -> np.ndarray:
    # uint8 [3, H, W]: per val scene (column) the scene, then scene + predicted change for the near
    # cube's colour and for the far cube's colour (the right cube should vanish in each)
    net.eval()
    dev = val.latent.device
    ids = torch.arange(n, device=dev)
    frame, state = val.latent[ids].float(), scene_state(val, ids, s_mean, s_std)
    rows = [frame]
    for which in (0, 1):
        w = torch.full((n,), which, device=dev)
        rows.append(frame + net.edit(frame, state, colour_prompts(val, ids, w)))

    def decode(z: Tensor) -> Tensor:
        x = vae.dec(z * vae.z_std[:, None, None] + vae.z_mean[:, None, None]).clamp(-1, 1)
        return ((x + 1) * 127.5).round().byte()  # [n, 3, 128, 256]

    grid = torch.cat(
        [torch.cat(list(decode(r)), 2) for r in rows], 1
    )  # rows stacked, scenes side by side
    return grid.cpu().numpy()
