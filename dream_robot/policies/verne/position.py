"""Step 22, stage 1: the video side learns WHERE the cube a colour word names is, in each camera.

A T5 reset scene's first frame and one colour word ('red') -> the named cube's centre in the top camera and in
the wrist camera, (row, col) in patches of that camera's 8 x 8 grid. Labels come from the erase renders (step 15):
the latent cells that change when the cube is hidden (> 0.3, as locate_probe), their change-weighted mean. Every
scene comes twice, once per cube's colour, so only the word says which point is wanted.

Two readouts (Verne.position) of the tokens entering the last video layer, where the arm's last layer reads:
'patch' = one score per token, softmax over each camera's patches, the weighted mean of patch centres (the answer
has to be marked on the named cube's own patches); 'free' = one linear layer on each camera's tokens flattened
(the answer may be spread anywhere). Same labels, same information: only where the lookup must happen differs.
'spread' = patch's weights, but each patch is charged its own distance from the target, so a vote split evenly
around the cube (rows 2 and 4 for a cube at 3) costs instead of averaging out: all weight on the cube's patch.

    patch, free:  loss = ‖position(scene, colour) − centre‖²            (4 numbers, patch units)
    spread:       loss = Σ_i w_i · ‖patch centre_i − centre‖² / 2       (per camera, then the mean)
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from frozendict import frozendict
from torch import Tensor

from . import lr as lr_mod
from .data import Scenes, colour_prompts
from .model import P, Verne
from .pretrain import scene_state


def centres(sc: Scenes, erased: Tensor) -> Tensor:
    # [N, 2, 4]: per scene and cube slot (0 near, 1 far), its centre: top (row, col), wrist (row, col), in patches
    change = (erased.float() - sc.latent.float()[:, None]).abs().sum(2)  # [N, 2, 16, 32]
    w = change * (change > 0.3)
    at = (torch.arange(16, device=w.device) + 0.5) / P  # a latent cell's centre, in patches
    out = []
    for cols in (slice(0, 16), slice(16, 32)):  # top, wrist
        c = w[..., cols]
        c = c / c.sum((2, 3), keepdim=True).clamp(min=1e-6)
        out += [(c.sum(3) * at).sum(2), (c.sum(2) * at).sum(2)]
    return torch.stack(out, 2)


def loss(
    net: Verne, frame: Tensor, state: Tensor, ins: Tensor, target: Tensor, readout: str
) -> Tensor:
    # target [B, 4] (centres) -> the readout's loss (module docstring)
    if readout != 'spread':
        return F.mse_loss(net.position(frame, state, ins, readout), target)
    w = net.position_weights(frame, state, ins)  # [B, 2, 8, 8]
    at = torch.arange(8, device=w.device) + 0.5
    t = target.view(-1, 2, 2)  # [B, camera, (row, col)]
    d2 = (at[:, None] - t[..., 0, None, None]) ** 2 + (at[None, :] - t[..., 1, None, None]) ** 2
    return (w * d2).sum((2, 3)).mean() / 2


@torch.no_grad()
def val_numbers(
    net: Verne, sc: Scenes, cen: Tensor, s_mean: Tensor, s_std: Tensor, readout: str
) -> dict[str, float]:
    # every val scene x both cubes: distance from the named cube's centre per camera (patches), and 'right': the
    # prediction is nearer the named cube's centre than the other cube's
    net.eval()
    n = len(sc.slot)
    ids = torch.arange(n, device=sc.latent.device).repeat(2)
    which = torch.arange(2, device=ids.device).repeat_interleave(n)
    pred = net.position(
        sc.latent[ids].float(),
        scene_state(sc, ids, s_mean, s_std),
        colour_prompts(sc, ids, which),
        readout,
    )
    out = {}
    for name, k in (('top', slice(0, 2)), ('wrist', slice(2, 4))):
        named = (pred[:, k] - cen[ids, which, k]).norm(dim=1)
        other = (pred[:, k] - cen[ids, 1 - which, k]).norm(dim=1)
        out[f'err_{name}'] = named.mean().item()
        out[f'right_{name}'] = (named < other).float().mean().item()
    net.train()
    return out


def pretrain(
    sc: Scenes,
    cen: Tensor,
    val: Scenes,
    val_cen: Tensor,
    s_mean: Tensor,
    s_std: Tensor,
    steps: int,
    readout: str,
    B: int = 64,
    lr: float = 3e-4,
    seed: int = 0,
    ckpt: Path | None = None,
) -> frozendict[str, Any]:
    # out: {'weights': the whole Verne on the cpu (its action side untouched), 'log': per 500 steps
    # {step, train, err_top, err_wrist, right_top, right_wrist}}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = sc.latent.device
    net = Verne().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    start, log = lr_mod.resume(ckpt, steps, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, 5000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        ids = torch.tensor(rng.integers(0, len(sc.slot), B), device=dev)
        which = torch.tensor(rng.integers(0, 2, B), device=dev)  # which cube the word names
        with torch.autocast('cuda', dtype=torch.bfloat16):
            lp = loss(
                net,
                sc.latent[ids].float(),
                scene_state(sc, ids, s_mean, s_std),
                colour_prompts(sc, ids, which),
                cen[ids, which],
                readout,
            )
        opt.zero_grad()
        lp.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 500 == 0:
            v = val_numbers(net, val, val_cen, s_mean, s_std, readout)
            log.append({'step': step, 'train': lp.item()} | v)
            print(
                f'step {step:6d}   train {lp.item():.4f}   val err top {v["err_top"]:.2f} wrist '
                f'{v["err_wrist"]:.2f} patches   right top {v["right_top"]:.3f} wrist {v["right_wrist"]:.3f}',
                flush=True,
            )
    return frozendict(weights={k: v.cpu() for k, v in net.state_dict().items()}, log=log)


@torch.no_grad()
def picture(
    net: Verne,
    vae,
    val: Scenes,
    val_cen: Tensor,
    s_mean: Tensor,
    s_std: Tensor,
    readout: str,
    n: int = 4,
) -> np.ndarray:
    # uint8 [H, W, 3]: per val scene (row) and word (column: the near cube's colour, the far one's) the decoded
    # frame, the target as a green cross and the prediction as a magenta one, in both cameras
    from PIL import Image, ImageDraw

    net.eval()
    dev = val.latent.device
    S, tiles = 3, []
    for i in range(n):
        frame = val.latent[i : i + 1].float()
        if frame.shape[1] == 3:  # vae.INPUT 'pixels': the canvas itself
            pic = ((frame[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        else:
            z = frame * vae.z_std[:, None, None] + vae.z_mean[:, None, None]
            pic = ((vae.dec(z)[0].clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        for which in (0, 1):
            ids, w = torch.tensor([i], device=dev), torch.tensor([which], device=dev)
            pred = net.position(
                frame, scene_state(val, ids, s_mean, s_std), colour_prompts(val, ids, w), readout
            )[0]
            im = Image.fromarray(pic).resize((256 * S, 128 * S), Image.NEAREST)
            d = ImageDraw.Draw(im)
            for pts, colour in (
                (val_cen[i, which].tolist(), (0, 200, 0)),
                (pred.tolist(), (220, 0, 220)),
            ):
                for cam in (0, 1):
                    r, c = pts[2 * cam], pts[2 * cam + 1]
                    x, y = (c + 8 * cam) * 16 * S, r * 16 * S
                    d.line([(x - 12, y), (x + 12, y)], fill=colour, width=3)
                    d.line([(x, y - 12), (x, y + 12)], fill=colour, width=3)
            tiles.append(np.asarray(im))
    net.train()
    rows = [
        np.concatenate([tiles[2 * i], np.full_like(tiles[0][:, :8], 255), tiles[2 * i + 1]], 1)
        for i in range(n)
    ]
    gap = np.full_like(rows[0][:8], 255)
    return np.concatenate([x for r in rows for x in (r, gap)][:-1], 0)
