"""LIBERO for Verne (2026-11-20): the demos from data/libero/npz (experiments/libero/scripts/to_npz.py), the skill
vocabulary, and stage 1: point at a named object in each camera (step 22's spread, on pixels; step 23).

The policy sees pixels: the [agentview | wrist] canvas, 3 x 128 x 256, 16 x 16 px patches (no VAE). State: 7 joints
+ 2 finger openings (9). Vocabulary (WORDS): the verbs, the names, and each name again as a target ('to_stove'), so a
put(object, target) call can say which is which with the word rows Verne already has (row 0 'no word').

Stage 1: a demo frame (the arm anywhere) + one name -> that object's centre in each camera, in patches of the
camera's 8 x 8 grid (to_npz's pixel centres / 16). Every frame is asked about any name in view in at least one camera
(not only its task's objects). The loss is step 22's spread per camera, skipped where the name is out of view
(mostly the wrist):

    loss = Σ_cam seen_cam · Σ_i w_i ‖patch centre_i − target_cam‖² / 2   /   Σ_cam seen_cam

Stage 2 (step 6): the demos with their skill call as words (verb + object + 'to_<target>'), Verne's flow-matching
action loss, and stage 1's loss kept on (weight 1.0), as T5 step 22. Actions are T5 step 20's 11 numbers (option B):
per chunk step the hand pose the demo was AIMING at (to_npz's goal pose: the recorded command applied to the
controller's hand), relative to the hand at the chunk's start (hand.relative: position in its frame + the relative
rotation's first two columns), then the gripper command (-1 open, +1 close) and progress (step / (length − 1)).
"""

from pathlib import Path
from types import SimpleNamespace as namespace

import numpy as np
import torch
from torch import Tensor

from . import hand as hand_mod
from . import lr as lr_mod
from .data import ACTION, CHUNK
from .model import Verne, loss as action_loss

REPO = Path(__file__).resolve().parents[3]
NPZ = REPO / 'data/libero/npz'
VERBS = ('turn_on', 'turn_off', 'put')
NAMES = ('stove_knob', 'stove', 'moka_pot', 'frying_pan')  # to_npz's NAMES without 'none', same order
WORDS = VERBS + NAMES + tuple(f'to_{n}' for n in NAMES)
TARGETS = ('stove',)  # names that are a put target in some skill: also asked as 'to_<name>' in stage 1
STATE = 9  # 7 joints + 2 fingers
VAL_EPISODES = 5  # of each task's 50 demos: episodes 0-4 held out (as the LIBERO VAE)


def load(dev: str = 'cuda', limit: int | None = None) -> namespace:
    # every task's npz -> one table on the GPU: frame [N, 3, 128, 256] uint8, state [N, 9], prev [N] (the row one
    # control step back, a demo's first row itself), centre [N, 4 names, 2 cams, 2] in patches, seen [N, 4, 2],
    # task [N], episode [N] (per task), val [N] bool, tasks (names), s_mean / s_std (train rows). limit: rows per task
    # (a quick check)
    parts = []
    for t, path in enumerate(sorted(NPZ.glob('*.npz'))):
        z = np.load(path)
        assert tuple(str(n) for n in z['names'])[1:] == NAMES, path
        n = len(z['episode']) if limit is None else limit
        frame = np.concatenate([z['top'][:n], z['wrist'][:n]], 2).transpose(0, 3, 1, 2)  # [n, 3, 128, 256]
        state = np.concatenate([z['joints'][:n], z['gripper'][:n]], 1)
        ep = z['episode'][:n]
        part = dict(frame=frame, state=state, centre=z['pix'][:n, 1:] / 16, seen=z['seen'][:n, 1:],
                    task=np.full(n, t), episode=ep, name=path.stem, skill=str(z['skill']), tokens=z['tokens'])
        for k in ('hand_pos', 'hand_rot', 'goal_pos', 'goal_rot'):
            part[k] = z[k][:n]
        part['grip'] = z['action'][:n, 6]
        parts.append(part)
    keys = ('frame', 'state', 'centre', 'seen', 'task', 'episode', 'hand_pos', 'hand_rot', 'goal_pos', 'goal_rot',
            'grip')
    cat = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    # previous row: same task and episode, one step back
    start = np.r_[True, (np.diff(cat['episode']) != 0) | (np.diff(cat['task']) != 0)]
    prev = np.arange(len(start)) - 1
    prev[start] = np.flatnonzero(start)
    val = cat['episode'] < VAL_EPISODES
    state = torch.tensor(cat['state'], device=dev, dtype=torch.float32)
    tr = torch.tensor(~val, device=dev)
    # each row's demo: its last row (chunks are held there) and progress
    end = np.r_[start[1:], True]
    first = np.maximum.accumulate(np.where(start, np.arange(len(start)), 0))
    last = np.flatnonzero(end)[np.searchsorted(np.flatnonzero(end), np.arange(len(end)))]
    progress = (np.arange(len(start)) - first) / np.maximum(last - first, 1)
    t = lambda x: torch.tensor(x, device=dev, dtype=torch.float32)  # noqa: E731
    return namespace(
        frame=torch.tensor(cat['frame'], device=dev),
        state=state,
        prev=torch.tensor(prev, device=dev),
        centre=torch.tensor(np.nan_to_num(cat['centre']), device=dev, dtype=torch.float32),
        seen=torch.tensor(cat['seen'], device=dev),
        task=cat['task'],
        episode=cat['episode'],
        val=val,
        tasks=[p['name'] for p in parts],
        skills=[p['skill'] for p in parts],
        task_words=torch.stack([skill_words(p['tokens']) for p in parts]).to(dev),  # [tasks, len(WORDS)]
        last=torch.tensor(last, device=dev),
        hand_pos=t(cat['hand_pos']),
        hand_rot=t(cat['hand_rot']),
        goal_pos=t(cat['goal_pos']),
        goal_rot=t(cat['goal_rot']),
        grip=t(cat['grip']),
        progress=t(progress),
        s_mean=state[tr].mean(0),
        s_std=state[tr].std(0).clamp(min=1e-6),
    )


def skill_words(tokens) -> Tensor:
    # to_npz's skill tokens [verb, object, target] (indices into its VERBS and ('none',) + NAMES) -> the instruction
    # multi-hot [len(WORDS)]: the verb, the object's name, and 'to_<target>' (none for turn_on / turn_off)
    v, o, tg = (int(x) for x in tokens)
    out = torch.zeros(len(WORDS))
    out[v] = 1.0
    out[len(VERBS) + o - 1] = 1.0
    if tg:
        out[len(VERBS) + len(NAMES) + tg - 1] = 1.0
    return out


def frames(d: namespace, rows: Tensor) -> Tensor:
    return d.frame[rows].float() / 127.5 - 1


def states(d: namespace, rows: Tensor) -> Tensor:
    # [B, 18] std units: [previous step, now]
    return (torch.cat([d.state[d.prev[rows]], d.state[rows]], 1) - d.s_mean.repeat(2)) / d.s_std.repeat(2)


def words(names: Tensor, as_target: Tensor) -> Tensor:
    # name index [B] (into NAMES) + whether to ask it as a target [B] -> the instruction multi-hot [B, len(WORDS)]
    out = torch.zeros(len(names), len(WORDS), device=names.device)
    col = len(VERBS) + names + as_target.long() * len(NAMES)
    out[torch.arange(len(names)), col] = 1.0
    return out


def queries(d: namespace, rows: Tensor, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
    # per row a random name in view in at least one camera, and (for TARGETS names) at random as 'to_<name>'
    any_seen = d.seen[rows].any(2)  # [B, 4]
    score = torch.tensor(rng.random(any_seen.shape), device=rows.device) * any_seen
    names = score.argmax(1)
    can_target = torch.tensor([n in TARGETS for n in NAMES], device=rows.device)[names]
    as_target = can_target & torch.tensor(rng.random(len(rows)) < 0.5, device=rows.device)
    return names, as_target


def spread_loss(w: Tensor, target: Tensor, seen: Tensor) -> Tensor:
    # w [B, 2 cams, 8, 8] (position_weights), target [B, 2, 2] (row, col) in patches, seen [B, 2] -> the module
    # docstring's loss: each camera's spread, out-of-view cameras skipped
    at = torch.arange(8, device=w.device) + 0.5
    d2 = (at[:, None] - target[..., 0, None, None]) ** 2 + (at[None, :] - target[..., 1, None, None]) ** 2
    per_cam = (w * d2).sum((2, 3)) / 2  # [B, 2]
    s = seen.float()
    return (per_cam * s).sum() / s.sum().clamp(min=1)


def predict(net: Verne, d: namespace, rows: Tensor, names: Tensor, as_target: Tensor) -> Tensor:
    # [B, 2 cams, 2]: the weighted mean of patch centres per camera
    return net.position(frames(d, rows), states(d, rows), words(names, as_target), 'spread').view(-1, 2, 2)


@torch.no_grad()
def val_numbers(net: Verne, d: namespace, every: int = 10) -> dict[str, float]:
    # every `every`-th held-out row x every name in view: per camera the error to the target (patches), and 'right':
    # the prediction is nearer the named object than every other object in view in that camera; per name and overall
    net.eval()
    rows = torch.tensor(np.flatnonzero(d.val)[::every], device=d.frame.device)
    out, total = {}, {'err': [], 'right': []}
    for j, name in enumerate(NAMES):
        r = rows[d.seen[rows, j].any(1)]
        if not len(r):
            continue
        errs, rights = [], []
        for i in range(0, len(r), 256):
            b = r[i : i + 256]
            pred = predict(net, d, b, torch.full_like(b, j), torch.zeros_like(b, dtype=torch.bool))
            dist = (pred[:, None] - d.centre[b]).norm(dim=-1)  # [B, 4 names, 2 cams]
            dist = torch.where(d.seen[b], dist, torch.inf)
            own = dist[:, j]  # [B, 2]
            ok = d.seen[b, j]
            errs.append(own[ok])
            rights.append((dist.argmin(1) == j)[ok].float())
        e, rt = torch.cat(errs), torch.cat(rights)
        out[f'err_{name}'], out[f'right_{name}'] = e.mean().item(), rt.mean().item()
        total['err'].append(e)
        total['right'].append(rt)
    out['err'] = torch.cat(total['err']).mean().item()
    out['right'] = torch.cat(total['right']).mean().item()
    net.train()
    return out


def stage1(d: namespace, steps: int, B: int = 64, lr: float = 3e-4, seed: int = 0, ckpt: Path | None = None) -> dict:
    # stage 1 (module docstring) -> {'weights': the whole Verne on the cpu, 'log': per 500 steps}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = d.frame.device
    train_rows = np.flatnonzero(~d.val & d.seen.any((1, 2)).cpu().numpy())
    net = Verne(n_words=len(WORDS), state_dim=STATE, frames='pixels').to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    start, log = lr_mod.resume(ckpt, steps, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, 1000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        rows = torch.tensor(rng.choice(train_rows, B), device=dev)
        names, as_target = queries(d, rows, rng)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            w = net.position_weights(frames(d, rows), states(d, rows), words(names, as_target))
        loss = spread_loss(w.float(), d.centre[rows, names], d.seen[rows, names])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 500 == 0:
            v = val_numbers(net, d)
            log.append({'step': step, 'train': loss.item()} | v)
            per = '  '.join(f'{n} {v.get(f"err_{n}", float("nan")):.2f}/{v.get(f"right_{n}", float("nan")):.2f}'
                            for n in NAMES)
            print(f'step {step:5d}  train {loss.item():.4f}  val err {v["err"]:.2f} right {v["right"]:.3f}  '
                  f'[{per}]', flush=True)
    return {'weights': {k: v.cpu() for k, v in net.state_dict().items()}, 'log': log}


@torch.no_grad()
def picture(net: Verne, d: namespace, n_per_task: int = 2) -> np.ndarray:
    # uint8 [H, W, 3]: held-out frames (n per task, mid-demo), each asked about every name in view; a cross per name
    # in its colour: target thin, prediction thick; 2x size
    from PIL import Image, ImageDraw

    net.eval()
    colours = {'stove_knob': (255, 200, 0), 'stove': (0, 80, 255), 'moka_pot': (255, 0, 0), 'frying_pan': (0, 170, 0)}
    tiles = []
    for t in range(len(d.tasks)):
        ids = np.flatnonzero(d.val & (d.task == t))
        for row in ids[np.linspace(len(ids) // 4, 3 * len(ids) // 4, n_per_task).astype(int)]:
            r = torch.tensor([row], device=d.frame.device)
            im = Image.fromarray(d.frame[row].permute(1, 2, 0).cpu().numpy()).resize((512, 256), Image.NEAREST)
            dr = ImageDraw.Draw(im)
            for j, name in enumerate(NAMES):
                if not d.seen[row, j].any():
                    continue
                pred = predict(net, d, r, torch.tensor([j], device=r.device), torch.tensor([False], device=r.device))[0]
                for cam in range(2):
                    if not d.seen[row, j, cam]:
                        continue
                    for (rr, cc), wd in ((d.centre[row, j, cam].tolist(), 2), (pred[cam].tolist(), 4)):
                        x, y = (cc + 8 * cam) * 32, rr * 32
                        dr.line([(x - 10, y), (x + 10, y)], fill=colours[name], width=wd)
                        dr.line([(x, y - 10), (x, y + 10)], fill=colours[name], width=wd)
            tiles.append(np.asarray(im))
    net.train()
    gap = np.full((6, tiles[0].shape[1], 3), 255, np.uint8)
    return np.concatenate([x for tl in tiles for x in (tl, gap)][:-1], 0)


def stage1_job(out: Path, steps: int = 5000, seed: int = 0) -> None:
    # stage 1 into out/: meta.json, log.json, weights.pt, ckpt/, position.png
    import json
    import time

    import imageio.v3 as iio

    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'meta.json').exists():
        (out / 'meta.json').write_text(json.dumps({
            'gpu': torch.cuda.get_device_name(), 'steps': steps, 'seed': seed, 'readout': 'spread', 'frames': 'pixels',
            'words': WORDS, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
    d = load()
    print(f'{len(d.val)} rows ({int(d.val.sum())} held out) from {d.tasks}', flush=True)
    if (out / 'weights.pt').exists():
        weights = torch.load(out / 'weights.pt')
    else:
        run = stage1(d, steps, seed=seed, ckpt=out / 'ckpt')
        weights = run['weights']
        (out / 'log.json').write_text(json.dumps(run['log']))
        torch.save(weights, out / 'weights.pt')
    net = Verne(n_words=len(WORDS), state_dim=STATE, frames='pixels').cuda()
    net.load_state_dict(weights)
    iio.imwrite(out / 'position.png', picture(net, d))
    print(f'done: {out}\n------ FINISHED ------', flush=True)


# --- stage 2: skills -----------------------------------------------------------------------------------------------


def chunk_targets(d: namespace, now: Tensor) -> Tensor:
    # raw targets [B, CHUNK, ACTION] (module docstring): the next CHUNK steps' aiming poses relative to the hand at
    # now (held at the demo's last row), the gripper command, progress
    rows = torch.minimum(now[:, None] + torch.arange(CHUNK, device=now.device), d.last[now][:, None])
    rel = hand_mod.relative_t(d.hand_pos[now][:, None], d.hand_rot[now][:, None], d.goal_pos[rows], d.goal_rot[rows])
    return torch.cat([rel, d.grip[rows][..., None], d.progress[rows][..., None]], -1)


def action_stats(d: namespace, every: int = 5) -> tuple[Tensor, Tensor]:
    # per-number mean, std of the raw targets over every `every`-th training row's chunk
    rows = torch.tensor(np.flatnonzero(~d.val)[::every], device=d.frame.device)
    raw = torch.cat([chunk_targets(d, rows[i : i + 4096]) for i in range(0, len(rows), 4096)])
    return raw.mean((0, 1)), raw.std((0, 1)).clamp(min=1e-6)


def demo_batch(d: namespace, rows: Tensor, a_mean: Tensor, a_std: Tensor):
    # frame, state, skill words, std-unit action chunk for training rows `rows`
    return (frames(d, rows), states(d, rows), d.task_words[torch.as_tensor(d.task[rows.cpu().numpy()], device=rows.device)],
            (chunk_targets(d, rows) - a_mean) / a_std)


SNAPSHOT_EVERY = 2000  # weights-only snapshots (ckpt/weights_step<N>.pt, 28 MB) to score the arm along the run
FULL_EVERY = 5000  # full checkpoints (+ optimizer, ~85 MB) to resume / rewind


def stage2(d: namespace, init: dict, steps: int, B: int = 32, pB: int = 64, position: float = 1.0, lr: float = 3e-4,
           seed: int = 0, ckpt: Path | None = None, on_stats=None) -> dict:
    # stage 2 (module docstring) from stage 1's weights `init` (the video side; the action side starts fresh, as
    # train.py) -> {'weights', 'log': per 1000 steps (step, action train, action val, position train, pointing val),
    # 'a_mean', 'a_std'}
    torch.manual_seed(seed)
    rng, val_rng = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    dev = d.frame.device
    net = Verne(n_words=len(WORDS), state_dim=STATE, frames='pixels').to(dev)
    fresh = ('action', 'time_in', 'time_out')
    missing, unexpected = net.load_state_dict({k: v for k, v in init.items() if not k.startswith(fresh)}, strict=False)
    assert not unexpected and all(k.startswith(fresh) for k in missing), (missing, unexpected)
    a_mean, a_std = action_stats(d)
    if on_stats is not None:  # saved before training: a snapshot can be rolled out mid-run
        on_stats(a_mean, a_std)
    train_rows = np.flatnonzero(~d.val)
    point_rows = np.flatnonzero(~d.val & d.seen.any((1, 2)).cpu().numpy())
    val_rows = np.flatnonzero(d.val)
    val = [demo_batch(d, torch.tensor(val_rng.choice(val_rows, 64), device=dev), a_mean, a_std) for _ in range(4)]
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    start, log = lr_mod.resume(ckpt, steps, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, FULL_EVERY, net, opt, step, rng, log)
        if ckpt is not None and 0 < step < steps and step % SNAPSHOT_EVERY == 0:
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save({k: v.cpu() for k, v in net.state_dict().items()}, ckpt / f'weights_step{step}.pt')
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        frame, state, ins, acts = demo_batch(d, torch.tensor(rng.choice(train_rows, B), device=dev), a_mean, a_std)
        prow = torch.tensor(rng.choice(point_rows, pB), device=dev)
        names, as_target = queries(d, prow, rng)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            la = action_loss(net, frame, state, ins, acts)
            w = net.position_weights(frames(d, prow), states(d, prow), words(names, as_target))
        lp = spread_loss(w.float(), d.centre[prow, names], d.seen[prow, names])
        total = la + position * lp
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                lv = float(np.mean([action_loss(net, *v).item() for v in val]))
            pv = val_numbers(net, d, every=20)
            log.append({'step': step, 'train': la.item(), 'val': lv, 'position': lp.item(), 'point_err': pv['err'],
                        'point_right': pv['right']})
            print(f'step {step:6d}   action train {la.item():.3f}  val {lv:.3f}   position {lp.item():.4f}   '
                  f'pointing err {pv["err"]:.2f} right {pv["right"]:.3f}', flush=True)
    return {'weights': {k: v.cpu() for k, v in net.state_dict().items()}, 'log': log,
            'a_mean': a_mean.cpu(), 'a_std': a_std.cpu()}


def stage2_job(out: Path, init: Path, steps: int = 10000, seed: int = 0) -> None:
    # stage 2 into out/: meta.json, log.json, weights.pt, stats.pt (the action / state stats a rollout needs, written
    # before training), ckpt/ (step<N>.pt full every FULL_EVERY, weights_step<N>.pt every SNAPSHOT_EVERY)
    import json
    import time

    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'meta.json').exists():
        (out / 'meta.json').write_text(json.dumps({
            'gpu': torch.cuda.get_device_name(), 'steps': steps, 'seed': seed, 'init': str(init), 'position': 1.0,
            'actions': 'option B: aiming hand poses relative to the chunk start + grip + progress', 'words': WORDS,
            'snapshot_every': SNAPSHOT_EVERY, 'full_every': FULL_EVERY,
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
    d = load()
    print(f'{len(d.val)} rows ({int(d.val.sum())} held out); skills {d.skills}', flush=True)
    def save_stats(a_mean, a_std):
        torch.save({'a_mean': a_mean.cpu(), 'a_std': a_std.cpu(), 's_mean': d.s_mean.cpu(), 's_std': d.s_std.cpu()},
                   out / 'stats.pt')

    run = stage2(d, torch.load(init / 'weights.pt'), steps, seed=seed, ckpt=out / 'ckpt', on_stats=save_stats)
    (out / 'log.json').write_text(json.dumps(run['log']))
    torch.save(run['weights'], out / 'weights.pt')
    print(f'done: {out}\n------ FINISHED ------', flush=True)
