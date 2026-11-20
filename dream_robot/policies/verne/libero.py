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

Suites (VERNE_LIBERO_SUITE, read at import): 'libero_90' (default; our scene-3 / scene-8 skills, data/libero/npz) or
'libero_spatial' (2026-11-20: the benchmark suite, data/libero/npz/libero_spatial). Spatial: every task is
put(bowl, plate) with a second identical bowl in the scene; the policy never reads which (the instruction's phrase):
the LLM POINTS at the bowl in the top camera. The point (row, col in [-1, 1], + a 'given' flag) joins the state, so
STATE is 12 there. Stage 1 asks 'plate' (no point), and 'bowl' + a point near either bowl (labels 'bowl' / 'other_bowl')
-> that bowl's centre: only the point can say which. The point is that bowl's top-camera centre in the demo's FIRST
frame (the LLM points once, before the call), jittered (JITTER patches), held for the whole episode.
"""

import os

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
SUITE = os.environ.get('VERNE_LIBERO_SUITE', 'libero_90')
VERBS = ('turn_on', 'turn_off', 'put')  # to_npz's VERBS
if SUITE == 'libero_90':
    FOLDERS = (REPO / 'data/libero/npz',)
    NAMES = ('stove_knob', 'stove', 'moka_pot', 'frying_pan')  # the suite's words (a subset of to_npz's NAMES)
    TARGETS = ('stove',)  # names that are a put target in some skill: also asked as 'to_<name>' in stage 1
    POINTED = ()  # labels asked with a point (none: no point input)
elif SUITE in ('libero_spatial', 'libero_spatial_noisy'):
    # _noisy: + the noisy replays (experiments/libero/scripts/noisy_replays.py: one per training demo, a push and the
    # driver's recovery; their own tasks in d.tasks, episodes 1000+, never held out)
    FOLDERS = (REPO / 'data/libero/npz/libero_spatial',) + (
        (REPO / 'data/libero/npz/libero_spatial_noisy',) if SUITE == 'libero_spatial_noisy' else ())
    NAMES = ('plate', 'bowl')
    TARGETS = ('plate',)
    POINTED = ('bowl', 'other_bowl')
else:
    raise ValueError(SUITE)
WORDS = VERBS + NAMES + tuple(f'to_{n}' for n in NAMES)
# label columns: the words' objects, then labels asked only by point ('other_bowl': the word 'bowl' + a point at it)
LABELS = NAMES + tuple(n for n in POINTED if n not in NAMES)
WORD_OF = [NAMES.index(n) if n in NAMES else NAMES.index('bowl') for n in LABELS]
POINT = bool(POINTED)
STATE = 9 + 3 * POINT  # 7 joints + 2 fingers (+ the point: row, col, given)
JITTER = 0.5  # the point's noise in training, patches (std per axis; the LLM's own test: ~0.35 patches off)
VAL_EPISODES = 5  # of each task's 50 demos: episodes 0-4 held out (as the LIBERO VAE)


def load(dev: str = 'cuda', limit: int | None = None) -> namespace:
    # VERNE_FRAMES_ON_CPU=1: frames stay in CPU memory (shared with the numpy array, no copy) and frames() moves
    # each batch (a small GPU: Spatial's 6 GB of frames don't fit the 4060)
    # every task's npz -> one table on the GPU: frame [N, 3, 128, 256] uint8, state [N, 9], prev [N] (the row one
    # control step back, a demo's first row itself), first [N] (its demo's first row), centre [N, LABELS, 2 cams, 2]
    # in patches, seen [N, LABELS, 2], task [N], episode [N] (per task), val [N] bool, tasks (names), s_mean / s_std
    # (train rows), task_point [tasks] (its object is pointed at). limit: rows per task (a quick check)
    parts = []
    paths = [p for f in FOLDERS for p in sorted(f.glob('*.npz'))]
    # frames straight into one preallocated array (no second full copy: Spatial + noisy replays are ~12 GB)
    lens = [len(np.load(p)['episode']) if limit is None else limit for p in paths]
    frames_all = np.empty((sum(lens), 3, 128, 256), np.uint8)
    at = 0
    for t, path in enumerate(paths):
        z = np.load(path)
        # to_npz's names -> this suite's: label columns and skill tokens by name (to_npz's list only grows; a label
        # an older npz lacks stays NaN / unseen)
        zn = [str(n) for n in z['names']]
        pix = np.concatenate([z['pix'], np.full_like(z['pix'][:, :1], np.nan)], 1)  # last column: missing
        seen = np.concatenate([z['seen'], np.zeros_like(z['seen'][:, :1])], 1)
        cols = [zn.index(n) if n in zn else -1 for n in LABELS]
        v, o, tg = (int(x) for x in z['tokens'])
        assert str(z['verbs'][v]) == VERBS[v], path
        tokens = np.array([v, NAMES.index(zn[o]) + 1, NAMES.index(zn[tg]) + 1 if tg else 0])
        n = len(z['episode']) if limit is None else limit
        frames_all[at:at + n, :, :, :128] = z['top'][:n].transpose(0, 3, 1, 2)  # [n, 3, 128, 256]: top | wrist
        frames_all[at:at + n, :, :, 128:] = z['wrist'][:n].transpose(0, 3, 1, 2)
        at += n
        state = np.concatenate([z['joints'][:n], z['gripper'][:n]], 1)
        ep = z['episode'][:n]
        part = dict(state=state, centre=pix[:n, cols] / 16, seen=seen[:n, cols],
                    task=np.full(n, t), episode=ep, name=path.stem, skill=str(z['skill']), tokens=tokens)
        for k in ('hand_pos', 'hand_rot', 'goal_pos', 'goal_rot'):
            part[k] = z[k][:n]
        part['grip'] = z['action'][:n, 6]
        parts.append(part)
    keys = ('state', 'centre', 'seen', 'task', 'episode', 'hand_pos', 'hand_rot', 'goal_pos', 'goal_rot',
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
        frame=torch.from_numpy(frames_all) if os.environ.get('VERNE_FRAMES_ON_CPU') == '1'
        else torch.tensor(frames_all, device=dev),
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
        first=torch.tensor(first, device=dev),
        task_point=torch.tensor([NAMES[int(p['tokens'][1]) - 1] in POINTED for p in parts], device=dev),
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
    # skill tokens [verb, object, target] (indices into VERBS and ('none',) + NAMES; load maps to_npz's) -> the instruction
    # multi-hot [len(WORDS)]: the verb, the object's name, and 'to_<target>' (none for turn_on / turn_off)
    v, o, tg = (int(x) for x in tokens)
    out = torch.zeros(len(WORDS))
    out[v] = 1.0
    out[len(VERBS) + o - 1] = 1.0
    if tg:
        out[len(VERBS) + len(NAMES) + tg - 1] = 1.0
    return out


def frames(d: namespace, rows: Tensor) -> Tensor:
    return d.frame[rows.to(d.frame.device)].to(rows.device, non_blocking=True).float() / 127.5 - 1


def join(prev: Tensor, now: Tensor, s_mean: Tensor, s_std: Tensor, point: Tensor | None) -> Tensor:
    # raw states [B, 9] -> the state input [B, 2 · STATE]: [previous, (point), now, (point)], states in std units;
    # point [B, 3] (row, col in [-1, 1], given flag) or None (no point: zeros)
    prev, now = (prev - s_mean) / s_std, (now - s_mean) / s_std
    if not POINT:
        return torch.cat([prev, now], 1)
    if point is None:
        point = torch.zeros(len(now), 3, device=now.device)
    return torch.cat([prev, point, now, point], 1)


def states(d: namespace, rows: Tensor, point: Tensor | None = None) -> Tensor:
    return join(d.state[d.prev[rows]], d.state[rows], d.s_mean, d.s_std, point)


def point_input(centre: Tensor) -> Tensor:
    # top-camera centre [B, 2] in patches (0..8) -> the point input [B, 3]: row, col in [-1, 1], given 1
    return torch.cat([centre / 4 - 1, torch.ones_like(centre[:, :1])], 1)


def points(d: namespace, rows: Tensor, labels: Tensor, jitter: bool = True) -> Tensor:
    # [B, 3] the point for each (row, label): for POINTED labels that label's top-camera centre in the row's demo's
    # first frame (+ JITTER noise), else zeros (no point)
    c = d.centre[d.first[rows], labels, 0]  # [B, 2] patches
    if jitter:
        c = c + JITTER * torch.randn_like(c)
    pointed = torch.tensor([n in POINTED for n in LABELS], device=rows.device)[labels]
    return point_input(c) * pointed[:, None]


def words(names: Tensor, as_target: Tensor) -> Tensor:
    # name index [B] (into NAMES) + whether to ask it as a target [B] -> the instruction multi-hot [B, len(WORDS)]
    out = torch.zeros(len(names), len(WORDS), device=names.device)
    col = len(VERBS) + names + as_target.long() * len(NAMES)
    out[torch.arange(len(names)), col] = 1.0
    return out


def label_words(labels: Tensor, as_target: Tensor) -> Tensor:
    # label index [B] (into LABELS) -> its word's multi-hot ('other_bowl' asks 'bowl')
    return words(torch.tensor(WORD_OF, device=labels.device)[labels], as_target)


def queries(d: namespace, rows: Tensor, rng: np.random.Generator) -> tuple[Tensor, Tensor]:
    # per row a random label in view in at least one camera (a POINTED one only if in the top camera at its demo's
    # first frame: where the point comes from), and (for TARGETS) at random as 'to_<name>'
    ok = d.seen[rows].any(2)  # [B, LABELS]
    pointed = torch.tensor([n in POINTED for n in LABELS], device=rows.device)
    ok &= ~pointed | d.seen[d.first[rows], :, 0]
    score = torch.tensor(rng.random(ok.shape), device=rows.device) * ok
    labels = score.argmax(1)
    can_target = torch.tensor([n in TARGETS for n in LABELS], device=rows.device)[labels]
    as_target = can_target & torch.tensor(rng.random(len(rows)) < 0.5, device=rows.device)
    return labels, as_target


def spread_loss(w: Tensor, target: Tensor, seen: Tensor) -> Tensor:
    # w [B, 2 cams, 8, 8] (position_weights), target [B, 2, 2] (row, col) in patches, seen [B, 2] -> the module
    # docstring's loss: each camera's spread, out-of-view cameras skipped
    at = torch.arange(8, device=w.device) + 0.5
    d2 = (at[:, None] - target[..., 0, None, None]) ** 2 + (at[None, :] - target[..., 1, None, None]) ** 2
    per_cam = (w * d2).sum((2, 3)) / 2  # [B, 2]
    s = seen.float()
    return (per_cam * s).sum() / s.sum().clamp(min=1)


def free_loss(pred: Tensor, target: Tensor, seen: Tensor) -> Tensor:
    # 'free' readout (step 22: one linear layer per camera on its 64 tokens -> (row, col), anywhere): squared
    # distance per camera, out-of-view cameras skipped. pred, target [B, 2, 2], seen [B, 2]
    per_cam = ((pred - target) ** 2).sum(-1)
    s = seen.float()
    return (per_cam * s).sum() / s.sum().clamp(min=1)


def position_loss(net: Verne, d: namespace, rows: Tensor, labels: Tensor, as_target: Tensor, readout: str) -> Tensor:
    # stage 1's loss for one batch of (row, label) queries, readout 'spread' or 'free'
    f, st, ins = frames(d, rows), states(d, rows, points(d, rows, labels)), label_words(labels, as_target)
    target, seen = d.centre[rows, labels], d.seen[rows, labels]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = net.position_weights(f, st, ins) if readout == 'spread' else net.position(f, st, ins, 'free')
    if readout == 'spread':
        return spread_loss(out.float(), target, seen)
    return free_loss(out.float().view(-1, 2, 2), target, seen)


def predict(net: Verne, d: namespace, rows: Tensor, labels: Tensor, as_target: Tensor,
            readout: str = 'spread') -> Tensor:
    # [B, 2 cams, 2]: spread: the weighted mean of patch centres per camera; free: its own point. Points unjittered
    st = states(d, rows, points(d, rows, labels, jitter=False))
    return net.position(frames(d, rows), st, label_words(labels, as_target), readout).view(-1, 2, 2)


@torch.no_grad()
def val_numbers(net: Verne, d: namespace, every: int = 10, readout: str = 'spread') -> dict[str, float]:
    # every `every`-th held-out row x every name in view: per camera the error to the target (patches), and 'right':
    # the prediction is nearer the named object than every other object in view in that camera; per name and overall
    net.eval()
    rows = torch.tensor(np.flatnonzero(d.val)[::every], device=d.state.device)
    out, total = {}, {'err': [], 'right': []}
    for j, name in enumerate(LABELS):
        r = rows[d.seen[rows, j].any(1)]
        if name in POINTED:
            r = r[d.seen[d.first[r], j, 0]]
        if not len(r):
            continue
        errs, rights = [], []
        for i in range(0, len(r), 256):
            b = r[i : i + 256]
            pred = predict(net, d, b, torch.full_like(b, j), torch.zeros_like(b, dtype=torch.bool), readout)
            dist = (pred[:, None] - d.centre[b]).norm(dim=-1)  # [B, LABELS, 2 cams]
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


def stage1(d: namespace, steps: int, B: int = 64, lr: float = 3e-4, seed: int = 0, ckpt: Path | None = None,
           readout: str = 'spread') -> dict:
    # stage 1 (module docstring) -> {'weights': the whole Verne on the cpu, 'log': per 500 steps}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = d.state.device
    train_rows = np.flatnonzero(~d.val & d.seen.any((1, 2)).cpu().numpy())
    net = Verne(n_words=len(WORDS), state_dim=STATE, frames='pixels').to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    start, log = lr_mod.resume(ckpt, steps, net, opt, rng)
    for step in range(start, steps + 1):
        lr_mod.checkpoint(ckpt, step, steps, 1000, net, opt, step, rng, log)
        lr_mod.set_lr(opt, lr * lr_mod.cosine(step, steps))
        rows = torch.tensor(rng.choice(train_rows, B), device=dev)
        names, as_target = queries(d, rows, rng)
        loss = position_loss(net, d, rows, names, as_target, readout)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 500 == 0:
            v = val_numbers(net, d, readout=readout)
            log.append({'step': step, 'train': loss.item()} | v)
            per = '  '.join(f'{n} {v.get(f"err_{n}", float("nan")):.2f}/{v.get(f"right_{n}", float("nan")):.2f}'
                            for n in LABELS)
            print(f'step {step:5d}  train {loss.item():.4f}  val err {v["err"]:.2f} right {v["right"]:.3f}  '
                  f'[{per}]', flush=True)
    return {'weights': {k: v.cpu() for k, v in net.state_dict().items()}, 'log': log}


@torch.no_grad()
def picture(net: Verne, d: namespace, n_per_task: int = 2, readout: str = 'spread') -> np.ndarray:
    # uint8 [H, W, 3]: held-out frames (n per task, mid-demo), each asked about every name in view; a cross per name
    # in its colour: target thin, prediction thick; 2x size
    from PIL import Image, ImageDraw

    net.eval()
    colours = {'stove_knob': (255, 200, 0), 'stove': (0, 80, 255), 'moka_pot': (255, 0, 0), 'frying_pan': (0, 170, 0),
               'plate': (0, 80, 255), 'bowl': (255, 0, 0), 'other_bowl': (0, 170, 0)}
    tiles = []
    for t in range(len(d.tasks)):
        ids = np.flatnonzero(d.val & (d.task == t))
        for row in ids[np.linspace(len(ids) // 4, 3 * len(ids) // 4, n_per_task).astype(int)]:
            r = torch.tensor([row], device=d.state.device)
            im = Image.fromarray(d.frame[row].permute(1, 2, 0).cpu().numpy()).resize((512, 256), Image.NEAREST)
            dr = ImageDraw.Draw(im)
            for j, name in enumerate(LABELS):
                if not d.seen[row, j].any() or (name in POINTED and not d.seen[d.first[row], j, 0]):
                    continue
                pred = predict(net, d, r, torch.tensor([j], device=r.device), torch.tensor([False], device=r.device),
                               readout)[0]
                for cam in range(2):
                    if not d.seen[row, j, cam]:
                        continue
                    for (rr, cc), wd in ((d.centre[row, j, cam].tolist(), 2), (pred[cam].tolist(), 4)):
                        x, y = (cc + 8 * cam) * 32, rr * 32
                        dr.line([(x - 10, y), (x + 10, y)], fill=colours.get(name, (255, 0, 0)), width=wd)
                        dr.line([(x, y - 10), (x, y + 10)], fill=colours.get(name, (255, 0, 0)), width=wd)
            tiles.append(np.asarray(im))
    net.train()
    gap = np.full((6, tiles[0].shape[1], 3), 255, np.uint8)
    return np.concatenate([x for tl in tiles for x in (tl, gap)][:-1], 0)


def stage1_job(out: Path, steps: int = 5000, seed: int = 0, readout: str = 'spread') -> None:
    # stage 1 into out/: meta.json, log.json, weights.pt, ckpt/, position.png
    import json
    import time

    import imageio.v3 as iio

    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'meta.json').exists():
        (out / 'meta.json').write_text(json.dumps({
            'gpu': torch.cuda.get_device_name(), 'steps': steps, 'seed': seed, 'readout': readout, 'frames': 'pixels',
            'suite': SUITE,
            'words': WORDS, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
    d = load()
    print(f'{len(d.val)} rows ({int(d.val.sum())} held out) from {d.tasks}', flush=True)
    if (out / 'weights.pt').exists():
        weights = torch.load(out / 'weights.pt')
    else:
        run = stage1(d, steps, seed=seed, ckpt=out / 'ckpt', readout=readout)
        weights = run['weights']
        (out / 'log.json').write_text(json.dumps(run['log']))
        torch.save(weights, out / 'weights.pt')
    net = Verne(n_words=len(WORDS), state_dim=STATE, frames='pixels').cuda()
    net.load_state_dict(weights)
    iio.imwrite(out / 'position.png', picture(net, d, readout=readout))
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
    rows = torch.tensor(np.flatnonzero(~d.val)[::every], device=d.state.device)
    raw = torch.cat([chunk_targets(d, rows[i : i + 4096]) for i in range(0, len(rows), 4096)])
    return raw.mean((0, 1)), raw.std((0, 1)).clamp(min=1e-6)


def demo_batch(d: namespace, rows: Tensor, a_mean: Tensor, a_std: Tensor):
    # frame, state (+ the point at the skill's object if its task points), skill words, std-unit action chunk
    task = torch.as_tensor(d.task[rows.cpu().numpy()], device=rows.device)
    point = None
    if POINT:  # the object's label = its word's (bowl): points() zeroes rows whose task doesn't point
        obj = d.task_words[task][:, len(VERBS):len(VERBS) + len(NAMES)].argmax(1)
        point = points(d, rows, obj) * d.task_point[task][:, None]
    return (frames(d, rows), states(d, rows, point), d.task_words[task], (chunk_targets(d, rows) - a_mean) / a_std)


SNAPSHOT_EVERY = 2000  # weights-only snapshots (ckpt/weights_step<N>.pt, 28 MB) to score the arm along the run
FULL_EVERY = 5000  # full checkpoints (+ optimizer, ~85 MB) to resume / rewind


def stage2(d: namespace, init: dict, steps: int, B: int = 32, pB: int = 64, position: float = 1.0, lr: float = 3e-4,
           seed: int = 0, ckpt: Path | None = None, on_stats=None, readout: str = 'spread') -> dict:
    # stage 2 (module docstring) from stage 1's weights `init` (the video side; the action side starts fresh, as
    # train.py) -> {'weights', 'log': per 1000 steps (step, action train, action val, position train, pointing val),
    # 'a_mean', 'a_std'}
    torch.manual_seed(seed)
    rng, val_rng = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    dev = d.state.device
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
        lp = position_loss(net, d, prow, names, as_target, readout)
        total = la + position * lp
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), lr_mod.CLIP)
        opt.step()
        if step % 1000 == 0:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                lv = float(np.mean([action_loss(net, *v).item() for v in val]))
            pv = val_numbers(net, d, every=20, readout=readout)
            log.append({'step': step, 'train': la.item(), 'val': lv, 'position': lp.item(), 'point_err': pv['err'],
                        'point_right': pv['right']})
            print(f'step {step:6d}   action train {la.item():.3f}  val {lv:.3f}   position {lp.item():.4f}   '
                  f'pointing err {pv["err"]:.2f} right {pv["right"]:.3f}', flush=True)
    return {'weights': {k: v.cpu() for k, v in net.state_dict().items()}, 'log': log,
            'a_mean': a_mean.cpu(), 'a_std': a_std.cpu()}


def stage2_job(out: Path, init: Path, steps: int = 10000, seed: int = 0, readout: str = 'spread') -> None:
    # stage 2 into out/: meta.json, log.json, weights.pt, stats.pt (the action / state stats a rollout needs, written
    # before training), ckpt/ (step<N>.pt full every FULL_EVERY, weights_step<N>.pt every SNAPSHOT_EVERY)
    import json
    import time

    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'meta.json').exists():
        (out / 'meta.json').write_text(json.dumps({
            'gpu': torch.cuda.get_device_name(), 'steps': steps, 'seed': seed, 'init': str(init), 'suite': SUITE, 'position': 1.0,
            'readout': readout,
            'actions': 'option B: aiming hand poses relative to the chunk start + grip + progress', 'words': WORDS,
            'snapshot_every': SNAPSHOT_EVERY, 'full_every': FULL_EVERY,
            'started': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=1))
    d = load()
    print(f'{len(d.val)} rows ({int(d.val.sum())} held out); skills {d.skills}', flush=True)
    def save_stats(a_mean, a_std):
        torch.save({'a_mean': a_mean.cpu(), 'a_std': a_std.cpu(), 's_mean': d.s_mean.cpu(), 's_std': d.s_std.cpu()},
                   out / 'stats.pt')

    run = stage2(d, torch.load(init / 'weights.pt'), steps, seed=seed, ckpt=out / 'ckpt', on_stats=save_stats,
                 readout=readout)
    (out / 'log.json').write_text(json.dumps(run['log']))
    torch.save(run['weights'], out / 'weights.pt')
    print(f'done: {out}\n------ FINISHED ------', flush=True)


# --- run time (the LIBERO venv): a trained stage 2 driving LIBERO's arm ---------------------------------------------

# a skill call -> to_npz's tokens (verb, object, target), the words the policy was trained with. The LLM-facing
# 'stove' of turn_on / turn_off is the knob
SKILLS = {
    ('turn_on', 'stove'): (0, 1, 0),
    ('turn_off', 'stove'): (1, 1, 0),
    ('put', 'moka_pot', 'stove'): (2, 3, 2),
    ('put', 'frying_pan', 'stove'): (2, 4, 2),
}
POS_MAX, ROT_MAX = 0.05, 0.5  # OSC_POSE output_max (the demos' controller)


class Runner:
    # a stage-2 Verne in LIBERO's sim. Every N_EXEC steps (or when the skill changes) it samples a chunk: cameras
    # (rows flipped upright, as to_npz) + [previous, now] joints and fingers + the skill words -> 16 aiming poses
    # relative to the hand now -> world poses. Every step the driver (experiments/libero/scripts/hand_check.py)
    # turns that step's pose into an OSC delta from the LIVE hand; gripper by sign
    N_EXEC = 8

    def __init__(self, run_dir: Path, weights: str = 'weights.pt', readout: str | None = None):
        self.st = {k: v.cuda() for k, v in torch.load(run_dir / 'stats.pt').items()}
        self.net = Verne(n_words=len(WORDS), state_dim=STATE, frames='pixels').cuda().eval()
        self.net.load_state_dict(torch.load(run_dir / weights))
        self.reset()

    def reset(self) -> None:
        self.plan, self.prev, self.tokens = [], None, None

    @staticmethod
    def hand(env) -> tuple[np.ndarray, np.ndarray]:
        c = env.env.robots[0].controller
        c.update(force=True)
        return np.array(c.ee_pos), np.array(c.ee_ori_mat)

    @staticmethod
    def observe(obs) -> tuple[np.ndarray, np.ndarray]:
        # -> canvas [3, 128, 256] uint8 (agentview | wrist, upright), state [9]
        top, wrist = obs['agentview_image'][::-1], obs['robot0_eye_in_hand_image'][::-1]
        canvas = np.concatenate([top, wrist], 1).transpose(2, 0, 1)
        return canvas, np.concatenate([obs['robot0_joint_pos'], obs['robot0_gripper_qpos']]).astype(np.float32)

    @staticmethod
    def command(p_now, r_now, p, r, grip) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        a = np.empty(7)
        a[:3] = np.clip((p - p_now) / POS_MAX, -1, 1)
        a[3:6] = np.clip(Rotation.from_matrix(r @ r_now.T).as_rotvec() / ROT_MAX, -1, 1)
        a[6] = 1.0 if grip > 0 else -1.0
        return a

    @torch.no_grad()
    def act(self, env, obs, tokens: tuple[int, int, int], point: tuple[float, float] | None = None) -> np.ndarray:
        # one control step's OSC command for skill `tokens`, given the latest observation; point: (row, col) in the
        # upright 128 px top camera (the LLM's), held for the call
        from .model import sample

        canvas, state = self.observe(obs)
        prev = state if self.prev is None else self.prev
        if not self.plan or tokens != self.tokens:
            frame = torch.from_numpy(canvas)[None].cuda().float() / 127.5 - 1
            pt = None if point is None else point_input(torch.tensor([point], dtype=torch.float32).cuda() / 16)
            s = join(torch.from_numpy(prev)[None].cuda(), torch.from_numpy(state)[None].cuda(), self.st['s_mean'],
                     self.st['s_std'], pt)
            ins = skill_words(tokens)[None].cuda()
            raw = (sample(self.net, frame, s, ins) * self.st['a_std'] + self.st['a_mean'])[0].cpu().numpy()
            p0, r0 = self.hand(env)
            pos, rot = hand_mod.absolute(p0[None], r0[None], raw[:, :9])
            self.plan = [(pos[k], rot[k], raw[k, 9]) for k in range(self.N_EXEC)]
            self.tokens = tokens
        p, r, g = self.plan.pop(0)
        self.prev = state
        return self.command(*self.hand(env), p, r, g)
