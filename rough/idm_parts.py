# The IDM idm.ipynb trained (steps 3-6), packed so video.ipynb can read actions off dreamed frames.
# Same code as idm.ipynb's cells. Frames in [0, 1] (video.ipynb's are in [-1, 1]: convert first).
import copy
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch
from torch import nn

dev = 'cuda'
C = 32  # keypoints per frame
REPO = Path(__file__).resolve().parent.parent
# idm.ipynb's train_idm() at its defaults (k=10, no extra rows): 19/20 in the oracle loop
RUN = 'cache/idm_nb/joblib/__main__--tmp-ipykernel-cell-698e59a60672/train_idm/31041588c1f890cbacd715d0fedb1250'


def down(i, o):
    # halves height and width
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1), nn.GroupNorm(8, o), nn.GELU())


def keypoints(maps):
    # [B, C, h, w] -> [B, C, 2]: each channel's softmax-weighted (x, y), in [-1, 1]
    B, _, h, w = maps.shape
    p = maps.flatten(2).softmax(-1).unflatten(2, (h, w))
    xs, ys = torch.linspace(-1, 1, w, device=dev), torch.linspace(-1, 1, h, device=dev)
    return torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)


class IDM(nn.Module):
    def __init__(self, k=10, n_past=0, wrist=False):  # k: steps between the frames = actions out
        # n_past: earlier frames also read (video.ipynb step 16); 0 = idm.ipynb's IDM
        # wrist: every frame is [top, wrist] (6 channels), each view with its own encoder (step 18)
        super().__init__()
        self.k, self.n_past, self.wrist = k, n_past, wrist
        self.enc = nn.Sequential(down(3, 32), down(32, 64), down(64, 64), nn.Conv2d(64, C, 1))
        if wrist:
            self.enc_w = nn.Sequential(down(3, 32), down(32, 64), down(64, 64), nn.Conv2d(64, C, 1))
        views = 2 if wrist else 1
        self.head = nn.Sequential(
            nn.Linear((3 + n_past) * views * C * 2 + 8, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, k * 8),
        )

    def forward(self, f0, f1, joints, past=()):
        # frames t, t+k [B, 3 (6 with wrist), 128, 128], joints [B, 8] (std units), n_past earlier
        # frames -> actions [B, k, 8] (std units)
        now, fut = self.feats(f0), self.feats(f1)
        before = [self.feats(p) for p in past]
        x = torch.cat([*before, now, fut, fut - now, joints], 1)
        return self.head(x).view(-1, self.k, 8)

    def feats(self, f):
        # one frame [B, 3 or 6, 128, 128] -> keypoints of each view, flat [B, views * C * 2]
        out = [keypoints(self.enc(f[:, :3])).flatten(1)]
        if self.wrist:
            out.append(keypoints(self.enc_w(f[:, 3:])).flatten(1))
        return torch.cat(out, 1)


def load_idm():
    # the trained k=10 IDM, eval mode, on the GPU
    net = IDM().to(dev).eval()
    result, _printed = joblib.load(REPO / RUN / 'output.pkl')  # the cache stores (result, printed text)
    net.load_state_dict(result['weights'])
    return net


# idm.ipynb steps 10-11: the oracle. Rewind the sim to show the IDM the expert's real future.
# robosuite keeps state outside MuJoCo's data (gripper rate limit, sensor timers, controllers,
# clock, our wrapper's counters); all of it is saved. robosuite rebuilds the sim at every
# env.reset, so env.mujoco is fetched on each call.
PLAIN = (np.ndarray, bool, int, float, type(None))
CLOCK = ('timestep', 'cur_time', 'done')  # robosuite's
WRAP = ('_raw', '_t', '_max_z', '_z0', '_max_cube_z', '_cube_z0')  # our task wrapper's


def parts(env):
    # the robosuite objects with their own state
    robot = env._env.robots[0]
    return [
        *env._env._observables.values(),
        *robot.gripper.values(),
        *robot.composite_controller.part_controllers.values(),
    ]


def snapshot(env, expert):
    # -> everything the next env.step reads, and the expert's phase
    model, data = env.mujoco
    physics = mujoco.MjData(model)
    mujoco.mj_copyData(physics, model, data)
    fields = [{k: v for k, v in vars(p).items() if isinstance(v, PLAIN)} for p in parts(env)]
    clock = {f: getattr(env._env, f) for f in CLOCK}
    wrap = {f: getattr(env, f) for f in WRAP if hasattr(env, f)}
    return physics, copy.deepcopy((fields, clock, wrap, expert._machine))


def restore(env, expert, snap):
    physics, rest = snap
    fields, clock, wrap, machine = copy.deepcopy(rest)
    model, data = env.mujoco
    mujoco.mj_copyData(data, model, physics)
    for p, f in zip(parts(env), fields, strict=True):
        vars(p).update(f)
    vars(env._env).update(clock)
    vars(env).update(wrap)
    expert._machine = machine


def imagine(env, expert, obs, k=10):
    # the expert's observation k steps ahead of obs (or where its episode ends), then rewind
    snap = snapshot(env, expert)
    for _ in range(k):
        res = env.step(expert(obs)[0])
        obs = res.observation
        if res.terminated or res.truncated:
            break
    restore(env, expert, snap)
    return obs
