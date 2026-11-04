# The pieces uwm.ipynb built, packed so uwm2.ipynb can start at its own question.
# Same code as uwm.ipynb's cells (step numbers in the comments), with the notebook's globals
# (TOP, STATE, s_mean, ...) gathered into one `Data` that each function takes.
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor, nn

device = 'cuda'
GRID, D = 8, 128  # 8 x 8 tokens per frame, each a D-dim vector
REPO = Path(__file__).resolve().parent.parent


def load_table(name: str) -> tuple[NDArray, NDArray, NDArray]:
    # out: state [N, 8], action [N, 8], episode [N]; straight from the table, no video decode
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f'dream_robot/{name}', root=REPO / 'data/robosuite' / name)
    return (
        np.asarray(ds.hf_dataset['observation.state'], dtype=np.float32),
        np.asarray(ds.hf_dataset['action'], dtype=np.float32),
        np.asarray(ds.hf_dataset['episode_index'], dtype=np.int64),
    )


def load_top(name: str) -> NDArray[np.uint8]:
    # out: [N, 3, 128, 128] uint8, every top-camera frame decoded once (step 2; wrist unused)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(f'dream_robot/{name}', root=REPO / 'data/robosuite' / name)
    top = np.zeros((len(ds), 3, 128, 128), dtype=np.uint8)
    for i in range(len(ds)):
        top[i] = (ds[i]['observation.images.top'] * 255).round().to(torch.uint8).numpy()
    return top


def rows_at(episode: NDArray[np.int64], offsets: list[int] | range) -> NDArray[np.int64]:
    # [N, len(offsets)]: the row of step t + o for every row t, clipped to t's own demo (step 1)
    first = np.searchsorted(episode, episode, side='left')
    last = np.searchsorted(episode, episode, side='right') - 1
    idx = np.arange(len(episode))[:, None] + np.asarray(offsets)
    return np.clip(idx, first[:, None], last[:, None]).astype(np.int64)


@dataclass
class Data:
    # everything on the GPU that training, scoring and the sim need (step 12's globals)
    TOP: Tensor  # [N, 3, 128, 128] uint8
    STATE: Tensor  # [N, 8]
    ACTION: Tensor  # [N, 8]
    PAST: Tensor  # [N, 4] rows seen: t-12, t-8, t-4, t
    CHUNK: Tensor  # [N, 16] rows acted: t .. t+15
    AFTER: Tensor  # [N, 1] row imagined: t+16
    REAL: Tensor  # [N, 16] False where the chunk ran past the demo's end
    A_MEAN: Tensor  # [8], train demos only
    A_STD: Tensor
    S_MEAN: Tensor
    S_STD: Tensor


def to_gpu(
    state: NDArray, action: NDArray, episode: NDArray, top: NDArray, train: NDArray[np.bool_]
) -> Data:
    # train: [N] which rows the action/state stats come from
    g = lambda x: torch.tensor(x, device=device)
    past, chunk = rows_at(episode, [-12, -8, -4, 0]), rows_at(episode, range(16))
    real = chunk == np.arange(len(episode))[:, None] + np.arange(16)
    return Data(
        TOP=g(top), STATE=g(state), ACTION=g(action),
        PAST=g(past), CHUNK=g(chunk), AFTER=g(rows_at(episode, [16])), REAL=g(real),
        A_MEAN=g(action[train].mean(0)), A_STD=g(action[train].std(0)),
        S_MEAN=g(state[train].mean(0)), S_STD=g(state[train].std(0)),
    )  # fmt: skip


class Patches(nn.Module):
    # uva step 3: a seen frame -> 64 tokens (step 5)
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.GELU(),  # -> [B, 32, 64, 64]
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GELU(),  # -> [B, 64, 32, 32]
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GELU(),  # -> [B, 128, 16, 16]
            nn.Conv2d(128, D, 4, 2, 1),  # -> [B, D, 8, 8]
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, 3, 128, 128] in [0, 1];  out: [B, 64, D], one token per patch, row by row
        return self.conv(x).flatten(2).transpose(1, 2)


def cut(x: Tensor) -> Tensor:
    # [B, 3, 128, 128] -> [B, 64, 768]: each 16 x 16 patch's pixels in a row, same order as Patches
    return F.unfold(x, 16, stride=16).transpose(1, 2)


class UWM(nn.Module):
    # step 11; the state stats are buffers now (not saved), so weights load across notebooks
    def __init__(self, d: Data, layers: int = 4):
        super().__init__()
        self.register_buffer('s_mean', d.S_MEAN, persistent=False)
        self.register_buffer('s_std', d.S_STD, persistent=False)
        self.patches = Patches()
        self.frame_in = nn.Linear(3 * 16 * 16, D)
        self.act_in = nn.Linear(8, D)
        self.tau_in = nn.Sequential(nn.Linear(1, D), nn.SiLU(), nn.Linear(D, D))
        self.joints_in = nn.Linear(8, D)
        self.when = nn.Embedding(12 + 16 + 1, D)  # row = step offset + 12
        self.where = nn.Embedding(GRID * GRID, D)
        self.role = nn.Embedding(4, D)  # seen frame, noisy frame, action, joints
        layer = nn.TransformerEncoderLayer(
            D, 4, 4 * D, 0.0, 'gelu', batch_first=True, norm_first=True
        )
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.frame_out = nn.Linear(D, 3 * 16 * 16)
        self.act_out = nn.Linear(D, 8)
        for head in (self.frame_out, self.act_out):  # step 10
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self, seen: Tensor, joints: Tensor, x_f: Tensor, x_a: Tensor, tau_v: Tensor, tau_a: Tensor
    ) -> tuple[Tensor, Tensor]:
        # seen: [B, 4, 3, 128, 128] in [0, 1];  joints: [B, 8] radians
        # x_f: [B, 64, 768] noisy patches;  x_a: [B, 16, 8] noisy actions;  tau_v, tau_a: [B]
        # out: the two heads, [B, 64, 768] and [B, 16, 8]
        B = seen.shape[0]
        when, where, role = self.when.weight, self.where.weight, self.role.weight
        grids = self.patches(seen.flatten(0, 1)).unflatten(0, (B, 4))  # [B, 4, 64, D]
        seen_tok = grids + when[[0, 4, 8, 12]][:, None] + where + role[0]
        tag_v = self.tau_in(tau_v[:, None])[:, None]
        tag_a = self.tau_in(tau_a[:, None])[:, None]
        frame_tok = self.frame_in(x_f) + tag_v + when[28] + where + role[1]
        act_tok = self.act_in(x_a) + tag_a + when[12:28] + role[2]
        joints_tok = self.joints_in((joints - self.s_mean) / self.s_std) + when[12] + role[3]
        seq = torch.cat([seen_tok.flatten(1, 2), frame_tok, act_tok, joints_tok[:, None]], 1)
        out = self.body(seq)  # [B, 337, D]
        return self.frame_out(out[:, 256:320]), self.act_out(out[:, 320:336])


class UWMxx(UWM):
    # step 27: both heads say the clean answer; forward returns the derived velocities
    def forward(
        self, seen: Tensor, joints: Tensor, x_f: Tensor, x_a: Tensor, tau_v: Tensor, tau_a: Tensor
    ) -> tuple[Tensor, Tensor]:
        clean_f, clean_a = super().forward(seen, joints, x_f, x_a, tau_v, tau_a)
        left_v = (1 - tau_v).clamp(min=0.05)[:, None, None]  # noise left in the frame, floored
        left_a = (1 - tau_a).clamp(min=0.05)[:, None, None]
        return (clean_f - x_f) / left_v, (clean_a - x_a) / left_a


def batch_loss_xx(net: UWMxx, d: Data, rows: Tensor) -> tuple[Tensor, Tensor]:
    # step 27: plain MSE of each raw head against the clean thing, each part at its own random tau
    clean_f = cut(d.TOP[d.AFTER[rows, 0]].float() / 255 * 2 - 1)  # [B, 64, 768] in [-1, 1]
    clean_a = (d.ACTION[d.CHUNK[rows]] - d.A_MEAN) / d.A_STD  # [B, 16, 8] std units
    noise_f, noise_a = torch.randn_like(clean_f), torch.randn_like(clean_a)
    tau_v, tau_a = torch.rand(2, len(rows), device=device)
    x_f = (1 - tau_v[:, None, None]) * noise_f + tau_v[:, None, None] * clean_f
    x_a = (1 - tau_a[:, None, None]) * noise_a + tau_a[:, None, None] * clean_a
    seen = d.TOP[d.PAST[rows]].float() / 255
    f_hat, a_hat = UWM.forward(net, seen, d.STATE[rows], x_f, x_a, tau_v, tau_a)
    keep = d.REAL[rows][..., None].float()  # [B, 16, 1]
    loss_a = (((a_hat - clean_a) ** 2) * keep).sum() / (keep.sum() * 8)
    return F.mse_loss(f_hat, clean_f), loss_a


@torch.no_grad()
def val_loss_xx(net: UWMxx, d: Data, rows: Tensor) -> tuple[float, float]:
    # out: batch_loss_xx over all of rows; same tau and noise on every call
    with torch.random.fork_rng(devices=[device]):  # don't disturb training's random stream
        torch.manual_seed(0)
        sums = [0.0, 0.0]
        for j in range(0, len(rows), 256):
            r = rows[j : j + 256]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                losses = batch_loss_xx(net, d, r)
            for k in range(2):
                sums[k] += losses[k].item() * len(r)
    return sums[0] / len(rows), sums[1] / len(rows)


@torch.no_grad()
def sample_chunk(net: UWM, d: Data, rows: Tensor, steps: int = 10) -> Tensor:
    # step 14's policy job: frame left as noise (tau_v 0), chunk denoised; out: [B, 16, 8] radians
    B = len(rows)
    x_f = torch.randn(B, 64, 768, device=device)
    tau_v = torch.zeros(B, device=device)
    x_a = torch.randn(B, 16, 8, device=device)
    seen = d.TOP[d.PAST[rows]].float() / 255
    for k in range(steps):
        tau_a = torch.full((B,), k / steps, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, v_a = net(seen, d.STATE[rows], x_f, x_a, tau_v, tau_a)
        x_a = x_a + v_a.float() / steps
    return x_a * d.A_STD + d.A_MEAN


def make_sim():
    # step 16: the T1 sim, robosuite's import-time warnings muted
    import logging
    import os

    os.environ.setdefault('MUJOCO_GL', 'egl')  # render without a window; before mujoco loads
    logging.disable(logging.WARNING)
    import robosuite  # noqa: F401

    logging.disable(logging.NOTSET)
    logging.getLogger('robosuite_logs').setLevel(logging.ERROR)
    from dream_robot.core.registry import make_env

    return make_env('robosuite/pick_place_cube')


@torch.no_grad()
def predict(net: UWM, d: Data, seen: list[NDArray], obs, steps: int = 10) -> NDArray:
    # step 16: every top frame since reset (newest last) -> [16, 8], a chunk from the policy job
    now = len(seen) - 1
    ims = np.stack([seen[max(0, now + o)] for o in (-12, -8, -4, 0)])  # [4, H, W, 3]
    past_ims = torch.from_numpy(ims).permute(0, 3, 1, 2)[None].to(device).float() / 255
    joints = torch.from_numpy(obs.state)[None].to(device)
    x_f = torch.randn(1, 64, 768, device=device)  # frame left as noise
    tau_v = torch.zeros(1, device=device)
    x_a = torch.randn(1, 16, 8, device=device)
    for k in range(steps):
        tau_a = torch.full((1,), k / steps, device=device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, v_a = net(past_ims, joints, x_f, x_a, tau_v, tau_a)
        x_a = x_a + v_a.float() / steps
    chunk = (x_a * d.A_STD + d.A_MEAN)[0].cpu().numpy()
    chunk[:, 7] = np.clip(chunk[:, 7], 0.0, 1.0)  # gripper commands must stay in [0, 1]
    return chunk


def rollout(env, net: UWM, d: Data, seed: int, n_exec: int) -> tuple[bool, str, int]:
    # step 16: predict a chunk, play its first n_exec actions blind, predict again
    # out: (success, failure mode, steps taken)
    obs = env.reset(seed=seed)
    seen = [obs.images['top']]
    plan = []
    for step in range(600):
        if not plan:
            plan = list(predict(net, d, seen, obs)[:n_exec])
        r = env.step(plan.pop(0))
        obs = r.observation
        seen.append(obs.images['top'])
        if r.terminated or r.truncated:
            break
    return r.success, str(r.failure_mode), step + 1
