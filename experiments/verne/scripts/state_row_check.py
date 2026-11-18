"""Step 12 check (CPU, random weights): the state [previous, now] as a cross-attention row on both sides.

- shapes through loss + sample, param count
- get_batch: previous = one row back, clamped to now at a demo's first row
- the action side reads the state directly: with the video keys and values held fixed, changing the
  state still changes the velocity (step 11: it could not, the state was only a video token)
"""

import numpy as np
import torch

from dream_robot.policies.verne.data import CHUNK, Demos, get_batch
from dream_robot.policies.verne.model import Verne, loss, sample
from dream_robot.policies.verne.vae import Z

torch.manual_seed(0)
net = Verne()
print('params', sum(p.numel() for p in net.parameters()))

B = 5
frame, state = torch.randn(B, Z, 16, 32), torch.randn(B, 16)
ins, acts = torch.zeros(B, 6), torch.randn(B, CHUNK, 8)
ins[:, 1] = ins[:, 4] = 1
print('loss', loss(net, frame, state, ins, acts).item())
print('sample', tuple(sample(net, frame, state, ins).shape))

# previous step in get_batch: two demos of 3 rows, state = row index, so prev = now - 1 unless first
N = 6
d = Demos(
    latent=torch.zeros(N, Z, 16, 32),
    state=torch.arange(N, dtype=torch.float32)[:, None].repeat(1, 8),
    instruction=torch.zeros(N, 6),
    action=torch.zeros(N, 8),
    episode=np.array([0, 0, 0, 1, 1, 1]),
    s_mean=torch.zeros(8),
    s_std=torch.ones(8),
    a_mean=torch.zeros(8),
    a_std=torch.ones(8),
)
b = get_batch(d, np.array([0, 1]), 200, np.random.default_rng(0))
prev, now = b.state[:, 0], b.state[:, 8]
first = (now == 0) | (now == 3)
assert torch.all(prev[first] == now[first]) and torch.all(prev[~first] == now[~first] - 1)
print('get_batch prev ok:', sorted({(int(p), int(n)) for p, n in zip(prev, now)}))

# the action side's own read of the state
with torch.no_grad():
    kvs = net.see(frame, state, ins)
    x, tau = torch.randn(B, CHUNK, 8), torch.rand(B)
    v1 = net.velocity(kvs, x, tau, ins, state)
    v2 = net.velocity(kvs, x, tau, ins, state + 1)
print('velocity change from the state row alone (video k,v fixed):', (v1 - v2).abs().max().item())
