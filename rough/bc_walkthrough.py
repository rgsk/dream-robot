import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("dream_robot/pick_place_cube", root="data/robosuite/pick_place_cube")

state = np.asarray(ds.hf_dataset["observation.state"], dtype=np.float32)
action = np.asarray(ds.hf_dataset["action"], dtype=np.float32)

print(state.shape, action.shape)
print(state[0])
print(action[0])

print("constant :", np.abs(action - action.mean(0)).mean().round(4))
print("constant per-dim:", np.abs(action - action.mean(0)).mean(0).round(4))
print("identity :", np.abs(action - state).mean().round(4))
print("identity per-dim:", np.abs(action - state).mean(0).round(4))


import torch
from torch import nn

torch.manual_seed(0)
episode = np.asarray(ds.hf_dataset["episode_index"])

# hold out 5 whole episodes, not 5 random frames -- neighbouring frames at 30 Hz
# are nearly identical, so a frame split leaks the answer into validation
val_eps = np.random.default_rng(0).permutation(25)[:5]

is_val = np.isin(episode, val_eps)

Xtr, Ytr = torch.tensor(state[~is_val]), torch.tensor(action[~is_val])
Xva, Yva = torch.tensor(state[is_val]),  torch.tensor(action[is_val])
print("train", len(Xtr), "val", len(Xva))

net = nn.Linear(8, 8)
opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
train = False
if train:
    for step in range(2001):
        i = torch.randint(0, len(Xtr), (256,))
        loss = (net(Xtr[i]) - Ytr[i]).abs().mean()      # L1
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            with torch.no_grad():
                val = (net(Xva) - Yva).abs().mean()
            print(f"{step:5d}  train {loss.item():.4f}  val {val.item():.4f}")

    with torch.no_grad():
        print("per-dim val:", (net(Xva) - Yva).abs().mean(0).numpy().round(4))



import matplotlib.pyplot as plt

# decode the whole top camera into RAM once: 5288 x 3 x 128 x 128 uint8 = 260 MB, ~5 s
top = np.zeros((len(ds), 3, 128, 128), dtype=np.uint8)
for i in range(len(ds)):
    top[i] = (np.asarray(ds[i]["observation.images.top"]) * 255).astype(np.uint8)
print(top.shape, top.dtype)

rows = np.flatnonzero(episode == 2)
picks = rows[[0, len(rows) // 3, 2 * len(rows) // 3, -1]]
print(f'{picks=}')
fig, ax = plt.subplots(1, 4, figsize=(11, 3))
for a, r in zip(ax, picks):
    a.imshow(top[r].transpose(1, 2, 0))
    a.axis("off")
    a.set_title(f"t={r - rows[0]}")
fig.savefig("scratch/top_try.png", dpi=110, bbox_inches="tight")
print('saved figure')