"""Is the colour in the pixels, and does our encoder keep it? Three probes on T5 top frames.

    MUJOCO_GL=egl uv run python experiments/t5/scripts/colour_probe.py

Data: the top camera's first frame after ``env.reset(seed)``, the arm up and out
of the way, for ``--train`` + ``--test`` fresh seeds (from 50000, never eval
seeds). Cached to scratch/t5/colour_probe.npz.

- **A, plain CNN** (the pixels): conv stack + global average pool, asked per
  colour "slot 0, slot 1, or absent?". Near 100% -> the colour is in the pixels.
- **B, the policy's encoder** (our architecture): BC's CameraEncoder (spatial
  softmax, 64 keypoints) + the colour one-hot concatenated *after* it, as in
  flow matching, asked "which slot holds this colour?". Chance is 50%.
- **C, the policy's encoder, one-hot inside** (FiLM): same conv stack and
  spatial softmax, but after every GroupNorm the one-hot sets a per-channel
  scale and shift, h -> (1 + γ(c)) h + β(c). The maps know the wanted colour
  *before* the softmax picks where they fire. Same head as B.

Both use random_shift(4) as in training and the same image normalisation.
Reported: held-out accuracy of "which slot holds colour c" for the two colours
present, so A and B answer the same question.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dream_robot.policies.bc.data import random_shift
from dream_robot.policies.bc.model import CameraEncoder
from dream_robot.policies.flow_matching.model import FlowConfig

CACHE = Path("scratch/t5/colour_probe.npz")
SEED0 = 50_000


def collect(n: int) -> dict[str, np.ndarray]:
    from dream_robot.sims.robosuite.tasks.instructed_pick_place.env import InstructedPickPlace

    env = InstructedPickPlace(pairs="seen", wording="template")
    palette = list(env._cfg.palette)
    images, slot_of = [], []
    started = time.time()
    try:
        for i in range(n):
            observation = env.reset(seed=SEED0 + i)
            ins = env.instruction
            # slot_of[c] = 0 / 1 for the two colours present, 2 for absent
            where = np.full(len(palette), 2, dtype=np.int64)
            where[palette.index(ins.color)] = ins.target_slot
            where[palette.index(ins.distractor_color)] = 1 - ins.target_slot
            images.append(observation.images["top"])
            slot_of.append(where)
            if (i + 1) % 250 == 0:
                print(f"collected {i + 1}/{n}  ({time.time() - started:.0f} s)")
    finally:
        env.close()
    return {"images": np.stack(images), "slot_of": np.stack(slot_of), "palette": np.array(palette)}


class PlainCNN(nn.Module):
    def __init__(self, colours: int):
        super().__init__()
        layers, c_in = [], 3
        for c_out in (32, 64, 128):
            layers += [nn.Conv2d(c_in, c_out, 3, 2, 1), nn.GroupNorm(8, c_out), nn.ReLU()]
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, colours * 3)
        self.colours = colours

    def forward(self, x):
        return self.head(self.conv(x).mean(dim=(2, 3))).view(-1, self.colours, 3)


class PolicyEncoderProbe(nn.Module):
    def __init__(self, colours: int):
        super().__init__()
        config = FlowConfig(state_dim=8, action_dim=8, arm_joints=7, grippers=1,
                            cameras=("top",), image_hw=(128, 128))
        self.encoder = CameraEncoder(config)
        self.head = nn.Sequential(nn.Linear(self.encoder.out_features + colours, 256),
                                  nn.SiLU(), nn.Linear(256, 2))

    def forward(self, x, onehot):
        return self.head(torch.cat([self.encoder(x), onehot], dim=-1))


class FiLMEncoderProbe(nn.Module):
    """CameraEncoder's layers, with the one-hot modulating every conv block (FiLM)."""

    def __init__(self, colours: int):
        super().__init__()
        config = FlowConfig(state_dim=8, action_dim=8, arm_joints=7, grippers=1,
                            cameras=("top",), image_hw=(128, 128))
        base = CameraEncoder(config)
        layers = list(base.conv)
        # conv, norm, relu per block, then the 1x1 keypoint conv
        self.blocks = nn.ModuleList(
            nn.ModuleList(layers[i:i + 3]) for i in range(0, len(layers) - 1, 3)
        )
        self.keypoint_conv = layers[-1]
        self.film = nn.ModuleList(
            nn.Linear(colours, 2 * block[0].out_channels) for block in self.blocks
        )
        self.spatial_softmax = base.spatial_softmax
        self.head = nn.Sequential(nn.Linear(base.out_features + colours, 256),
                                  nn.SiLU(), nn.Linear(256, 2))

    def forward(self, x, onehot):
        for (conv, norm, relu), film in zip(self.blocks, self.film, strict=True):
            gamma, beta = film(onehot)[:, :, None, None].chunk(2, dim=1)
            x = relu((1 + gamma) * norm(conv(x)) + beta)
        keypoints = self.spatial_softmax(self.keypoint_conv(x))
        return self.head(torch.cat([keypoints, onehot], dim=-1))


def queries(slot_of: torch.Tensor):
    """Every (image, present colour) pair -> (image index, colour index, slot)."""
    rows, cols = torch.nonzero(slot_of < 2, as_tuple=True)
    return rows, cols, slot_of[rows, cols]


def train_probe(name, model, x_tr, s_tr, x_te, s_te, *, epochs, device):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    colours = s_tr.shape[1]
    eye = torch.eye(colours, device=device)
    r_tr, c_tr, y_tr = queries(s_tr)

    def predict_slots(x, s):
        """Accuracy of 'which slot holds c' over the present colours."""
        model.eval()
        r, c, y = queries(s)
        correct = 0
        with torch.no_grad():
            for i in range(0, len(r), 512):
                rb, cb, yb = r[i:i + 512], c[i:i + 512], y[i:i + 512]
                if name == "A":
                    logits = model(x[rb])[torch.arange(len(rb)), cb][:, :2]
                else:
                    logits = model(x[rb], eye[cb])
                correct += int((logits.argmax(-1) == yb).sum())
        model.train()
        return correct / len(r)

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(r_tr), device=device)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            rb, cb, yb = r_tr[idx], c_tr[idx], y_tr[idx]
            xb = random_shift(x_tr[rb], 4)
            if name == "A":
                loss = nn.functional.cross_entropy(model(xb).reshape(-1, 3), s_tr[rb].reshape(-1))
            else:
                loss = nn.functional.cross_entropy(model(xb, eye[cb]), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{name}] epoch {epoch:3d}  loss {float(loss):.4f}  "
                  f"train {predict_slots(x_tr, s_tr):.3f}  test {predict_slots(x_te, s_te):.3f}")
    return predict_slots(x_te, s_te)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train", type=int, default=3000)
    p.add_argument("--test", type=int, default=600)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--probes", default="ABC", help="which probes to run, e.g. C")
    args = p.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = args.train + args.test
    if CACHE.exists() and len(np.load(CACHE)["images"]) >= n:
        data = dict(np.load(CACHE))
        print(f"frames from {CACHE}")
    else:
        data = collect(n)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **data)

    x = torch.from_numpy(data["images"][:n]).permute(0, 3, 1, 2).float().div(255).to(device)
    mean, std = x.mean(dim=(0, 2, 3), keepdim=True), x.std(dim=(0, 2, 3), keepdim=True)
    x = (x - mean) / std
    s = torch.from_numpy(data["slot_of"][:n]).to(device)
    x_tr, x_te, s_tr, s_te = x[:args.train], x[args.train:], s[:args.train], s[args.train:]
    colours = s.shape[1]
    print(f"{args.train} train / {args.test} test frames, palette {list(data['palette'])}")

    probes = {
        "A": ("plain CNN + global pool", PlainCNN),
        "B": ("policy encoder + one-hot after", PolicyEncoderProbe),
        "C": ("policy encoder + one-hot inside (FiLM)", FiLMEncoderProbe),
    }
    scores = {}
    for name in args.probes:
        label, cls = probes[name]
        scores[name] = train_probe(name, cls(colours), x_tr, s_tr, x_te, s_te,
                                   epochs=args.epochs, device=device)
    print(f"\nwhich slot holds colour c, held-out ({args.test} frames, chance 0.50):")
    for name, score in scores.items():
        print(f"  {name} {probes[name][0]:40s} {score:.3f}")
    print("------ FINISHED ------")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
