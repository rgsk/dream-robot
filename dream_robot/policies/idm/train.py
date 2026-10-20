"""Train the IDM on a recorded dataset.

    uv run python -m dream_robot.policies.idm.train --k 10 \
        --root data/robosuite/instructed_pick_place_clean_n150 \
        --repo-id dream_robot/instructed_pick_place_clean_n150 --out experiments/idm/k10

ACT's loop without the CVAE: masked L1 over the chunk. Reported alongside:
``val_l1_rad``, the error on the first action (BC and ACT's column), and
``val_last_rad``, on the k-th -- the one the future frame pins down hardest.

The shift augmentation moves the two frames of a sample *together*: the signal
is the few pixels the arm moves between them, and shifting them apart would add
a fake motion as large as the real one.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dream_robot.policies.act.model import masked_l1
from dream_robot.policies.bc.data import make_loader, random_shift, read_spec, split_episodes
from dream_robot.policies.bc.train import move, pick_device, write_curve
from dream_robot.policies.idm.data import PairDataset, compute_stats
from dream_robot.policies.idm.model import IDMConfig, IDMNet

DEFAULT_ROOT = Path("data/robosuite/instructed_pick_place_clean_n150")
DEFAULT_REPO_ID = "dream_robot/instructed_pick_place_clean_n150"


@dataclass(frozen=True)
class TrainConfig:
    k: int = 10
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    validation_episodes: int = 10
    shift_pad: int = 4
    grad_clip: float = 10.0
    seed: int = 0
    workers: int = 4


def shift_together(now: dict, future: dict, pad: int) -> tuple[dict, dict]:
    """One random shift per camera, applied to both frames of every sample."""
    out_now, out_future = {}, {}
    for name in now:
        both = random_shift(torch.cat([now[name], future[name]]), pad)
        out_now[name], out_future[name] = both.chunk(2)
    return out_now, out_future


def evaluate(net: IDMNet, loader, device) -> tuple[float, float, float]:
    """Held-out (normalised chunk L1, first-action L1 rad, last-action L1 rad)."""
    net.eval()
    norm = first = last = count = 0.0
    with torch.no_grad():
        for now, state, actions, pad, future in loader:
            now, future = move(now, device), move(future, device)
            state, actions, pad = state.to(device), actions.to(device), pad.to(device)
            predicted = net(now, future, state)
            n = state.shape[0]
            norm += float(masked_l1(predicted, net.action_norm.normalize(actions), pad)) * n
            raw = net.action_norm.denormalize(predicted)
            first += float((raw[:, 0] - actions[:, 0]).abs().mean()) * n
            real = ~pad[:, -1]                  # samples whose k-th action is real
            if real.any():
                last += float((raw[real, -1] - actions[real, -1]).abs().mean()) * int(real.sum())
            count += n
    net.train()
    return norm / count, first / count, last / count


def train(*, repo_id: str, root: Path, out: Path, config: TrainConfig, device) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    spec = read_spec(repo_id, root)
    train_eps, val_eps = split_episodes(
        spec.episodes, validation=config.validation_episodes, seed=config.seed
    )
    train_set = PairDataset(repo_id, root, spec.cameras, train_eps, config.k)
    val_set = PairDataset(repo_id, root, spec.cameras, val_eps, config.k)
    print(
        f"{spec.repo_id}: {spec.episodes} episodes, {spec.frames} frames, "
        f"cameras {list(spec.cameras)}, k={config.k}\n"
        f"split: {len(train_eps)} train ({len(train_set)} samples) / "
        f"{len(val_eps)} validation ({len(val_set)})"
    )

    stats = compute_stats(train_set, spec.cameras)
    model_config = IDMConfig(
        state_dim=spec.state_dim, action_dim=spec.action_dim,
        arm_joints=spec.embodiment.arm_joints, grippers=spec.embodiment.grippers,
        cameras=spec.cameras, image_hw=spec.image_hw, k=config.k,
    )
    net = IDMNet(model_config, stats).to(device)
    print(f"model: {net.parameter_count / 1e6:.2f}M parameters on {device}")

    train_loader = make_loader(train_set, batch_size=config.batch_size, shuffle=True,
                               workers=config.workers)
    val_loader = make_loader(val_set, batch_size=config.batch_size, shuffle=False,
                             workers=config.workers)
    optimiser = torch.optim.AdamW(net.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)

    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    history: list[dict] = []
    best = float("inf")
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        running = count = 0.0
        for now, state, actions, pad, future in train_loader:
            now, future = shift_together(move(now, device), move(future, device),
                                         config.shift_pad)
            state, actions, pad = state.to(device), actions.to(device), pad.to(device)
            loss = masked_l1(net(now, future, state), net.action_norm.normalize(actions), pad)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimiser.step()
            running += float(loss.detach()) * state.shape[0]
            count += state.shape[0]
        schedule.step()

        val_l1, first, last = evaluate(net, val_loader, device)
        history.append({"epoch": epoch, "train_l1": running / count, "val_l1": val_l1,
                        "val_l1_rad": first, "val_last_rad": last,
                        "lr": schedule.get_last_lr()[0]})
        marker = ""
        if val_l1 < best:
            best, marker = val_l1, "  <- best"
            torch.save({
                "state_dict": net.state_dict(),
                "model_config": model_config.to_dict(),
                "normalization": stats.to_dict(),
                "dataset": {"repo_id": spec.repo_id, "fps": spec.fps,
                            "episodes": spec.episodes, "train_episodes": train_eps,
                            "val_episodes": val_eps},
                "train_config": asdict(config),
                "epoch": epoch, "val_l1": val_l1, "val_l1_rad": first,
                "val_last_rad": last,
            }, checkpoint_path)
        if epoch % 5 == 0 or epoch == 1 or marker:
            print(f"epoch {epoch:3d}  train {running / count:.4f}  val {val_l1:.4f}  "
                  f"first {first * 1000:6.2f} mrad  last {last * 1000:6.2f} mrad{marker}",
                  flush=True)

    elapsed = time.time() - started
    summary = {
        "repo_id": spec.repo_id, "checkpoint": str(checkpoint_path),
        "parameters": net.parameter_count, "device": str(device),
        "train_episodes": train_eps, "val_episodes": val_eps,
        "train_config": asdict(config), "best_val_l1": best,
        "wall_clock_seconds": round(elapsed, 1), "history": history,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_curve(history, out / "training_curve.png")
    print(f"\ntrained {config.epochs} epochs in {elapsed / 60:.1f} min -> {checkpoint_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    d = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=d.k)
    p.add_argument("--epochs", type=int, default=d.epochs)
    p.add_argument("--batch-size", type=int, default=d.batch_size)
    p.add_argument("--lr", type=float, default=d.learning_rate)
    p.add_argument("--val-episodes", type=int, default=d.validation_episodes)
    p.add_argument("--shift-pad", type=int, default=d.shift_pad)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    train(repo_id=a.repo_id, root=a.root, out=a.out, device=pick_device(a.device),
          config=TrainConfig(k=a.k, epochs=a.epochs, batch_size=a.batch_size,
                             learning_rate=a.lr, validation_episodes=a.val_episodes,
                             shift_pad=a.shift_pad, seed=a.seed, workers=a.workers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
