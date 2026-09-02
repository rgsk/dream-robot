"""Train BC on a recorded dataset.

    uv run python -m dream_robot.policies.bc.train \
        --root data/robosuite/pick_place_cube --epochs 60

Reads a directory and writes a checkpoint. It never learns which simulator
produced the directory, which is the entire point of the seam -- the same
command trains on Isaac data with a different ``--root``.

**The number to watch is `val_l1_rad`**, the mean absolute action error on
held-out episodes in radians. It has a natural yardstick: the expert's own
controller lags its commanded target by ~0.018 rad (see `core/record.py`). A
policy whose error is around that has learned the expert's actions about as
precisely as the actuator executes them. A policy an order of magnitude above it
has not, and no amount of rollout tuning will fix that.

What a good validation loss does *not* tell you: whether the policy can hold a
trajectory together for 200 steps. BC is trained on single frames from the
expert's own distribution, and at rollout it sees its own drift instead --
compounding error, which only `core/eval.py` can measure. Treat this script's
numbers as necessary, never sufficient.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dream_robot.policies.bc.data import (
    FrameDataset,
    compute_stats,
    make_loader,
    random_shift,
    read_spec,
    split_episodes,
)
from dream_robot.policies.bc.model import BCPolicyNet, ModelConfig

DEFAULT_ROOT = Path("data/robosuite/pick_place_cube")
DEFAULT_REPO_ID = "dream_robot/pick_place_cube"
DEFAULT_OUT = Path("experiments/bc_pick_place_cube")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_episodes: int = 5
    shift_pad: int = 4
    seed: int = 0
    workers: int = 4


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move(images: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: frame.to(device, non_blocking=True) for name, frame in images.items()}


def evaluate(net: BCPolicyNet, loader, device: torch.device) -> tuple[float, float]:
    """Held-out ``(normalised L1, raw L1 in radians)``.

    Both are reported because they answer different questions. The normalised
    one is what the optimiser sees and what compares across datasets with
    different joint ranges. The raw one is in the units the robot moves in, so
    it can be held against the controller's own tracking error and against the
    task's tolerances.
    """
    net.eval()
    total_norm = total_raw = count = 0.0
    with torch.no_grad():
        for images, state, action in loader:
            images, state = move(images, device), state.to(device)
            action = action.to(device)
            predicted = net(images, state)
            target = net.action_norm.normalize(action)
            n = state.shape[0]
            total_norm += float(nn.functional.l1_loss(predicted, target)) * n
            raw = net.action_norm.denormalize(predicted)
            total_raw += float((raw - action).abs().mean()) * n
            count += n
    net.train()
    return total_norm / count, total_raw / count


def write_curve(history: list[dict], path: Path) -> Path | None:
    """A loss curve, because a table of sixty numbers is not something you read."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:            # plotting is a convenience, never a dependency
        return None

    epochs = [h["epoch"] for h in history]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    left.plot(epochs, [h["train_l1"] for h in history], label="train")
    left.plot(epochs, [h["val_l1"] for h in history], label="validation")
    left.set(xlabel="epoch", ylabel="L1 (normalised)", title="Loss")
    left.legend(), left.grid(alpha=0.3)

    right.plot(epochs, [h["val_l1_rad"] for h in history], color="tab:red")
    # The yardstick: the controller's own lag behind its commanded target.
    right.axhline(0.018, ls="--", c="gray", lw=1, label="controller tracking error")
    right.set(xlabel="epoch", ylabel="mean |Δaction| (rad)", title="Held-out action error")
    right.legend(), right.grid(alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def train(
    *,
    repo_id: str,
    root: Path,
    out: Path,
    config: TrainConfig,
    device: torch.device,
) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    spec = read_spec(repo_id, root)
    train_episodes, val_episodes = split_episodes(
        spec.episodes, validation=config.validation_episodes, seed=config.seed
    )
    train_set = FrameDataset(repo_id, root, spec.cameras, train_episodes)
    val_set = FrameDataset(repo_id, root, spec.cameras, val_episodes)

    print(
        f"{spec.repo_id}: {spec.episodes} episodes, {spec.frames} frames, "
        f"cameras {list(spec.cameras)} at {spec.image_hw[0]}x{spec.image_hw[1]}\n"
        f"split: {len(train_episodes)} train episodes ({len(train_set)} frames) / "
        f"{len(val_episodes)} validation ({len(val_set)} frames)"
    )

    stats = compute_stats(train_set, spec.cameras)
    model_config = ModelConfig(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        arm_joints=spec.embodiment.arm_joints,
        grippers=spec.embodiment.grippers,
        cameras=spec.cameras,
        image_hw=spec.image_hw,
    )
    net = BCPolicyNet(model_config, stats).to(device)
    print(f"model: {net.parameter_count / 1e6:.2f}M parameters on {device}")

    train_loader = make_loader(
        train_set, batch_size=config.batch_size, shuffle=True, workers=config.workers
    )
    val_loader = make_loader(
        val_set, batch_size=config.batch_size, shuffle=False, workers=config.workers
    )
    optimiser = torch.optim.AdamW(
        net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)

    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "checkpoint.pt"
    history: list[dict] = []
    best = float("inf")
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        running = count = 0.0
        for images, state, action in train_loader:
            images = {
                # Augmentation on device, after the batch has landed.
                name: random_shift(frame.to(device, non_blocking=True), config.shift_pad)
                for name, frame in images.items()
            }
            state, action = state.to(device), action.to(device)

            # L1, not MSE. Deliberately the same loss ACT uses, so that when the
            # matrix compares BC against ACT the difference is the architecture
            # and not the objective. It also suits the gripper column, which is
            # bimodal at 0 and 1: L1 fits a median and commits to a mode, where
            # L2 fits a mean and sits between them with the fingers half closed.
            loss = nn.functional.l1_loss(
                net(images, state), net.action_norm.normalize(action)
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()

            n = state.shape[0]
            running += float(loss) * n
            count += n
        schedule.step()

        val_l1, val_l1_rad = evaluate(net, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_l1": running / count,
            "val_l1": val_l1,
            "val_l1_rad": val_l1_rad,
            "lr": schedule.get_last_lr()[0],
        })
        marker = ""
        if val_l1 < best:
            best = val_l1
            marker = "  <- best"
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "model_config": model_config.to_dict(),
                    "normalization": stats.to_dict(),
                    "dataset": {
                        "repo_id": spec.repo_id,
                        "fps": spec.fps,
                        "prompt": spec.prompt,
                        "episodes": spec.episodes,
                        "train_episodes": train_episodes,
                        "val_episodes": val_episodes,
                    },
                    "train_config": asdict(config),
                    "epoch": epoch,
                    "val_l1": val_l1,
                    "val_l1_rad": val_l1_rad,
                },
                checkpoint_path,
            )
        if epoch % 5 == 0 or epoch == 1 or marker:
            print(
                f"epoch {epoch:3d}  train {running / count:.4f}  val {val_l1:.4f}  "
                f"val {val_l1_rad * 1000:6.2f} mrad{marker}"
            )

    elapsed = time.time() - started
    summary = {
        "repo_id": spec.repo_id,
        "checkpoint": str(checkpoint_path),
        "parameters": net.parameter_count,
        "device": str(device),
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "train_frames": len(train_set),
        "val_frames": len(val_set),
        "train_config": asdict(config),
        "best_val_l1": best,
        "best_val_l1_rad": min(h["val_l1_rad"] for h in history),
        "final_val_l1_rad": history[-1]["val_l1_rad"],
        "wall_clock_seconds": round(elapsed, 1),
        "history": history,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    curve = write_curve(history, out / "training_curve.png")

    print(
        f"\ntrained {config.epochs} epochs in {elapsed / 60:.1f} min\n"
        f"best held-out action error {summary['best_val_l1_rad'] * 1000:.2f} mrad "
        f"(controller's own tracking lag is ~18 mrad)\n"
        f"checkpoint -> {checkpoint_path.resolve()}"
    )
    if curve:
        print(f"curve      -> {curve.resolve()}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.learning_rate)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--val-episodes", type=int, default=TrainConfig.validation_episodes)
    p.add_argument("--shift-pad", type=int, default=TrainConfig.shift_pad)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--workers", type=int, default=TrainConfig.workers)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    train(
        repo_id=args.repo_id,
        root=args.root,
        out=args.out,
        config=TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            validation_episodes=args.val_episodes,
            shift_pad=args.shift_pad,
            seed=args.seed,
            workers=args.workers,
        ),
        device=pick_device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
