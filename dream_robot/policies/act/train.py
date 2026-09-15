"""Train ACT on a recorded dataset.

    uv run python -m dream_robot.policies.act.train \
        --root data/robosuite/pick_place_cube --out experiments/act_pick_place_cube

Mirrors ``policies/bc/train.py``: same episode split, same augmentation, same
L1 objective, same checkpoint layout. Two additions: the loss is over a whole
chunk (padded steps masked out) plus the CVAE's KL term, and the reported
``val_l1_rad`` is the error on the **first** action of each chunk, so it is the
same quantity BC reports and the two columns can be read side by side.

As with BC, held-out error is necessary and never sufficient; the rollout is the
number (``core.run``).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dream_robot.policies.act.data import ChunkDataset, compute_stats
from dream_robot.policies.act.model import ACTConfig, ACTPolicyNet, kl_divergence, masked_l1
from dream_robot.policies.bc.data import make_loader, random_shift, read_spec, split_episodes
from dream_robot.policies.bc.train import move, pick_device, write_curve

DEFAULT_ROOT = Path("data/robosuite/pick_place_cube")
DEFAULT_REPO_ID = "dream_robot/pick_place_cube"
DEFAULT_OUT = Path("experiments/act_pick_place_cube")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    # 1e-4, not LeRobot's 1e-5: that value is for fine-tuning a pretrained
    # backbone. This one starts from scratch.
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    validation_episodes: int = 5
    shift_pad: int = 4
    chunk: int = 50
    kl_weight: float = 10.0
    grad_clip: float = 10.0
    seed: int = 0
    workers: int = 4


def evaluate(net: ACTPolicyNet, loader, device: torch.device) -> tuple[float, float]:
    """Held-out ``(normalised chunk L1, first-action L1 in radians)``, latent at the prior mean."""
    net.eval()
    total_norm = total_raw = count = 0.0
    with torch.no_grad():
        for images, state, actions, pad in loader:
            images, state = move(images, device), state.to(device)
            actions, pad = actions.to(device), pad.to(device)
            predicted, _, _ = net(images, state)
            n = state.shape[0]
            total_norm += float(masked_l1(predicted, net.action_norm.normalize(actions), pad)) * n
            first = net.action_norm.denormalize(predicted[:, 0])
            total_raw += float((first - actions[:, 0]).abs().mean()) * n
            count += n
    net.train()
    return total_norm / count, total_raw / count


def train(
    *, repo_id: str, root: Path, out: Path, config: TrainConfig, device: torch.device
) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    spec = read_spec(repo_id, root)
    train_episodes, val_episodes = split_episodes(
        spec.episodes, validation=config.validation_episodes, seed=config.seed
    )
    train_set = ChunkDataset(repo_id, root, spec.cameras, train_episodes, config.chunk)
    val_set = ChunkDataset(repo_id, root, spec.cameras, val_episodes, config.chunk)
    print(
        f"{spec.repo_id}: {spec.episodes} episodes, {spec.frames} frames, "
        f"cameras {list(spec.cameras)} at {spec.image_hw[0]}x{spec.image_hw[1]}\n"
        f"split: {len(train_episodes)} train episodes ({len(train_set)} frames) / "
        f"{len(val_episodes)} validation ({len(val_set)} frames), chunk {config.chunk}"
    )

    stats = compute_stats(train_set, spec.cameras)
    model_config = ACTConfig(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        arm_joints=spec.embodiment.arm_joints,
        grippers=spec.embodiment.grippers,
        cameras=spec.cameras,
        image_hw=spec.image_hw,
        chunk=config.chunk,
    )
    net = ACTPolicyNet(model_config, stats).to(device)
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
        running = running_kl = count = 0.0
        for images, state, actions, pad in train_loader:
            images = {
                name: random_shift(frame.to(device, non_blocking=True), config.shift_pad)
                for name, frame in images.items()
            }
            state, actions, pad = state.to(device), actions.to(device), pad.to(device)

            predicted, mu, logvar = net(images, state, actions, pad)
            reconstruction = masked_l1(predicted, net.action_norm.normalize(actions), pad)
            kl = kl_divergence(mu, logvar)
            loss = reconstruction + config.kl_weight * kl

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimiser.step()

            n = state.shape[0]
            running += float(reconstruction.detach()) * n
            running_kl += float(kl.detach()) * n
            count += n
        schedule.step()

        val_l1, val_l1_rad = evaluate(net, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_l1": running / count,
                "train_kl": running_kl / count,
                "val_l1": val_l1,
                "val_l1_rad": val_l1_rad,
                "lr": schedule.get_last_lr()[0],
            }
        )
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
                f"epoch {epoch:3d}  train {running / count:.4f}  kl {running_kl / count:.3f}  "
                f"val {val_l1:.4f}  first-action {val_l1_rad * 1000:6.2f} mrad{marker}"
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
        f"best held-out first-action error {summary['best_val_l1_rad'] * 1000:.2f} mrad\n"
        f"checkpoint -> {checkpoint_path.resolve()}"
    )
    if curve:
        print(f"curve      -> {curve.resolve()}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    defaults = TrainConfig()
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.learning_rate)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--val-episodes", type=int, default=defaults.validation_episodes)
    p.add_argument("--shift-pad", type=int, default=defaults.shift_pad)
    p.add_argument("--chunk", type=int, default=defaults.chunk)
    p.add_argument("--kl-weight", type=float, default=defaults.kl_weight)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--workers", type=int, default=defaults.workers)
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
            chunk=args.chunk,
            kl_weight=args.kl_weight,
            seed=args.seed,
            workers=args.workers,
        ),
        device=pick_device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
