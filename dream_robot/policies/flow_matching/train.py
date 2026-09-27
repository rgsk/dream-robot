"""Train the flow matching policy on a recorded dataset.

    uv run python -m dream_robot.policies.flow_matching.train \
        --root data/robosuite/pick_place_two_bins --repo-id dream_robot/pick_place_two_bins \
        --out experiments/t2_fm_half100

Mirrors ``policies/act/train.py``: same episode split, same augmentation, same
chunk dataset (padded steps masked out), same checkpoint layout. Differences:

- **The loss is flow matching MSE**, not L1 against one answer (``model.py``).
- **The checkpoint is an EMA of the weights** (decay 0.999), as in Diffusion
  Policy. A sampler integrates the network ten times per plan, so step-to-step
  jitter in the weights compounds; the average is smoother than any single
  iterate.
- **The checkpoint is the last epoch, not the best validation epoch.** Flow
  loss is a noisy regression onto random velocities and a weak proxy for plan
  quality; picking on it would select noise. Both validation numbers are still
  logged each epoch.

``val_l1_rad`` is the first action of one sampled plan against the demo, the
quantity BC and ACT report. On a two-mode task it is *meant* to be worse than
theirs: a sample that picks the other bin than the held-out demo is a correct
action with a large error. The rollout is the number (``core.run``).

``--film`` additionally feeds that vector into every conv block of the camera
encoders (``model.FiLMCameraEncoder``) instead of only after them.

``--condition-words`` (T5) turns each frame's prompt into a 0/1 vector over those
words and feeds it to the net (``model.condition_vector``). Every prompt in the
dataset must hit at least one word, so a typo fails here, not as silent zeros.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from dream_robot.policies.act.data import ChunkDataset, compute_stats
from dream_robot.policies.bc.data import make_loader, random_shift, read_spec, split_episodes
from dream_robot.policies.bc.train import move, pick_device
from dream_robot.policies.flow_matching.model import (
    FlowConfig,
    FlowPolicyNet,
    condition_vector,
)

DEFAULT_ROOT = Path("data/robosuite/pick_place_cube")
DEFAULT_REPO_ID = "dream_robot/pick_place_cube"
DEFAULT_OUT = Path("experiments/fm_pick_place_cube")

#: Fixed seed for validation noise and times, so val loss moves only when the weights do.
VAL_SEED = 1234


class ConditionedChunks(ChunkDataset):
    """ACT's chunks plus each frame's prompt as a ``(len(words),)`` vector (empty if no words)."""

    def __init__(self, *args, words: tuple[str, ...], **kwargs):
        super().__init__(*args, **kwargs)
        table = self._dataset.hf_dataset
        tasks = self._dataset.meta.tasks
        # LeRobot 0.5: a DataFrame indexed by prompt with a task_index column.
        pairs = zip(tasks.index, tasks["task_index"], strict=True)
        prompts = {int(i): str(prompt) for prompt, i in pairs}
        vectors = {i: condition_vector(prompt, words) for i, prompt in prompts.items()}
        if words:
            empty = [prompts[i] for i, v in vectors.items() if not v.any()]
            if empty:
                raise ValueError(f"prompts matching none of {list(words)}: {empty}")
        task_index = np.asarray(table["task_index"])
        self.conditions = np.stack([vectors[int(i)] for i in task_index]).astype(np.float32)

    def __getitem__(self, i: int):
        return (*super().__getitem__(i), torch.from_numpy(self.conditions[self.indices[i]]))


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    validation_episodes: int = 5
    shift_pad: int = 4
    chunk: int = 16
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    seed: int = 0
    workers: int = 4
    condition_words: tuple[str, ...] = ()
    film: bool = False


def evaluate(net: FlowPolicyNet, loader, device: torch.device) -> tuple[float, float]:
    """Held-out ``(flow MSE, first-action L1 of one sampled plan, in radians)``."""
    net.eval()
    generator = torch.Generator(device=device).manual_seed(VAL_SEED)
    total_loss = total_raw = count = 0.0
    with torch.no_grad():
        for images, state, actions, pad, condition in loader:
            images, state = move(images, device), state.to(device)
            actions, pad, condition = actions.to(device), pad.to(device), condition.to(device)
            n = state.shape[0]
            total_loss += float(net.loss(images, state, actions, pad, generator, condition)) * n
            plan = net.sample(images, state, generator=generator, condition=condition)
            total_raw += float((plan[:, 0] - actions[:, 0]).abs().mean()) * n
            count += n
    return total_loss / count, total_raw / count


def write_curve(history: list[dict], path: Path) -> Path | None:
    """Loss and held-out action error by epoch. Plotting is a convenience, never a dependency."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    epochs = [h["epoch"] for h in history]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    left.plot(epochs, [h["train_loss"] for h in history], label="train")
    left.plot(epochs, [h["val_loss"] for h in history], label="validation (EMA)")
    left.set(xlabel="epoch", ylabel="flow MSE (normalised)", title="Loss", yscale="log")
    left.legend(), left.grid(alpha=0.3)

    right.plot(epochs, [h["val_l1_rad"] for h in history], color="tab:red")
    right.axhline(0.018, ls="--", c="gray", lw=1, label="controller tracking error")
    right.set(xlabel="epoch", ylabel="mean |Δaction| (rad)", title="Sampled first action")
    right.legend(), right.grid(alpha=0.3)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def train(
    *, repo_id: str, root: Path, out: Path, config: TrainConfig, device: torch.device
) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    spec = read_spec(repo_id, root)
    train_episodes, val_episodes = split_episodes(
        spec.episodes, validation=config.validation_episodes, seed=config.seed
    )
    words = tuple(config.condition_words)
    train_set = ConditionedChunks(
        repo_id, root, spec.cameras, train_episodes, config.chunk, words=words
    )
    val_set = ConditionedChunks(
        repo_id, root, spec.cameras, val_episodes, config.chunk, words=words
    )
    print(
        f"{spec.repo_id}: {spec.episodes} episodes, {spec.frames} frames, "
        f"cameras {list(spec.cameras)} at {spec.image_hw[0]}x{spec.image_hw[1]}\n"
        f"split: {len(train_episodes)} train episodes ({len(train_set)} frames) / "
        f"{len(val_episodes)} validation ({len(val_set)} frames), chunk {config.chunk}"
    )

    stats = compute_stats(train_set, spec.cameras)
    model_config = FlowConfig(
        state_dim=spec.state_dim,
        action_dim=spec.action_dim,
        arm_joints=spec.embodiment.arm_joints,
        grippers=spec.embodiment.grippers,
        cameras=spec.cameras,
        image_hw=spec.image_hw,
        chunk=config.chunk,
        condition_words=words,
        film=config.film,
    )
    if words:
        seen = np.unique(train_set.conditions[train_set.indices], axis=0)
        print(f"condition words {list(words)}: {len(seen)} distinct vectors in train")
    net = FlowPolicyNet(model_config, stats).to(device)
    ema = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay), use_buffers=True)
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
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        net.train()
        running = count = 0.0
        for images, state, actions, pad, condition in train_loader:
            images = {
                name: random_shift(frame.to(device, non_blocking=True), config.shift_pad)
                for name, frame in images.items()
            }
            state, actions, pad = state.to(device), actions.to(device), pad.to(device)
            condition = condition.to(device)

            loss = net.loss(images, state, actions, pad, condition=condition)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
            optimiser.step()
            ema.update_parameters(net)

            n = state.shape[0]
            running += float(loss.detach()) * n
            count += n
        schedule.step()

        val_loss, val_l1_rad = evaluate(ema.module, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / count,
                "val_loss": val_loss,
                "val_l1_rad": val_l1_rad,
                "lr": schedule.get_last_lr()[0],
            }
        )
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d}  train {running / count:.4f}  val {val_loss:.4f}  "
                f"sampled first-action {val_l1_rad * 1000:6.2f} mrad"
            )

    torch.save(
        {
            "state_dict": ema.module.state_dict(),
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
            "epoch": config.epochs,
            "val_loss": history[-1]["val_loss"],
            "val_l1_rad": history[-1]["val_l1_rad"],
        },
        checkpoint_path,
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
        "final_val_loss": history[-1]["val_loss"],
        "final_val_l1_rad": history[-1]["val_l1_rad"],
        "wall_clock_seconds": round(elapsed, 1),
        "history": history,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    curve = write_curve(history, out / "training_curve.png")
    print(
        f"\ntrained {config.epochs} epochs in {elapsed / 60:.1f} min\n"
        f"final held-out sampled first-action error {summary['final_val_l1_rad'] * 1000:.2f} mrad\n"
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
    p.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--workers", type=int, default=defaults.workers)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--condition-words", nargs="*", default=[],
        help="T5: e.g. red green blue yellow left right -- the target as a one-hot",
    )
    p.add_argument(
        "--film", action="store_true",
        help="also feed the condition into every conv block of the camera encoders",
    )
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
            ema_decay=args.ema_decay,
            seed=args.seed,
            workers=args.workers,
            condition_words=tuple(args.condition_words),
            film=args.film,
        ),
        device=pick_device(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
