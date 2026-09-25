"""Fine-tune SmolVLA on a recorded dataset. Our loop, LeRobot's model.

    uv run python -m dream_robot.policies.smolvla.train \
        --root data/robosuite/instructed_pick_place_clean_n150 \
        --repo-id dream_robot/instructed_pick_place --out experiments/t5/smolvla_clean_n150

**Why not ``lerobot-train``.** It parses its command line through draccus,
which Python 3.14's argparse rejects (see ``policy._config``). What it does per
step is small, so it is written out here instead, and every piece mirrors it:

1. ``build`` the policy exactly as it does at step 0 (base weights, our
   dataset's shapes and stats).
2. A ``LeRobotDataset`` with ``delta_timestamps`` from the config, so each
   sample carries the next ``chunk_size`` (50) actions and an ``action_is_pad``
   mask where the episode ran out.
3. ``batch = preprocessor(batch)`` -- normalise, tokenise the instruction.
4. ``loss, _ = policy.forward(batch)`` -- flow matching MSE on the action
   chunk, the same objective as our ``flow_matching`` policy.
5. AdamW + cosine schedule with warmup, grad clip 10: SmolVLA's own presets.

**What trains.** ``train_expert_only`` (the config default): the VLM is frozen
and the action expert, state projection and action in/out projections learn.
About 100M of the 450M parameters. Gradients still flow *through* the frozen
VLM, because the state token sits in its prefix, so activation memory is the
real cost, not optimiser state.

**Episode split, as every trainer in this repo.** Whole episodes are held out;
their flow loss is logged, with fixed noise so it only moves when the weights
do. As with ``flow_matching``, it is a weak proxy: the rollout is the number.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from dream_robot.policies.bc.data import split_episodes
from dream_robot.policies.bc.train import pick_device
from dream_robot.policies.smolvla.policy import BASE_CHECKPOINT, build, save

#: Fixed seed for validation noise and flow times.
VAL_SEED = 1234


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20_000                # SmolVLA's fine-tuning recipe
    batch_size: int = 32
    validation_episodes: int = 6
    val_every: int = 1_000
    save_every: int = 5_000
    log_every: int = 50
    workers: int = 4
    seed: int = 0
    bf16: bool = True                  # autocast on CUDA; the base was trained in bf16


def _loader(dataset, *, batch_size: int, shuffle: bool, workers: int):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        persistent_workers=workers > 0,
    )


def _forever(loader):
    while True:
        yield from loader


@torch.no_grad()
def validate(policy, pre, loader, device: torch.device, *, bf16: bool) -> float:
    """Mean flow loss on held-out episodes, same noise every time."""
    policy.eval()
    gen = torch.Generator(device=device).manual_seed(VAL_SEED)
    total, count = 0.0, 0
    for batch in loader:
        batch = pre(batch)
        b = batch["action"].shape[0]
        cfg = policy.config
        noise = torch.randn(
            b, cfg.chunk_size, cfg.max_action_dim, device=device, generator=gen
        )
        t = torch.rand(b, device=device, generator=gen) * 0.999 + 0.001
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            loss, _ = policy.forward(batch, noise=noise, time=t)
        total += float(loss) * b
        count += b
    policy.train()
    return total / max(count, 1)


def train(
    root: Path, repo_id: str, out: Path, cfg: TrainConfig, *,
    checkpoint: str = BASE_CHECKPOINT, device: torch.device,
) -> Path:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    torch.manual_seed(cfg.seed)
    out.mkdir(parents=True, exist_ok=True)
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    train_eps, val_eps = split_episodes(
        meta.total_episodes, validation=cfg.validation_episodes, seed=cfg.seed
    )

    policy, pre, post = build(checkpoint, meta=meta, device=device)
    policy.train()
    deltas = resolve_delta_timestamps(policy.config, meta)
    train_set = LeRobotDataset(repo_id, root=root, episodes=train_eps, delta_timestamps=deltas)
    val_set = LeRobotDataset(repo_id, root=root, episodes=val_eps, delta_timestamps=deltas)
    train_loader = _loader(train_set, batch_size=cfg.batch_size, shuffle=True, workers=cfg.workers)
    val_loader = _loader(val_set, batch_size=cfg.batch_size, shuffle=False, workers=cfg.workers)

    params = [p for p in policy.parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in params)
    total = sum(p.numel() for p in policy.parameters())
    opt_cfg = policy.config.get_optimizer_preset()
    optimizer = torch.optim.AdamW(
        params, lr=opt_cfg.lr, betas=opt_cfg.betas, eps=opt_cfg.eps,
        weight_decay=opt_cfg.weight_decay,
    )
    scheduler = policy.config.get_scheduler_preset().build(optimizer, cfg.steps)
    bf16 = cfg.bf16 and device.type == "cuda"

    print(f"{meta.total_episodes} episodes ({len(train_eps)} train / {len(val_eps)} val), "
          f"{len(train_set)} train frames; {trainable / 1e6:.0f}M of {total / 1e6:.0f}M params "
          f"train; batch {cfg.batch_size}, {cfg.steps} steps, bf16 {bf16}", flush=True)
    (out / "train_config.json").write_text(json.dumps(
        {**asdict(cfg), "root": str(root), "repo_id": repo_id, "checkpoint": checkpoint,
         "train_episodes": train_eps, "val_episodes": val_eps}, indent=2))

    history: list[dict] = []
    running, t0 = 0.0, time.time()
    batches = _forever(train_loader)
    for step in range(1, cfg.steps + 1):
        batch = pre(next(batches))
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            loss, _ = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, opt_cfg.grad_clip_norm)
        optimizer.step()
        scheduler.step()
        running += loss.item()

        if step % cfg.log_every == 0:
            row = {
                "step": step,
                "loss": running / cfg.log_every,
                "grad_norm": float(grad_norm),
                "lr": scheduler.get_last_lr()[0],
                "seconds": round(time.time() - t0, 1),
            }
            if step == cfg.log_every and device.type == "cuda":
                row["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if step % cfg.val_every == 0 or step == cfg.steps:
                row["val_loss"] = validate(policy, pre, val_loader, device, bf16=bf16)
            history.append(row)
            running = 0.0
            rate = step / (time.time() - t0)
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                           for k, v in row.items())
                  + f" | {rate:.2f} it/s, eta {(cfg.steps - step) / rate / 60:.0f} min", flush=True)
            (out / "history.json").write_text(json.dumps(history, indent=1))

        if step % cfg.save_every == 0 or step == cfg.steps:
            path = save(policy, pre, post, out / "checkpoints" / f"{step:06d}" / "pretrained_model")
            print(f"saved {path}", flush=True)

    last = out / "checkpoints" / "last"
    last.unlink(missing_ok=True)
    last.symlink_to(f"{cfg.steps:06d}", target_is_directory=True)
    return last / "pretrained_model"


def main(argv: list[str] | None = None) -> int:
    defaults = TrainConfig()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", default=BASE_CHECKPOINT)
    p.add_argument("--steps", type=int, default=defaults.steps)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--val-episodes", type=int, default=defaults.validation_episodes)
    p.add_argument("--val-every", type=int, default=defaults.val_every)
    p.add_argument("--save-every", type=int, default=defaults.save_every)
    p.add_argument("--log-every", type=int, default=defaults.log_every)
    p.add_argument("--workers", type=int, default=defaults.workers)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    cfg = TrainConfig(
        steps=args.steps, batch_size=args.batch_size, validation_episodes=args.val_episodes,
        val_every=args.val_every, save_every=args.save_every, log_every=args.log_every,
        workers=args.workers, seed=args.seed, bf16=not args.no_bf16,
    )
    path = train(args.root, args.repo_id, args.out, cfg,
                 checkpoint=args.checkpoint, device=pick_device(args.device))
    print(f"done -> {path}")
    return 0


if __name__ == "__main__":
    # DataLoader workers under forkserver re-import __main__; the guard stops a restart.
    raise SystemExit(main())
