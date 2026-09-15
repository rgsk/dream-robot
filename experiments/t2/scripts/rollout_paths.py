"""T2 diagnostic: where does each rollout carry the cube, and does it commit?

The harness reports success and a failure bucket. This records, per eval seed,
the gripper's top-down path, where the cube ended (which bin / between / other,
and whether it is resting or still held), and the step the gripper first
committed to a side. Then it draws every path over the two bins.

    # roll out, save JSON
    MUJOCO_GL=glfw uv run python experiments/t2/scripts/rollout_paths.py run \
        --policy bc --checkpoint experiments/t2_bc_half100/checkpoint.pt \
        --out experiments/t2/paths/bc.json

    # ACT with the CVAE latent sampled from N(0, I) once per episode, not fixed at 0:
    # if the latent carries the bin choice, the bins should split.
    ... run --policy act --checkpoint ... --sample-z --out experiments/t2/paths/act_sampled_z.json

    # plot any number of runs side by side
    uv run python experiments/t2/scripts/rollout_paths.py plot \
        experiments/t2/paths/{expert,bc,act}.json --out scratch/t2/paths.png
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

EVAL_SEEDS = range(1000, 1020)
#: |gripper y| beyond which the carry has visibly picked a side (bins at ±0.20).
COMMIT_Y = 0.10


def run(args) -> None:
    import mujoco
    import torch
    from torch import nn

    from dream_robot.core.record import JointNoise
    from dream_robot.core.registry import make_env, make_policy
    from dream_robot.sims.robosuite.tasks.pick_place_cube.expert import EEF_SITE

    env = make_env(args.env)
    policy = make_policy(args.policy, env, checkpoint=args.checkpoint)
    noise = JointNoise(args.noise_sigma, env.embodiment) if args.noise_sigma > 0 else None

    shift = None
    if args.sample_z:
        net = policy._net

        class ShiftedLatent(nn.Module):
            """At rollout the latent is zeros, so latent_in(0 + z) == latent_in(z)."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.z = None

            def forward(self, x):
                return self.inner(x + self.z)

        shift = ShiftedLatent(net.latent_in)
        net.latent_in = shift
        gen = torch.Generator().manual_seed(0)

    table_y = float(env.bin_centers[0][1] + env.bin_centers[1][1]) / 2
    episodes = []
    for seed in EVAL_SEEDS:
        obs = env.reset(seed=seed)
        policy.reset()
        rng = np.random.default_rng(seed)             # as core.eval's perturb
        if shift is not None:
            shift.z = torch.randn(1, net.config.latent, generator=gen).to(policy._device)
        model, data = env.mujoco
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        target = getattr(getattr(policy, "_policy", None), "target_bin", None)
        path, commit = [], None
        for t in range(env._cfg.horizon_steps):
            action = policy(obs)
            result = env.step(action if noise is None else noise(action, rng))
            obs = result.observation
            eef = data.site_xpos[site].copy()
            path.append([round(float(eef[0]), 4), round(float(eef[1]), 4)])
            if commit is None and abs(eef[1] - table_y) > COMMIT_Y:
                commit = t
            if result.terminated or result.truncated:
                break
        cube = env.cube_pos
        placed = env.which_bin(cube)
        if placed is not None:
            where = env.bin_names[placed]
        elif env.between_bins(cube):
            where = "between"
        else:
            where = "other"
        held = bool(cube[2] > env.bin_centers[0][2] + env._cfg.resting_z_margin)
        episodes.append({
            "seed": seed,
            "success": bool(result.success),
            "failure_mode": str(result.failure_mode),
            "steps": t + 1,
            "cube_end": [round(float(v), 4) for v in cube],
            "cube_where": where,
            "cube_held": held,
            "commit_step": commit,
            "expert_target": None if target is None else env.bin_names[target],
            "path": path,
        })
        print(f"seed {seed}: {where}{' (held)' if held else ''} success={result.success} "
              f"steps={t + 1} commit_step={commit}")
    env.close()

    out = {
        "policy": args.policy,
        "env": args.env,
        "noise_sigma": args.noise_sigma,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "sample_z": bool(args.sample_z),
        "bins": {n: [float(c[0]), float(c[1])] for n, c in zip(env.bin_names, env.bin_centers)},
        "bin_inner_half": env._cfg.bins[0].inner_half,
        "summary": summarize(episodes),
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + "\n")
    print(json.dumps(out["summary"], indent=2))


def summarize(episodes) -> dict:
    commits = [e["commit_step"] for e in episodes if e["commit_step"] is not None]
    return {
        "successes": sum(e["success"] for e in episodes),
        "cube_where": dict(Counter(e["cube_where"] for e in episodes)),
        "held_at_end": sum(e["cube_held"] for e in episodes),
        "never_committed": sum(e["commit_step"] is None for e in episodes),
        "median_commit_step": float(np.median(commits)) if commits else None,
    }


def plot(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    runs = [json.loads(Path(p).read_text()) for p in args.runs]
    fig, axes = plt.subplots(1, len(runs), figsize=(max(4.2 * len(runs), 7.0), 4.4), squeeze=False)
    colours = {"right": "tab:blue", "left": "tab:orange", "between": "tab:red", "other": "0.4"}
    for ax, r in zip(axes[0], runs):
        h = r["bin_inner_half"]
        for name, (bx, by) in r["bins"].items():
            # Horizontal = world y (matches left/right in the front camera), vertical = world x.
            ax.add_patch(Rectangle((by - h, bx - h), 2 * h, 2 * h, fill=False, lw=2, ec="navy"))
            ax.text(by, bx + h + 0.01, name, ha="center", fontsize=9)
        ys = [c[1] for c in r["bins"].values()]
        ax.axvspan(min(ys) + h, max(ys) - h, color="tab:red", alpha=0.06)
        for e in r["episodes"]:
            p = np.array(e["path"])
            c = colours[e["cube_where"]]
            ax.plot(p[:, 1], p[:, 0], color=c, alpha=0.5, lw=1)
            ax.plot(e["cube_end"][1], e["cube_end"][0], "s" if not e["cube_held"] else "^",
                    color=c, ms=6, mec="k", mew=0.5)
        s = r["summary"]
        label = r["policy"] + (" (z sampled)" if r["sample_z"] else "")
        where = ", ".join(f"{k} {v}" for k, v in sorted(s["cube_where"].items()))
        ax.set_title(f"{label}: {s['successes']}/20\n{where}", fontsize=10)
        ax.set_xlim(-0.32, 0.32)
        ax.set_ylim(-0.2, 0.2)
        ax.set_aspect("equal")
        ax.set_xlabel("world y (m)  — left | right in front camera")
    axes[0][0].set_ylabel("world x (m)")
    fig.suptitle("gripper paths, eval seeds 1000–1019 · square = cube resting, triangle = still held",
                 fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"-> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--policy", required=True)
    r.add_argument("--env", default="robosuite/pick_place_two_bins")
    r.add_argument("--noise-sigma", type=float, default=0.0)
    r.add_argument("--checkpoint", type=Path, default=None)
    r.add_argument("--sample-z", action="store_true")
    r.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("plot")
    q.add_argument("runs", nargs="+")
    q.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    (run if args.cmd == "run" else plot)(args)


if __name__ == "__main__":
    main()
