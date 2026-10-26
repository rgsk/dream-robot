"""UVA stage A: does the LIBERO-10 checkpoint load, and what does a call cost on the 4060?

Speed and memory only -- inputs are random, so the actions mean nothing. One way to run it:
the policy as their LIBERO runner calls it (video not generated, action head only).

    envs/uva/.venv/bin/python experiments/uva/scripts/stage_a.py --ckpt envs/uva/ckpt/libero10.ckpt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import dill
import hydra
import torch
from omegaconf import OmegaConf, open_dict

SRC = Path(__file__).resolve().parents[3] / "envs/uva/src"
sys.path.insert(0, str(SRC))


def millions(module: torch.nn.Module | None) -> float:
    return 0.0 if module is None else sum(p.numel() for p in module.parameters()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()
    ckpt = args.ckpt.resolve()
    os.chdir(SRC)  # their config points at pretrained_models/... relative to the repo

    payload = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    with open_dict(cfg):
        cfg.output_dir = "/tmp/uva_stage_a"
    policy_cfg = cfg.model.policy
    T = cfg.task.dataset.n_obs_steps
    print("task", cfg.task.name, "| n_obs_steps", T, "| n_action_steps", policy_cfg.n_action_steps)
    print("shape_meta", OmegaConf.to_container(cfg.task.shape_meta))
    print("autoregressive", OmegaConf.to_container(policy_cfg.autoregressive_model_params))

    workspace = hydra.utils.get_class(cfg.model._target_)(cfg, output_dir=cfg.output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model.to("cuda").eval()
    print(f"params (M): mar {millions(policy.model):.1f}  vae {millions(policy.vae_model):.1f}"
          f"  text {millions(policy.text_model):.1f}")

    image = cfg.task.shape_meta.obs.agentview_rgb.shape  # [C, H, W]
    goal = ["put both the alphabet soup and the tomato sauce in the basket"]

    def call() -> torch.Tensor:
        # pixels in [0, 1], as their env runner passes them (process_data scales by 255)
        obs = {"agentview_image": torch.rand(1, T, *image, device="cuda")}
        if policy.use_history_action:
            obs["past_action"] = torch.zeros(1, 16, policy.action_dim, device="cuda")
        with torch.no_grad():
            return policy.predict_action(obs, language_goal=goal)["action_pred"]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    action = call()
    torch.cuda.synchronize()
    print(f"first call {time.perf_counter() - t0:.2f} s, action_pred {tuple(action.shape)}")

    seconds = []
    for _ in range(args.calls):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - t0)
    print(f"per call: {' '.join(f'{s:.3f}' for s in seconds)} s"
          f"  (mean {sum(seconds) / len(seconds):.3f})")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    # where a call's time goes: wrap each stage with a synced timer, then rerun the calls
    import unified_video_action.policy.unified_video_action_policy as P

    spent: dict[str, float] = {}

    def timed(name, fn):
        def wrapper(*a, **k):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            spent[name] = spent.get(name, 0.0) + time.perf_counter() - t0
            return out
        return wrapper

    P.extract_text_features = timed("CLIP text", P.extract_text_features)
    P.extract_latent_autoregressive = timed("VAE encode", P.extract_latent_autoregressive)
    policy.model.forward_mae_encoder = timed("MAR encoder", policy.model.forward_mae_encoder)
    policy.model.forward_mae_decoder = timed("MAR decoder", policy.model.forward_mae_decoder)
    head = policy.model.diffactloss
    head.sample = timed(f"action head ({head.gen_diffusion.num_timesteps} steps)", head.sample)
    for _ in range(args.calls):
        call()
    total = sum(seconds) / len(seconds)
    for name, s in spent.items():
        print(f"  {name:<26} {s / args.calls:.3f} s  ({100 * s / args.calls / total:.0f}%)")
    print(f"  {'rest':<26} {total - sum(spent.values()) / args.calls:.3f} s")


if __name__ == "__main__":
    main()
