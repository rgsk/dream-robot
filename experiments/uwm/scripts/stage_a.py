"""UWM stage A: does the LIBERO-90 checkpoint load, and what does a call cost on the 4060?

Speed and memory only -- inputs are random, so the actions mean nothing. Three ways to run it:
- policy:  as built -- 10 action denoising steps, the 98 video tokens in the input as pure noise;
- dropped: the same, with the video tokens removed from the sequence (the B2 idea). This
           checkpoint never trained that way, so only its time is meaningful;
- joint:   actions and video denoised together (10 steps), plus decoding the imagined frames --
           the "full video thinking" end of the dial.

    envs/uwm/.venv/bin/python experiments/uwm/scripts/stage_a.py --ckpt envs/uwm/ckpt/libero_90.pt
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

SRC = Path(__file__).resolve().parents[3] / "envs/uwm/src"
sys.path.insert(0, str(SRC))


def forward_dropped(self, global_cond, action, action_t, next_obs, next_obs_t):
    """DualNoisePredictionNet.forward without the next-obs tokens: actions + registers only."""
    if action_t.dim() == 0:
        action_t = action_t.expand(action.shape[0]).to(dtype=torch.long, device=action.device)
    if next_obs_t.dim() == 0:
        next_obs_t = next_obs_t.expand(action.shape[0]).to(dtype=torch.long, device=action.device)
    a0, a1 = self.action_inds
    o0, o1 = self.next_obs_inds
    registers = self.registers.expand(action.shape[0], -1, -1)
    x = torch.cat((self.action_encoder(action), registers), dim=1)
    x = x + torch.cat((self.pos_embed[:, a0:a1], self.pos_embed[:, o1:]), dim=1)
    cond = torch.cat((global_cond, self.timestep_embedding(action_t, next_obs_t)), dim=-1)
    for block in self.blocks:
        x = block(x, cond)
    x = self.head(x, cond)
    return self.action_decoder(x[:, a0:a1]), None


def timed(fn, repeats: int) -> float:
    fn()  # warm-up
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / repeats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()

    with initialize_config_dir(config_dir=str(SRC / "configs"), version_base=None):
        cfg = compose(config_name="train_uwm_robomimic")
    model = instantiate(cfg.model)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    result = model.load_state_dict(ckpt["model"], strict=False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"checkpoint step {ckpt.get('step')}: missing {len(result.missing_keys)}, "
          f"unexpected {len(result.unexpected_keys)}; trainable params {trainable / 1e6:.1f}M, "
          f"all {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    net = model.noise_pred_net
    print(f"tokens: actions {net.action_inds[1]}, video {net.next_obs_inds[1] - net.next_obs_inds[0]}, "
          f"registers {net.registers.shape[1]}", flush=True)

    model = model.cuda().eval()
    meta = cfg.dataset.shape_meta.obs
    obs = {k: torch.randint(0, 256, (1, cfg.obs_num_frames, *meta[k].shape), dtype=torch.uint8).cuda()
           for k in meta}
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        policy = timed(lambda: model.sample_marginal_action(obs), args.repeats)
        joint = timed(lambda: model.obs_encoder.apply_vae(model.sample_joint(obs)[0], inverse=True),
                      args.repeats)
        original = net.forward
        net.forward = types.MethodType(forward_dropped, net)
        dropped = timed(lambda: model.sample_marginal_action(obs), args.repeats)
        net.forward = original
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"seconds per call (batch 1, fp32, {model.num_inference_steps} steps): "
          f"policy {policy:.4f}, dropped {dropped:.4f}, joint+decode {joint:.4f}; peak {peak:.2f} GB",
          flush=True)


if __name__ == "__main__":
    main()
