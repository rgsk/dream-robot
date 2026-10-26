"""UVA as a local action server for our evals (dream_robot/policies/uva talks to it).

Same protocol as experiments/cosmos_policy/scripts/serve.py (plain dicts of str / bytes / lists),
but UVA reads the last 16 steps of the top camera and of the joints, and returns 16 actions
(policy mode: one pass, video never drawn):
    request  {"prompt": str, "top": bytes (16*H*W*3 uint8, oldest first, upright), "hw": [H, W],
              "proprio": [[8 floats] x 16] (oldest first),
              "wrist": bytes (as "top"; read if the model was trained with --wrist)}
    reply    {"actions": [[8 floats] x 16], "seconds": float}
             (+ "future_top": H*W*3 uint8 with --imagine: the top frame it expects 16 steps ahead
              if it plays these actions -- their forward-model mode, given the chosen chunk;
              the eval video shows it beside the real cameras; + "future_wrist" from a model
              trained with --imagine-wrist)
    "hello"  -> {"ckpt", "chunk", "step", "wrist"}

    envs/uva/.venv/bin/python experiments/uva/scripts/serve.py --ckpt /workspace/runs/b1/model.pt --port 6003 [--imagine]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
from omegaconf import open_dict

SRC = Path(__file__).resolve().parents[3] / "envs/uva/src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from uva_patch import WRIST_KEY, add_inputs  # noqa: E402
from unified_video_action.model.common.normalizer import LinearNormalizer  # noqa: E402
from unified_video_action.utils.data_utils import decode_from_sample_autoregressive  # noqa: E402

AUTHKEY = b"dream-robot"
HISTORY = 16  # their n_obs_steps; the 4 frames it looks at are steps 3, 7, 11, 15 of these


def load_policy(path: Path):
    """The EMA policy train_uva.py saved: their model from its cfg, our joint input, the weights."""
    ckpt = torch.load(open(path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = ckpt["cfg"]
    with open_dict(cfg):  # weights come from this file, not from LIBERO-10
        cfg.model.policy.autoregressive_model_params.pretrained_model_path = None
    policy = hydra.utils.instantiate(
        cfg.model.policy, task_name=cfg.task.name, task_modes=cfg.task.task_modes,
        normalizer_type=cfg.task.dataset.normalizer_type,
        language_emb_model=cfg.task.dataset.language_emb_model)
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(ckpt["normalizer"])
    policy.set_normalizer(normalizer)
    add_inputs(policy, wrist=WRIST_KEY in cfg.task.shape_meta.obs,
               imagine_wrist=bool(cfg.task.get("imagine_wrist", False)))
    policy.load_state_dict(ckpt["ema"])
    return policy.cuda().eval(), ckpt, WRIST_KEY in cfg.task.shape_meta.obs


def record_policy_calls(policy) -> dict:
    """Keep the inputs and scaled actions of the latest policy-mode call, so the same scene can be
    re-run as a forward model without repeating their preprocessing."""
    last: dict = {}
    sample = policy.model.sample_tokens

    def recording(*args, **kwargs):
        out = sample(*args, **kwargs)
        if kwargs.get("task_mode") == "policy_model":
            last.update(kwargs=kwargs, actions=out[1])
        return out

    policy.model.sample_tokens = recording
    return last


def last_frame(policy, tokens: torch.Tensor, size: tuple[int, int]) -> bytes:
    """Patch codes [4, 16, 16, 16] (4 future frames) -> the last frame decoded, at the env's size."""
    frames = decode_from_sample_autoregressive(policy.vae_model, tokens.float() / 0.2325)  # [4, 3, 256, 256]
    frame = torch.nn.functional.interpolate(frames[-1:].clamp(-1, 1), size=size, mode="bilinear")
    return ((frame[0] + 1) * 127.5).permute(1, 2, 0).byte().cpu().numpy().tobytes()


@torch.no_grad()
def imagine(policy, last: dict, size: tuple[int, int]) -> dict[str, bytes]:
    """Their dynamic_model mode on the last call: past frames + the chosen actions -> future top
    frames (steps 4, 8, 12, 16 ahead), and, with a wrist head, the wrist ones from the same decoder
    output; the last frame of each, at the env's size, uint8."""
    mar = policy.model
    seen = {}
    sample = mar.diffloss.sample

    def keep_z(z, *args, **kwargs):  # the decoder output the top head reads; one round (num_iter 1)
        seen["z"] = z
        return sample(z, *args, **kwargs)

    mar.diffloss.sample = keep_z
    try:
        kwargs = {**last["kwargs"], "task_mode": "dynamic_model", "nactions": last["actions"]}
        tokens, _ = type(mar).sample_tokens(mar, **kwargs)  # unrecorded
    finally:
        del mar.diffloss.sample
    out = {"future_top": last_frame(policy, tokens, size)}
    if hasattr(mar, "diffloss_wrist"):
        assert kwargs["num_iter"] == 1, "z of one round only"
        wrist = mar.diffloss_wrist.sample(seen["z"], kwargs["temperature"])  # [4 * 256, 16]
        out["future_wrist"] = last_frame(policy, mar.unpatchify(wrist.reshape(4, mar.seq_len, -1)), size)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", type=Path, required=True, help="model.pt written by train_uva.py")
    p.add_argument("--port", type=int, default=6003)
    p.add_argument("--imagine", action="store_true", help="also return the expected top frame 16 steps ahead")
    args = p.parse_args()
    path = args.ckpt.resolve()
    os.chdir(SRC)  # their config points at pretrained_models/... relative to the repo
    policy, ckpt, wrist = load_policy(path)
    last = record_policy_calls(policy)

    with Listener(("localhost", args.port), authkey=AUTHKEY) as listener:
        print(f"serving {path} (step {ckpt['step']}) on localhost:{args.port}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if msg == "hello":
                        conn.send({"ckpt": str(path), "chunk": 16, "step": ckpt["step"], "wrist": wrist})
                        continue
                    h, w = msg["hw"]

                    def window(name):  # [1, 16, 3, H, W] in [0, 1], as their env runner passes them
                        frames = np.frombuffer(msg[name], np.uint8).reshape(HISTORY, h, w, 3)
                        return torch.from_numpy(frames.copy()).cuda().permute(0, 3, 1, 2)[None].float().div(255)

                    obs = {"agentview_image": window("top"),
                           "joint_qpos": torch.tensor(msg["proprio"], dtype=torch.float32)[None].cuda()}
                    if wrist:
                        obs[WRIST_KEY] = window("wrist")
                    start = time.time()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        action = policy.predict_action(obs, language_goal=[msg["prompt"]])["action_pred"]
                    torch.cuda.synchronize()
                    reply = {"actions": action.float()[0].cpu().numpy().tolist(),
                             "seconds": time.time() - start}  # the policy's own time, not the drawing's
                    if args.imagine:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            reply.update(imagine(policy, last, (h, w)))
                    conn.send(reply)


if __name__ == "__main__":
    main()
