"""UWM as a local action server for our T5 eval (dream_robot/policies/uwm talks to it).

Same protocol as experiments/cosmos_policy/scripts/serve.py (plain dicts of str / bytes / lists),
but UWM reads the last two frames per camera and returns 16 actions, no imagined frames:
    request  {"prompt": str, "top": bytes, "wrist": bytes (2*H*W*3 uint8, older first), "hw": [H, W],
              "proprio": [[8 floats] x 2] (older first; read if the model was trained with --joints)}
    reply    {"actions": [[8 floats] x 16], "seconds": float}
             (+ "future_top", "future_wrist": H*W*3 uint8 with --joint: the imagined frame after the
              chunk, decoded by their VAE -- the eval video shows it beside the real cameras)
    "hello"  -> {"ckpt", "chunk", "drop", "registers"}

    PYTHONPATH=envs/uwm/src:experiments/uwm/scripts envs/uwm/.venv/bin/python \
        experiments/uwm/scripts/serve.py --ckpt /workspace/runs/b/model.pt [--drop] --port 6001

--drop runs with the video tokens removed (only meaningful for a model trained with --video-drop).
--joint denoises the future frames together with the actions (their sample_joint), instead of
holding the frames at pure noise (sample_marginal_action, their policy mode).
"""

from __future__ import annotations

import argparse
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
from PIL import Image
from transformers import CLIPTokenizer

from t5_dataset import TOKENS
from train_t5 import build_model, dropped

AUTHKEY = b"dream-robot"


@torch.no_grad()
def imagined(model, latents, size: tuple[int, int]) -> dict:
    """The last imagined frame per camera (views sorted: agentview = top, eye_in_hand = wrist)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        images = model.obs_encoder.apply_vae(latents.clone(), inverse=True, microbatch_size=4)
    images = images[0, :, :, -1].float().clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()
    return {f"future_{cam}": np.asarray(Image.fromarray(img).resize(size)).tobytes()
            for cam, img in zip(("top", "wrist"), images)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="model.pt written by train_t5.py")
    p.add_argument("--drop", action="store_true", help="act with the video tokens removed")
    p.add_argument("--joint", action="store_true", help="denoise video and actions together")
    p.add_argument("--port", type=int, default=6001)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    joints = ckpt.get("joints", False)
    model, _ = build_model(ckpt["registers"], ckpt.get("patches", False), joints)
    model.load_state_dict(ckpt["model"])
    model = model.cuda().eval()
    scale, offset = ckpt["action_scale"].cuda(), ckpt["action_offset"].cuda()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    tokens: dict[str, tuple] = {}

    with Listener(("localhost", args.port), authkey=AUTHKEY) as listener:
        print(f"serving {args.ckpt} (step {ckpt['step']}, drop={args.drop}, joint={args.joint}) "
          f"on localhost:{args.port}",
              flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if msg == "hello":
                        conn.send({"ckpt": args.ckpt, "chunk": model.action_len, "drop": args.drop,
                                   "joint": args.joint,
                                   "registers": ckpt["registers"]})
                        continue
                    if msg["prompt"] not in tokens:
                        t = tokenizer(msg["prompt"], padding="max_length", max_length=TOKENS,
                                      return_tensors="pt")
                        tokens[msg["prompt"]] = (t["input_ids"].cuda(), t["attention_mask"].cuda())
                    h, w = msg["hw"]
                    obs = {
                        key: torch.from_numpy(np.frombuffer(msg[cam], np.uint8).reshape(1, 2, h, w, 3).copy()).cuda()
                        for key, cam in (("agentview_rgb", "top"), ("eye_in_hand_rgb", "wrist"))
                    }
                    obs["input_ids"], obs["attention_mask"] = tokens[msg["prompt"]]
                    if joints:
                        state = torch.tensor(msg["proprio"], dtype=torch.float32)
                        obs["robot_states"] = ((state - ckpt["state_offset"]) / ckpt["state_scale"])[None].cuda()
                    start = time.time()
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        if args.drop:
                            with dropped(model):
                                action = model.sample_marginal_action(obs)
                        elif args.joint:
                            latents, action = model.sample_joint(obs)
                        else:
                            action = model.sample_marginal_action(obs)
                    seconds = time.time() - start  # the policy's own time, not the drawing's
                    action = (action.float()[0] * scale + offset).cpu().numpy()
                    reply = {"actions": action.tolist(), "seconds": seconds}
                    if args.joint:
                        reply.update(imagined(model, latents, (w, h)))
                    conn.send(reply)


if __name__ == "__main__":
    main()
