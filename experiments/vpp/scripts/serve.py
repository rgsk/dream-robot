"""VPP2 (T5-trained Action expert + frozen Wan2.1-14B video) as a local action server for our T5 eval.

Same protocol as experiments/cosmos_policy/scripts/serve.py, so our ``cosmos_policy`` client plays
its chunks (dream_robot/policies/vpp2 only changes the address and the default execute):
    request  {"prompt": str, "top": bytes, "wrist": bytes (H*W*3 uint8), "hw": [H, W],
              "proprio": [8 floats]}
    reply    {"actions": [[8 floats] x 32], "seconds": float,
              "future_top", "future_wrist": bytes (H*W*3 uint8, the one-step imagined frame
              16 steps on: their video frame 4 of t0, t4 ... t64) -- with --video}
    "hello"  -> {"ckpt", "chunk", "video"}

Run from their repo root (envs/vpp2/src), PYTHONPATH=src:
    python /workspace/dream-robot/experiments/vpp/scripts/serve.py \
        --ckpt runs/t5/action/checkpoints/weights/step_002000.pt --stats runs/t5/action/dataset_stats.json \
        [--video] --port 6002

As their LIBERO eval: tokenizer 128, shift 5, seeds fixed per call; Euler steps 10 by default,
--steps 4 is as accurate on recorded demos and 0.86 s instead of 1.29 s on an A100
(profile_call.py). Actions come
back in data units (our absolute joint targets + gripper opening), no gripper flip / binarising.
"""

from __future__ import annotations

import argparse
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import call, normalize_proprio  # noqa: E402
from train_t5 import config  # noqa: E402

from vpp2.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402

AUTHKEY = b"dream-robot"
CHUNK = 32
AHEAD = 16  # imagined frame shown: their frame 4 = 16 steps on (frames are t0, t4, ... t64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="compact action checkpoint from train_t5.py")
    p.add_argument("--stats", required=True, help="dataset_stats.json from the same run")
    p.add_argument("--video", action="store_true", help="also return the one-step imagined frame")
    p.add_argument("--steps", type=int, default=10, help="action denoising steps")
    p.add_argument("--port", type=int, default=6002)
    args = p.parse_args()

    cfg = config()
    cfg.model.load_text_encoder = True
    cfg.model.tokenizer_max_len = 128
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(args.ckpt, video_checkpoint_override=cfg.model.dit_checkpoint_path)
    model = model.to("cuda").eval()
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(args.stats))
    normalizer = processor.normalizer.normalizers["action"]["default"]

    with Listener(("localhost", args.port), authkey=AUTHKEY) as listener:
        print(f"serving {args.ckpt} (video={args.video}, steps={args.steps}) on localhost:{args.port}",
              flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if msg == "hello":
                        conn.send({"ckpt": args.ckpt, "chunk": CHUNK, "video": args.video, "steps": args.steps})
                        continue
                    h, w = msg["hw"]
                    top, wrist = (np.frombuffer(msg[k], np.uint8).reshape(h, w, 3) for k in ("top", "wrist"))
                    proprio = normalize_proprio(np.asarray(msg["proprio"], np.float32), processor)
                    out, seconds = call(model, cfg, processor, top, wrist, msg["prompt"], proprio, args.video,
                                        steps=args.steps)
                    action = normalizer.backward(out["action"].reshape(1, CHUNK, -1).float())[0].numpy()
                    reply = {"actions": action.tolist(), "seconds": seconds}
                    if args.video:
                        frame = np.asarray(out["video_one_step"][AHEAD // 4])
                        half = frame.shape[1] // 2
                        for name, img in (("top", frame[:, :half]), ("wrist", frame[:, half:])):
                            reply[f"future_{name}"] = np.asarray(Image.fromarray(img).resize((w, h))).tobytes()
                    conn.send(reply)


if __name__ == "__main__":
    main()
