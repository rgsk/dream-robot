"""Cosmos Policy as a local action server, so our T5 eval (Python 3.14, robosuite) can call it.

Their model needs their env (Python 3.10, cu128 torch); ours can't import it. So it runs here,
in theirs, and our ``cosmos_policy`` policy (dream_robot/policies/cosmos_policy) talks to it over
a localhost socket. Run from their repo root (see pod_setup.sh):

    uv run --extra cu128 --group libero --python 3.10 python \
        /workspace/dream-robot/experiments/cosmos_policy/scripts/serve.py \
        --ckpt /root/out/.../checkpoints/iter_000001500 \
        --stats /root/data/cosmos_policy/t5_224/dataset_statistics.json \
        --embeddings /root/data/cosmos_policy/t5_224/t5_embeddings.pkl

Messages are plain dicts of str / bytes / lists, never numpy objects: numpy 2 pickles (ours)
don't load under numpy 1 and the other way round.
    request  {"prompt": str, "top": bytes, "wrist": bytes (H*W*3 uint8), "hw": [H, W],
              "proprio": [8 floats]}
    reply    {"actions": [[8 floats] x chunk], "seconds": float,
              "future_top": bytes, "future_wrist": bytes (H*W*3 uint8, imagined `chunk` steps on)}

The imagined frames are kept: on the parallel path one denoising pass draws actions, future
frames and value together, and decoding them is inside the 0.6 s stage A measured. The eval shows
them beside the real ones -- does a wrong-cube episode imagine the wrong cube, or the right one?
    "hello"  -> {"ckpt": ..., "chunk": ..., "action_dim": ...}
"""

from __future__ import annotations

import argparse
import pickle
import time
from multiprocessing.connection import Listener

import numpy as np
import torch
from PIL import Image

import cosmos_policy.experiments.robot.cosmos_utils as cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, load_dataset_stats
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

AUTHKEY = b"dream-robot"
ACTION_DIM = 8  # our 7 joints + gripper


def frame_bytes(image, size: tuple[int, int]) -> bytes:
    """An imagined frame as uint8 bytes at the env's size (w, h)."""
    image = np.asarray(image)
    image = (image[0] if image.ndim == 4 else image).astype(np.uint8)
    return np.asarray(Image.fromarray(image).resize(size)).tobytes()


def config(args) -> PolicyEvalConfig:
    return PolicyEvalConfig(
        config="cosmos_policy_t5_from_libero__inference",
        ckpt_path=args.ckpt,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=args.stats,
        t5_text_embeddings_path=args.embeddings,
        use_wrist_image=True, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True, chunk_size=args.chunk, num_open_loop_steps=args.chunk,
        trained_with_image_aug=True, use_jpeg_compression=True,  # JPEG q95, as the training frames
        flip_images=False,  # our frames are upright already
        num_denoising_steps_action=args.denoise, num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True, help="fine-tuned DCP dir (.../checkpoints/iter_N)")
    p.add_argument("--stats", required=True, help="dataset_statistics.json written by training")
    p.add_argument("--embeddings", required=True, help="t5_embeddings.pkl (all 40 T5 wordings)")
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--denoise", type=int, default=5)
    p.add_argument("--port", type=int, default=6000)
    args = p.parse_args()

    # Their inference reads the action width from a constant picked by sniffing argv
    # (LIBERO -> 7); training never used it. Ours is 8.
    cosmos_utils.ACTION_DIM = ACTION_DIM
    cfg = config(args)
    stats = load_dataset_stats(cfg.dataset_stats_path)
    assert len(stats["actions_min"]) == ACTION_DIM, stats["actions_min"]
    with open(args.embeddings, "rb") as f:
        embeddings = {k: v.float().cpu().numpy() for k, v in pickle.load(f).items()}
    model, _ = get_model(cfg)

    with Listener(("localhost", args.port), authkey=AUTHKEY) as listener:
        print(f"serving {args.ckpt} on localhost:{args.port}", flush=True)
        while True:
            with listener.accept() as conn:
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if msg == "hello":
                        conn.send({"ckpt": args.ckpt, "chunk": args.chunk, "action_dim": ACTION_DIM,
                                   "denoise": args.denoise})
                        continue
                    h, w = msg["hw"]
                    obs = {
                        "primary_image": np.frombuffer(msg["top"], np.uint8).reshape(h, w, 3),
                        "wrist_image": np.frombuffer(msg["wrist"], np.uint8).reshape(h, w, 3),
                        "proprio": np.asarray(msg["proprio"], np.float32),
                    }
                    start = time.time()
                    out = get_action(cfg, model, stats, obs, embeddings[msg["prompt"]],
                                     num_denoising_steps_action=args.denoise,
                                     generate_future_state_and_value_in_parallel=True)
                    torch.cuda.synchronize()
                    seconds = time.time() - start
                    actions = np.asarray(out["actions"], np.float32).reshape(args.chunk, ACTION_DIM)
                    future = out["future_image_predictions"]
                    conn.send({"actions": actions.tolist(), "seconds": seconds,
                               "future_top": frame_bytes(future["future_image"], (w, h)),
                               "future_wrist": frame_bytes(future["future_wrist_image"], (w, h))})


if __name__ == "__main__":
    main()
