"""Stage A on the pod: encode our instructions, then see what the LIBERO checkpoint imagines on T5.

Run inside the Cosmos Policy env, from its repo root (see pod_setup.sh):

    uv run --extra cu128 --group libero --python 3.10 python \
        /workspace/dream-robot/experiments/cosmos_policy/scripts/stage_a.py \
        --data /root/data/cosmos_policy/t5_224 --out /workspace/stage_a

1. T5 text embeddings for every T5 wording (prompts.json, 40) -> <data>/t5_embeddings.pkl.
   The only use of the 11B text encoder: the pickle comes home, later pods skip it.
2. Their README sample (a LIBERO observation): the model loads and runs; time + memory.
3. Zero-shot on our frames: the LIBERO checkpoint fed T5 frames + instruction. Its actions
   mean nothing for our robot (their 7-D delta EEF, its proprio = LIBERO's middle), but its
   *predicted future frames* do: does it already imagine the arm going to the named cube?
   Saves now | imagined (16 steps on) | real (16 steps on), top over wrist, per sample.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

from cosmos_policy.datasets.t5_embedding_utils import generate_t5_embeddings
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig

PROMPTS = Path(__file__).resolve().parents[1] / "prompts.json"
LIBERO = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
CHUNK = 16  # the LIBERO checkpoint's chunk: its future frame is 16 steps ahead


def frame(group, key: str, t: int) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(group[key][t].tobytes())))


def config() -> PolicyEvalConfig:
    return PolicyEvalConfig(
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path=LIBERO,
        config_file="cosmos_policy/config/config.py",
        dataset_stats_path=f"{LIBERO}/libero_dataset_statistics.json",
        t5_text_embeddings_path=f"{LIBERO}/libero_t5_embeddings.pkl",
        use_wrist_image=True, use_proprio=True, normalize_proprio=True,
        unnormalize_actions=True, chunk_size=CHUNK, num_open_loop_steps=CHUNK,
        trained_with_image_aug=True, use_jpeg_compression=True,
        flip_images=False,  # our frames are upright already
        num_denoising_steps_action=5, num_denoising_steps_future_state=1,
        num_denoising_steps_value=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    # 1. text embeddings
    embeddings_path = args.data / "t5_embeddings.pkl"
    if not embeddings_path.exists():
        start = time.time()
        prompts = json.loads(PROMPTS.read_text())
        embeddings = generate_t5_embeddings(prompts)
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)
        report["t5_seconds"] = round(time.time() - start, 1)
        del embeddings
        torch.cuda.empty_cache()
    print(f"embeddings -> {embeddings_path}", flush=True)

    # 2. load + their sample
    cfg = config()
    stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    model, _ = get_model(cfg)
    with open(embeddings_path, "rb") as f:
        ours = pickle.load(f)
    sample = "cosmos_policy/experiments/robot/libero/sample_libero_10_observation.pkl"
    with open(sample, "rb") as f:
        observation = pickle.load(f)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        start = time.time()
        out = get_action(cfg, model, stats, observation,
                         "put both the alphabet soup and the tomato sauce in the basket",
                         num_denoising_steps_action=5,
                         generate_future_state_and_value_in_parallel=True)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    report["libero_sample"] = {
        "actions_shape": list(np.asarray(out["actions"]).shape),
        "seconds_per_call": [round(t, 3) for t in times],
        "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(report["libero_sample"], flush=True)

    # 3. zero-shot on T5 frames: start and a third of the way in, one file per instruction
    middle_proprio = (stats["proprio_min"] + stats["proprio_max"]) / 2
    files = sorted(args.data.glob("*_demo.hdf5"))
    rows = []
    for n in range(args.samples):
        path = files[n % len(files)]
        prompt = path.name[: -len("_demo.hdf5")].replace("_", " ")
        with h5py.File(path) as f:
            group = f["data/demo_0"]
            steps = len(group["actions"])
            t = 0 if n < len(files) else steps // 3
            top, wrist = frame(group, "obs/agentview_rgb_jpeg", t), frame(group, "obs/eye_in_hand_rgb_jpeg", t)
            ahead = min(t + CHUNK, steps - 1)
            real = (frame(group, "obs/agentview_rgb_jpeg", ahead),
                    frame(group, "obs/eye_in_hand_rgb_jpeg", ahead))
        obs = {"primary_image": top, "wrist_image": wrist, "proprio": middle_proprio}
        out = get_action(cfg, model, stats, obs, ours[prompt].float().cpu().numpy(),
                         num_denoising_steps_action=5,
                         generate_future_state_and_value_in_parallel=True)
        imagined = {k: np.asarray(v) for k, v in out["future_image_predictions"].items()}
        imagined = {k: (v[0] if v.ndim == 4 else v).astype(np.uint8) for k, v in imagined.items()}
        size = top.shape[:2][::-1]
        column = lambda a, b: np.concatenate([np.array(Image.fromarray(a).resize(size)),
                                              np.array(Image.fromarray(b).resize(size))], 0)
        grid = np.concatenate([column(top, wrist),
                               column(imagined["future_image"], imagined["future_wrist_image"]),
                               column(*real)], 1)
        name = f"{n:02d}_{path.name[:-10]}_t{t}.png"
        Image.fromarray(grid).save(args.out / name)
        rows.append({"file": name, "prompt": prompt, "t": t})
        print(f"{name}: now | imagined +{CHUNK} | real +{CHUNK}", flush=True)
    report["zero_shot"] = rows
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
