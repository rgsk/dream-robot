"""Where do VPP2's 1.4 s per call go, and what do cheap fixes buy (no retraining)?

Times each part of one infer_action call (image VAE, CLIP, prompt encoding by the 11B UMT5, the
one 14B pass that builds the video K/V cache, each action denoising step), then variants:
prompt encoded once and passed as context, and fewer action steps (10 / 4 / 2 / 1). For each
variant also the joint error and gripper vs the demo on recorded frames, so speed is not bought
with wrong actions.

    cd envs/vpp2/src && PYTHONPATH=src ../.venv/bin/python $EXP/profile_call.py \
        --ckpt runs/t5/b/checkpoints/weights/step_000500.pt --stats runs/t5/b/dataset_stats.json
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage_a import normalize_proprio  # noqa: E402
from train_t5 import config  # noqa: E402

from vpp2.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from vpp2.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from vpp2.libero.eval_helpers import compose_libero_camera_views  # noqa: E402

DATA = Path("/workspace/dream-robot/data/cosmos_policy/t5_224")
PARTS = ("_encode_input_image_latents_tensor", "_encode_clip_fea_from_image", "encode_prompt",
         "_build_frozen_video_cache", "_predict_action_noise_with_cache")
times: dict[str, list[float]] = defaultdict(list)


def timed(model, name):
    fn = getattr(model, name)

    def wrapper(*a, **k):
        torch.cuda.synchronize()
        start = time.time()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        times[name].append(time.time() - start)
        return out
    setattr(model, name, wrapper)


def jpeg(group, key, t):
    return np.array(Image.open(io.BytesIO(group[key][t].tobytes())).convert("RGB"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--stats", required=True)
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
    for name in PARTS:
        timed(model, name)
    shapes = [m["shape"] for m in processor.shape_meta["images"]]

    # recorded samples: 4 demos, at the grasp and 20 steps before it
    samples = []
    for path in sorted(DATA.glob("*_demo.hdf5"))[:4]:
        prompt = path.name[: -len("_demo.hdf5")].replace("_", " ")
        with h5py.File(path) as f:
            g = f["data/demo_0"]
            actions, states = g["actions"][:], g["robot_states"][:]
            close = int(np.flatnonzero(actions[:, 7] < 0.5)[0])
            for t in (close - 20, close - 2):
                rgb = compose_libero_camera_views(
                    {"image": jpeg(g, "obs/agentview_rgb_jpeg", t), "wrist_image": jpeg(g, "obs/eye_in_hand_rgb_jpeg", t)},
                    concatenation="horizontal", image_shapes=shapes, num_cameras=2, preprocess_mode="training_exact")
                x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to("cuda", model.torch_dtype) * (2.0 / 255.0) - 1.0
                samples.append((prompt, x, normalize_proprio(states[t].astype(np.float32), processor),
                                actions[t: t + 32]))

    contexts = {}
    with torch.no_grad():
        for prompt, *_ in samples:
            if prompt not in contexts:
                contexts[prompt] = model.encode_prompt(DEFAULT_PROMPT.format(task=prompt))

    def run(prompt, x, proprio, steps, cached):
        kw = dict(input_image=x, action_horizon=32, proprio=proprio, num_inference_steps=steps,
                  sigma_shift=5.0, seed=42, video_seed=43, num_video_frames=17)
        if cached:
            kw["prompt"] = None
            kw["context"], kw["context_mask"] = contexts[prompt]
        else:
            kw["prompt"] = DEFAULT_PROMPT.format(task=prompt)
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out = model.infer_action(**kw)
        torch.cuda.synchronize()
        return out, time.time() - start

    run(*samples[0][:3], 10, False)  # warm-up
    times.clear()
    for variant, steps, cached in (("as eval (prompt each call), 10 steps", 10, False),
                                   ("cached prompt, 10 steps", 10, True), ("cached prompt, 4 steps", 4, True),
                                   ("cached prompt, 2 steps", 2, True), ("cached prompt, 1 step", 1, True)):
        times.clear()
        totals, joint, grip = [], [], []
        for prompt, x, proprio, truth in samples:
            out, sec = run(prompt, x, proprio, steps, cached)
            pred = normalizer.backward(out["action"].reshape(1, 32, -1).float())[0].numpy()[: len(truth)]
            totals.append(sec)
            joint.append(np.abs(pred[:, :7] - truth[:, :7]).mean())
            grip.append(np.abs(np.clip(pred[:, 7], 0, 1) - truth[:, 7]).mean())
        n = len(samples)
        parts = "  ".join(f"{k.strip('_').split('_')[1] if k.startswith('_') else k}="
                          f"{sum(v) / n * 1000:.0f}ms" for k, v in times.items())
        print(f"{variant:38s} total {np.mean(totals) * 1000:5.0f} ms | joint err {np.mean(joint):.3f} rad, "
              f"gripper err {np.mean(grip):.3f} | per call: {parts}", flush=True)


if __name__ == "__main__":
    main()
