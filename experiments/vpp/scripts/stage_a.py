"""Stage A on the pod: the LIBERO VPP2 checkpoints on T5 frames -- does it load, how fast, what does it imagine?

Run inside the VPP2 env, from its repo root (envs/vpp2/src on the pod; see pod_setup.sh):

    python /workspace/dream-robot/experiments/vpp/scripts/stage_a.py \
        --data /workspace/dream-robot/data/cosmos_policy/t5_224 --out /workspace/stage_a

1. Build their LIBERO policy exactly as their eval does (evaluate.worker_config): Wan2.1-14B
   Video-10k (frozen) + 1B Action-30k, LIBERO min/max stats, tokenizer 128, 10 Euler steps,
   shift 5. Time + peak memory per call, with and without the one-step video.
2. Zero-shot on our frames: top + wrist (224, side by side, as their training_exact), our
   instruction in their prompt template, proprio = the middle of LIBERO's state range (ours is
   joints, theirs EEF pose -- meaningless either way here). Its actions mean nothing for our
   robot; its *one-step video* does: does it imagine the arm going to the named cube?
   Saves now | imagined +16 / +64 steps | real +16 / +64, top over wrist, per sample.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from vpp2.datasets.lerobot.processors.vpp2_processor import VPP2Processor
from vpp2.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from vpp2.libero.eval_helpers import compose_libero_camera_views
from vpp2.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

FRAME_STRIDE = 4  # their 17 video frames are t0, t4, ..., t64


def jpeg(group, key: str, t: int) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(group[key][t].tobytes())).convert("RGB"))


def normalize_proprio(proprio: np.ndarray, processor: VPP2Processor) -> torch.Tensor:
    """Their eval's _normalize_proprio (vpp2/libero/eval_libero_single.py, which imports the
    LIBERO simulator, not installed here)."""
    key = processor.shape_meta["state"][0]["key"]
    batch = {"state": {key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    return processor.normalizer.forward(processor.action_state_transform(batch))["state"][key]


def build(weights: Path, repo: Path):
    """Their worker_config (vpp2/libero/evaluate.py), minus the LIBERO simulator parts."""
    cfg = OmegaConf.load(repo / "configs/libero/train_action.yaml")
    weights = weights.resolve()  # their loader resolves a relative video path from the action file's folder
    wan = weights / "Wan2.1-I2V-14B-480P"
    cfg.model.model_id = str(wan)
    cfg.model.tokenizer_model_id = str(wan / "google/umt5-xxl")
    cfg.model.dit_checkpoint_path = str(weights / "libero/video_step010000.pt")
    cfg.model.load_text_encoder = True
    cfg.model.tokenizer_max_len = 128
    cfg.model.skip_dit_load_from_pretrain = True
    cfg.model.action_dit_pretrained_path = None
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(weights / "libero/action_step030000.pt"),
                          video_checkpoint_override=str(weights / "libero/video_step010000.pt"))
    model = model.to("cuda").eval()
    processor: VPP2Processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(repo / "configs/libero/dataset_stats.json")))
    return cfg, model, processor


def call(model, cfg, processor, top, wrist, prompt, proprio, video: bool, seed: int = 42, steps: int = 10):
    shapes = [m["shape"] for m in processor.shape_meta["images"]]
    rgb = compose_libero_camera_views({"image": top, "wrist_image": wrist}, concatenation="horizontal",
                                      image_shapes=shapes, num_cameras=2, preprocess_mode="training_exact")
    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to("cuda", model.torch_dtype) * (2.0 / 255.0) - 1.0
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model.infer_action(prompt=DEFAULT_PROMPT.format(task=prompt), input_image=x, action_horizon=32,
                                 proprio=proprio, num_inference_steps=steps, sigma_shift=5.0, seed=seed,
                                 video_seed=seed + 1, num_video_frames=17, return_one_step_video=video)
    torch.cuda.synchronize()
    return out, time.time() - start


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, required=True, help="data/cosmos_policy/t5_224 (our HDF5)")
    p.add_argument("--weights", type=Path, default=Path("weights"), help="their downloaded weights/ tree")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--samples", type=int, default=12, help="start + a third in, one per instruction file")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    repo = Path.cwd()

    start = time.time()
    cfg, model, processor = build(args.weights, repo)
    report = {"load_seconds": round(time.time() - start, 1),
              "load_gb": round(torch.cuda.memory_allocated() / 2**30, 2)}
    print(report, flush=True)

    stats = json.loads((repo / "configs/libero/dataset_stats.json").read_text())["state"]["default"]
    middle = (np.array(stats["global_min"]) + np.array(stats["global_max"])) / 2
    proprio = normalize_proprio(middle.astype(np.float32), processor)

    files = sorted(args.data.glob("*_demo.hdf5"))
    rows = []
    for n in range(args.samples):
        path = files[n % len(files)]
        prompt = path.name[: -len("_demo.hdf5")].replace("_", " ")
        with h5py.File(path) as f:
            group = f["data/demo_0"]
            steps = len(group["actions"])
            t = 0 if n < len(files) else steps // 3
            cams = ("obs/agentview_rgb_jpeg", "obs/eye_in_hand_rgb_jpeg")
            now = [jpeg(group, k, t) for k in cams]
            real = {d: [jpeg(group, k, min(t + d, steps - 1)) for k in cams] for d in (16, 64)}
        if n == 0:  # timing: three calls each way on the first sample
            report["seconds_action_only"] = [round(call(model, cfg, processor, *now, prompt, proprio, False)[1], 3)
                                             for _ in range(3)]
            report["seconds_with_video"] = [round(call(model, cfg, processor, *now, prompt, proprio, True)[1], 3)
                                            for _ in range(3)]
            report["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            print(report, flush=True)
        out, _ = call(model, cfg, processor, *now, prompt, proprio, True)
        video = out["video_one_step"]  # 17 PIL frames, [top | wrist] side by side
        imagined = {d: np.asarray(video[d // FRAME_STRIDE]) for d in (16, 64)}

        def column(pair):  # top over wrist, 224 each
            return np.concatenate([np.asarray(Image.fromarray(a).resize((224, 224))) for a in pair], 0)

        def split(frame):  # their [top | wrist] -> (top, wrist)
            w = frame.shape[1] // 2
            return frame[:, :w], frame[:, w:]

        grid = np.concatenate([column(now), column(split(imagined[16])), column(real[16]),
                               column(split(imagined[64])), column(real[64])], 1)
        name = f"{n:02d}_{path.name[:-10]}_t{t}.png"
        Image.fromarray(grid).save(args.out / name)
        rows.append({"file": name, "prompt": prompt, "t": t,
                     "actions_first3": np.round(out["action"].reshape(-1, 7)[:3].numpy(), 3).tolist()})
        print(f"{name}: now | imagined +16 | real +16 | imagined +64 | real +64", flush=True)
    report["zero_shot"] = rows
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
