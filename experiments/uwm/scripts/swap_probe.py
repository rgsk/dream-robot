"""Does the instruction reach the actions? Swap the colour word, keep everything else, compare.

On validation episodes, at the first windows of each episode (before the arm commits to a cube),
predict the 16-action chunk three ways:
- the true prompt, noise seed s;
- the same prompt with the colour word replaced by each other colour, same seed s
  -> any difference is the text's doing;
- the true prompt with another seed -> how much actions vary from sampling alone (the yardstick).
Reports mean |difference| in data units (radians for joints), over the chunk and at its last step.
Colour swap ~ 0 => the instruction is ignored.

    PYTHONPATH=envs/uwm/src:experiments/uwm/scripts envs/uwm/.venv/bin/python \
        experiments/uwm/scripts/swap_probe.py /workspace/runs/b/model_2500.pt /workspace/runs/b/model_5000.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPTokenizer

from t5_dataset import TOKENS, make_t5_dataset
from train_t5 import REPO, SHAPE_META, build_model, dropped

COLOURS = ("red", "green", "blue", "yellow")
OFFSETS = (0, 10, 20)  # window starts within each episode: before the arm commits to a cube


def windows(val_set):
    """Validation windows starting OFFSETS steps into each validation episode."""
    starts = np.concatenate([[0], val_set.buffer.episode_ends[:-1]])
    wanted = {int(starts[e]) + o for e in np.flatnonzero(val_set.val_mask) for o in OFFSETS}
    return [i for i, (s, _) in enumerate(val_set.sampler.indices) if s in wanted]


@torch.no_grad()
def predict(model, obs, ids, mask, seed: int, drop: bool):
    torch.manual_seed(seed)
    batch = {**obs, "input_ids": ids, "attention_mask": mask}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if drop:
            with dropped(model):
                return model.sample_marginal_action(batch).float()
        return model.sample_marginal_action(batch).float()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--drop", action="store_true")
    args = p.parse_args()

    _, val_set = make_t5_dataset("t5", str(REPO / "data/cosmos_policy/t5_224/*.hdf5"),
                                 str(REPO / "data/uwm/t5_224/buffer.zarr"), SHAPE_META, seq_len=19,
                                 val_ratio=0.05)
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    samples = [val_set[i] for i in windows(val_set)]
    prompts = [val_set.commands[int(np.searchsorted(val_set.buffer.episode_ends,
                                                    val_set.sampler.indices[i][0], side="right"))]
               for i in windows(val_set)]
    obs = {k: torch.stack([s["obs"][k][:2] for s in samples]).cuda() for k in samples[0]["obs"]}

    def encode(texts):
        t = tok(texts, padding="max_length", max_length=TOKENS, return_tensors="pt")
        return t["input_ids"].cuda(), t["attention_mask"].cuda()

    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cpu")
        model, _ = build_model(ckpt["registers"], ckpt.get("patches", False), ckpt.get("joints", False))
        model.load_state_dict(ckpt["model"])
        model = model.cuda().eval()
        scale = ckpt["action_scale"].cuda()

        true = predict(model, obs, *encode(prompts), seed=0, drop=args.drop)
        resample = predict(model, obs, *encode(prompts), seed=1, drop=args.drop)
        # 3 batches shaped like `true`, window i getting its k-th other colour: same seed -> same
        # noise per window, so a difference can only come from the text
        owns = [next(c for c in COLOURS if f" {c} " in f" {pr} ") for pr in prompts]
        swapped = [predict(model, obs, *encode([pr.replace(own, [c for c in COLOURS if c != own][k])
                                                  for pr, own in zip(prompts, owns)]),
                           seed=0, drop=args.drop) for k in range(3)]

        def diff(a, b):  # data units: (normalised difference) x scale
            d = (a - b).abs() * scale
            return round(d.mean().item(), 4), round(d[:, -1].mean().item(), 4)

        row = {"ckpt": Path(path).name, "step": ckpt["step"], "drop": args.drop, "windows": len(prompts),
               "colour_swap": diff(true.repeat(3, 1, 1), torch.cat(swapped)),
               "resample": diff(true, resample)}
        print(json.dumps(row), flush=True)
        print("  (mean over chunk, at step 16) -- colour_swap ~ 0 means the text is ignored", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
