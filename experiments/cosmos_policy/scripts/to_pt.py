"""A training checkpoint (DCP folder) -> one .pt file that their inference loader can read.

Their get_model reads a local checkpoint with easy_io.load, i.e. a single torch file, and applies
it with strict=False -- a key mismatch would load silently and we'd evaluate the wrong weights.
So the .pt is checked against the LIBERO policy .pt, which loaded with 0 missing: every network
tensor there must be here with the same shape, and some must have changed (it was fine-tuned).
TransformerEngine's ``*._extra_state`` entries (one per block, 28) are bookkeeping, not weights:
training never saves them, and the LIBERO load already reported them as the 28 unused.
Run in their env (CPU is enough):

    uv run --extra cu128 --group libero --python 3.10 python \
        /workspace/dream-robot/experiments/cosmos_policy/scripts/to_pt.py \
        /root/out/cosmos_policy/cosmos_policy_t5/b1/checkpoints/iter_000002000
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

LIBERO_PT = "/root/hf/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/snapshots/*/Cosmos-Policy-LIBERO-Predict2-2B.pt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ckpt", type=Path, help=".../checkpoints/iter_N (has a model/ folder)")
    args = p.parse_args()
    out = args.ckpt.with_suffix(".pt")

    if not out.exists():
        dcp_to_torch_save(str(args.ckpt / "model"), str(out))
    ours = torch.load(out, map_location="cpu", weights_only=False)
    ref = torch.load(glob.glob(LIBERO_PT)[0], map_location="cpu", weights_only=False)
    ref = ref.get("model", ref.get("state_dict", ref))

    net = [k for k in ref if k.startswith("net.") and not k.endswith("._extra_state")]
    missing = [k for k in net if k not in ours]
    shape = [k for k in net if k in ours and ours[k].shape != ref[k].shape]
    changed = sum(not torch.equal(ours[k].float(), ref[k].float()) for k in net if k in ours)
    print(f"{out}: {len(ours)} tensors; LIBERO net tensors {len(net)}, missing {len(missing)}, "
          f"shape mismatch {len(shape)}, changed by fine-tuning {changed}")
    assert not missing and not shape, (missing[:5], shape[:5])
    assert changed, "identical to the LIBERO policy: nothing was trained?"


if __name__ == "__main__":
    main()
