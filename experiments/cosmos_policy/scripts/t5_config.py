"""Cosmos Policy experiment config for T5: their LIBERO config, our data, from their LIBERO policy.

pod_setup.sh copies this file into <cosmos-policy>/cosmos_policy/config/experiment/; their config
loader imports every module there, so the experiment below registers itself. Train with:

    uv run --extra cu128 --group libero --python 3.10 \
        torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
        --config=cosmos_policy/config/config.py -- experiment=cosmos_policy_t5_from_libero

What changes from cosmos_predict2_2b_480p_libero, and why:
- data: our 6 HDF5 files (T5_DATA_DIR). Their dataset reads the action / proprio width from the
  arrays, so our 8-D joints + gripper flow through untouched; the model tiles any chunk x dim into
  its action latent. Stats (min / max per dim -> [-1, 1]) are computed on first load and written
  next to the data as dataset_statistics.json -- the eval needs that file.
- chunk 32 (execute 16 at eval): RoboCasa's; 1.07 s at 30 Hz.
- no rollout_data_dir: demos only (value head learns "every demo succeeds" returns).
- start from the LIBERO policy, not the base video model. Their DCP checkpointer silently skips a
  `.pt` load_path ("Training from scratch"), so PtInitCheckpointer below loads it explicitly,
  the same way their inference loader does, and fails loudly if weights are missing.
- one GPU: fsdp_shard_size 1 (theirs 8), batch / schedule for a ~2-3 h run, no wandb (action L1
  printed to the log instead).
"""

from __future__ import annotations

import os

import torch
from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.callback import Callback
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_policy._src.predict2.checkpointer.dcp import DistributedCheckpointer, ModelWrapper
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE

T5_DATA_DIR = os.environ.get("T5_DATA_DIR", "/root/data/cosmos_policy/t5_224")
LIBERO_POLICY = "hf://nvidia/Cosmos-Policy-LIBERO-Predict2-2B/Cosmos-Policy-LIBERO-Predict2-2B.pt"
CHUNK = 32


class PtInitCheckpointer(DistributedCheckpointer):
    """Their DCP checkpointer, plus: a fresh run with a `.pt` load_path starts from those weights.

    Resuming (a latest_checkpoint.txt in this job's dir) still goes through DCP as usual.
    """

    def load(self, model, optimizer=None, scheduler=None, grad_scaler=None) -> int:
        pt = str(self.load_path or "")
        if self._read_latest_checkpoint_file() is not None or not pt.endswith(".pt"):
            return super().load(model, optimizer, scheduler, grad_scaler)
        log.critical(f"Initialising model weights from {pt}")
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        ckpt = ckpt.get("model", ckpt.get("state_dict", ckpt))
        wrapper = ModelWrapper(model)
        state = wrapper.state_dict()
        missing = [k for k in state if k not in ckpt]
        unused = [k for k in ckpt if k not in state]
        for k in state:
            if k in ckpt:
                state[k] = ckpt[k]
        log.critical(f"loaded {len(state) - len(missing)} tensors, missing {len(missing)}, unused {len(unused)}")
        assert not [k for k in missing if k.startswith("net.")], f"network weights missing: {missing[:5]}"
        wrapper.load_state_dict(state)
        torch.cuda.empty_cache()
        return 0


class ActionL1Log(Callback):
    """Prints the demo action L1 (normalised units, their target ~0.010-0.015) and total loss."""

    def __init__(self, every_n: int = 10):
        super().__init__()
        self.every_n = every_n
        self.l1, self.loss, self.n = 0.0, 0.0, 0

    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration=0):
        l1 = output_batch.get("demo_sample_action_l1_loss")
        if l1 is not None and not torch.isnan(l1):
            self.l1 += float(l1)
            self.loss += float(loss)
            self.n += 1
        if iteration % self.every_n == 0 and self.n:
            log.info(f"[t5] iter {iteration}  action_l1 {self.l1 / self.n:.4f}  loss {self.loss / self.n:.4f}")
            self.l1, self.loss, self.n = 0.0, 0.0, 0


t5_dataset = L(LIBERODataset)(
    data_dir=T5_DATA_DIR,
    t5_text_embeddings_path=os.path.join(T5_DATA_DIR, "t5_embeddings.pkl"),
    chunk_size=CHUNK,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # Wan tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir="",
    demonstration_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
)

cosmos_policy_t5_from_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            {"override /callbacks": ["basic", "long", "cluster_speed"]},
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(t5_action_l1=L(ActionL1Log)(every_n=10)),
            logging_iter=10,
            max_iter=3000,
        ),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            # warm up 200, decay to 0.3x by 3000; the run is stopped by budget, not by max_iter
            cycle_lengths=[3000, 100000000000000],
            warm_up_steps=[200, 0],
            f_start=[1e-6, 0.3],
            f_max=[1.0, 0.3],
            f_min=[0.3, 0.3],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(fsdp_shard_size=1),
        ),
        checkpoint=dict(
            type=L(PtInitCheckpointer)(),
            load_path=get_checkpoint_path(LIBERO_POLICY),
            load_training_state=False,
            save_iter=500,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=t5_dataset,
            sampler=L(DistributedSampler)(
                dataset=t5_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,  # theirs 30 with optimizer states sharded over 8 GPUs; the smoke run sizes it
            drop_last=True,
        ),
        job=dict(group="cosmos_policy_t5", name="cosmos_policy_t5_from_libero"),
    )
)

# Inference: as their *__inference_only configs, sampling noise from 80 down to 4.
cosmos_policy_t5_from_libero__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_policy_t5_from_libero", "_self_"],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(sde=L(HybridEDMSDE)(sigma_max=80, sigma_min=4)),
        ),
        job=dict(group="cosmos_policy_t5_inference", name="cosmos_policy_t5_from_libero__inference"),
    )
)

for _item in (cosmos_policy_t5_from_libero, cosmos_policy_t5_from_libero__inference):
    ConfigStore.instance().store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
