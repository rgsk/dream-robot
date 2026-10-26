"""The LIBERO-10 checkpoint's own config, pointed at one of our tasks.

Starts from the cfg saved in the checkpoint (so model sizes, the action head type ``conv_ori``,
the 16-step history / 32-step window all match the weights) and changes only what our data needs:
- task name keeps ``libero``: their resize_image / predict_action only handle the top camera for
  task names containing it;
- 8-D actions, top camera (+ wrist with ``wrist=True``), ``joint_qpos`` (see uva_patch.py) and the
  instruction tokens; frames at the size they were recorded (pick_place_cube 128, T5 224 px);
- ``abs_action: false``: our actions are joint targets, not poses to convert;
- no colour jitter (``data_aug: false``);
- the checkpoint itself as ``pretrained_model_path``: their loader copies every weight whose
  shape still fits (all but the 10-D action layers) from its EMA model;
- no rollouts during training (their LIBERO env runner), no wandb.
"""

from __future__ import annotations

from pathlib import Path

import dill
import torch
from omegaconf import DictConfig, open_dict


def shape_meta(image: int = 128, wrist: bool = False, action_dim: int = 8, joint_dim: int = 8) -> dict:
    cameras = ["agentview_rgb"] + (["eye_in_hand_rgb"] if wrist else [])  # names as in to_hdf5.py
    return {
        "image_resolution": image,
        "action": {"shape": [action_dim]},
        "obs": {
            **{camera: {"shape": [3, image, image], "type": "rgb"} for camera in cameras},
            "joint_qpos": {"shape": [joint_dim], "type": "low_dim"},
            "language": {"shape": [15]},
        },
    }


def build_cfg(ckpt: Path, data: Path, name: str, output_dir: Path, *, image: int = 128,
              wrist: bool = False, imagine_wrist: bool = False) -> DictConfig:
    cfg = torch.load(open(ckpt, "rb"), pickle_module=dill, map_location="cpu")["cfg"]
    meta = shape_meta(image, wrist)
    with open_dict(cfg):
        cfg.output_dir = str(output_dir)
        cfg.task.name = f"libero_{name}"
        cfg.task.imagine_wrist = imagine_wrist  # ours (uva_patch): a second video head, on the wrist
        cfg.task.shape_meta = meta
        cfg.task.dataset.shape_meta = meta
        cfg.model.policy.shape_meta = meta
        cfg.task.dataset.dataset_path = str(data)
        cfg.task.dataset.abs_action = False
        cfg.task.dataset.val_ratio = 0.08  # 2 of 25 demos
        cfg.task.dataset.data_aug = False  # their colour jitter: slow on CPU, and shifts hue (T5 = colour)
        cfg.model.policy.autoregressive_model_params.pretrained_model_path = str(ckpt)
        cfg.training.rollout_every = 10**9
        cfg.training.resume = False
        cfg.logging.mode = "disabled"
    return cfg
