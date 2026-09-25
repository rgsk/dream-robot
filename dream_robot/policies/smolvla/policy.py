"""SmolVLA, wrapped as something ``core.eval`` can roll out.

Satisfies ``core.record.Policy``. The model is LeRobot's; this file is only the
adapter between our ``Observation`` and LeRobot's batch dict. The other
policies in this repo were built from the spec; this one is not, on purpose:
the VLA rung starts from a pretrained checkpoint, and the point is to run it
before opening it up (see experiments/t5/notes.md).

**It is built the way ``lerobot-train`` builds it, at step 0.** ``make_policy``
with our dataset's metadata sets the feature shapes and loads the pretrained
weights; the processors come from the checkpoint with our dataset's
normalisation stats swapped in. So a zero-shot run of the base checkpoint is
exactly the fine-tune before its first gradient step, and a fine-tuned
checkpoint loads through the same path.

**The instruction comes from the env, once per episode.** ``reset()`` reads
``env.prompt`` -- the rollout resets the env before the policy, so T5's draw
already exists. Every call sends it as ``task``, the key LeRobot's datasets
store language under.

**Shapes.** SmolVLA pads state and action to 32 dims (``max_state_dim`` /
``max_action_dim``), so the Panda's 8 need no new layers; the postprocessor
hands back the first 8. Images go in as float CHW in [0, 1], as a LeRobot
dataset yields them; the model resizes them to 512x512 with padding itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dream_robot.core.dataset import frame_to_float_chw
from dream_robot.core.schema import Observation

BASE_CHECKPOINT = "lerobot/smolvla_base"


def _config(checkpoint: str | Path):
    """A checkpoint's ``config.json`` as a ``SmolVLAConfig``, without draccus.

    LeRobot's own ``PreTrainedConfig.from_pretrained`` parses through draccus,
    which registers each field with argparse -- and Python 3.14's argparse
    rejects draccus's type objects (``typing.Dict[...] | None is not
    callable``). No draccus release fixes it (0.10.0 is pinned; 0.11.6 fails
    the same way). The file is plain JSON, so it is read here, and every LeRobot
    loader below is handed the finished config, which skips draccus entirely.
    """
    import dataclasses
    import json

    from huggingface_hub import hf_hub_download
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    local = Path(checkpoint)
    path = local / "config.json" if local.is_dir() else Path(
        hf_hub_download(str(checkpoint), "config.json")
    )
    raw = json.loads(path.read_text())
    if raw.pop("type", "smolvla") != "smolvla":
        raise ValueError(f"{checkpoint} is not a SmolVLA checkpoint")

    def features(d):
        return {k: PolicyFeature(FeatureType(v["type"]), tuple(v["shape"])) for k, v in d.items()}

    fields = {f.name: f for f in dataclasses.fields(SmolVLAConfig) if f.init}
    kwargs = {}
    for key, value in raw.items():
        if key not in fields:
            continue                              # keys a newer LeRobot wrote; ignored
        if key in ("input_features", "output_features"):
            value = features(value or {})
        elif key == "normalization_mapping":
            value = {k: NormalizationMode(v) for k, v in value.items()}
        elif isinstance(value, list):
            value = tuple(value)                  # JSON has no tuples; the dataclass does
        kwargs[key] = value
    return SmolVLAConfig(**kwargs)


def build(checkpoint: str | Path, *, meta=None, device: torch.device):
    """(LeRobot policy, preprocessor, postprocessor). Training and eval both come here.

    Two cases, told apart by ``meta`` (a ``LeRobotDatasetMetadata``):

    - With it: as ``lerobot-train`` builds at step 0 -- the checkpoint's
      weights, the dataset's shapes and stats. Zero-shot, and the start of a
      fine-tune.
    - Without it: a directory ``save`` wrote, whose config and processors
      already carry our features and stats. A fine-tuned checkpoint.
    """
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as _LeRobotSmolVLA

    cfg = _config(checkpoint)
    cfg.device = device.type
    if meta is None:
        policy = _LeRobotSmolVLA.from_pretrained(str(checkpoint), config=cfg).to(device)
        pre, post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device.type}},
        )
        return policy, pre, post

    cfg.pretrained_path = str(checkpoint)
    # make_policy keeps a config's input features when it has any, and the
    # base checkpoint's are SO-100's (6-dim state, camera1..3). Cleared, they
    # come from our dataset: 8-dim state, top, wrist. Output features are
    # always taken from the dataset.
    cfg.input_features = {}
    policy = make_policy(cfg=cfg, ds_meta=meta)

    # lerobot_train.py's overrides, verbatim in intent: our stats and our
    # features in the normaliser, our device.
    features = {**policy.config.input_features, **policy.config.output_features}
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )
    return policy, pre, post


def save(policy, pre, post, directory: Path) -> Path:
    """Write a checkpoint ``build`` can load without a dataset.

    The layout is lerobot-train's ``pretrained_model/``: config.json,
    model.safetensors, and the two processor pipelines (which carry the
    normalisation stats). Only config.json is written by hand, because
    LeRobot's own writer goes through draccus (see ``_config``).
    """
    import dataclasses
    import json
    from enum import Enum

    from safetensors.torch import save_model

    directory.mkdir(parents=True, exist_ok=True)

    def plain(value):
        if isinstance(value, Enum):
            return value.value
        if dataclasses.is_dataclass(value):
            return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    cfg = plain(policy.config)
    cfg["pretrained_path"] = None      # the weights are these, not the base's
    (directory / "config.json").write_text(json.dumps({"type": "smolvla", **cfg}, indent=2))
    save_model(policy, str(directory / "model.safetensors"))
    pre.save_pretrained(directory)
    post.save_pretrained(directory)
    return directory


class SmolVLAPolicy:
    """Load once, call every control step, reset every episode."""

    def __init__(self, policy, preprocessor, postprocessor, *, env, device: torch.device):
        self._policy = policy.eval()
        self._pre = preprocessor
        self._post = postprocessor
        self._env = env
        self._device = device
        self._prompt: str | None = None
        self.cameras = tuple(
            key.rsplit(".", 1)[-1] for key in policy.config.image_features
        )

    @classmethod
    def load(
        cls,
        env,
        *,
        checkpoint: str | Path = BASE_CHECKPOINT,
        dataset_root: Path | None = None,
        repo_id: str | None = None,
        device: torch.device | str | None = None,
    ) -> SmolVLAPolicy:
        """Zero-shot with ``dataset_root``; a fine-tuned checkpoint without. See ``build``."""
        meta = None
        if dataset_root is not None:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

            if repo_id is None:
                raise ValueError("building from a dataset needs its repo_id too")
            meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
        resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        policy, pre, post = build(checkpoint, meta=meta, device=resolved)
        return cls(policy, pre, post, env=env, device=resolved)

    @property
    def prompt(self) -> str | None:
        return self._prompt

    def reset(self) -> None:
        self._policy.reset()                      # drops the queued action chunk
        self._prompt = self._env.prompt

    @torch.no_grad()
    def __call__(self, observation: Observation) -> np.ndarray:
        if self._prompt is None:
            raise RuntimeError("call reset() after env.reset() before stepping")
        batch = {
            "observation.state": torch.from_numpy(np.asarray(observation.state, np.float32)),
            "task": self._prompt,
        }
        for name in self.cameras:
            if name not in observation.images:
                raise ValueError(
                    f"policy expects camera {name!r}; the env gives {sorted(observation.images)}"
                )
            batch[f"observation.images.{name}"] = torch.from_numpy(
                frame_to_float_chw(observation.images[name])
            )
        action = self._post(self._policy.select_action(self._pre(batch)))
        action = action.squeeze(0).cpu().numpy().astype(np.float32)
        grip = self._env.embodiment.gripper_slice
        action[grip] = np.clip(action[grip], 0.0, 1.0)    # 0 closed .. 1 open
        return action
