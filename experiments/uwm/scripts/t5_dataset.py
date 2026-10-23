"""T5 demos for UWM: their LIBERO loader, reading our files, plus the instruction as CLIP tokens.

Their RobomimicDataset (envs/uwm/src/datasets/robomimic) reads raw RGB from LIBERO HDF5 into a
Zarr buffer and yields no instruction -- their LIBERO-90 model never saw one. T5 is all about the
instruction, so this subclass:
- reads data/cosmos_policy/t5_224 (the same files Cosmos trained on): JPEG frames decoded once
  into the buffer, 8-D joint + gripper actions, upright frames (no flip);
- remembers each episode's instruction (from the file name, as LIBERO and Cosmos do), saved next
  to the buffer so a restored buffer still knows them;
- adds ``input_ids`` / ``attention_mask`` (CLIP tokenizer, padded to 16; our prompts are 10-14
  tokens) to every sample, which their ``process_batch`` already passes to the model;
- adds ``obs.robot_states`` (our 7 joints + gripper, min/max-normalised to [-1, 1]): our actions
  are absolute joint targets, so the model must know where the arm is (their LIBERO actions are
  relative EEF moves, and needed no state). Read straight from the HDF5, not the buffer, so a
  buffer built without them still serves.
A sample's episode is found from where its window starts: their sampler never pads, so every
window lies inside one episode.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

from datasets.robomimic.dataset import RobomimicDataset
from datasets.utils.buffer import CompressedTrajectoryBuffer
from datasets.utils.file_utils import glob_all

TOKENS = 16
CAMERAS = {"agentview_rgb": "agentview_rgb_jpeg", "eye_in_hand_rgb": "eye_in_hand_rgb_jpeg"}


def instruction(path: str) -> str:
    return Path(path).name[: -len("_demo.hdf5")].replace("_", " ")


def decode(jpegs) -> np.ndarray:
    return np.stack([np.array(Image.open(io.BytesIO(j.tobytes())).convert("RGB")) for j in jpegs])


class T5Dataset(RobomimicDataset):
    def _init_buffer(self, hdf5_path_globs, buffer_path):
        paths = sorted(glob_all(hdf5_path_globs))
        commands_path = Path(f"{buffer_path}.commands.json")
        metadata = {f"obs.{k}": {"shape": s, "dtype": np.uint8} for k, s in self._image_shapes.items()}
        metadata["action"] = {"shape": self._action_shape, "dtype": np.float32}

        episodes = []
        for path in paths:
            with h5py.File(path) as f:
                episodes += [(path, f"demo_{i}", f[f"data/demo_{i}/actions"].shape[0])
                             for i in range(len(f["data"]))]
        buffer = CompressedTrajectoryBuffer(
            storage_path=buffer_path, metadata=metadata, capacity=sum(n for *_, n in episodes)
        )
        self.states = np.concatenate([h5py.File(path)[f"data/{demo}/robot_states"][:]
                                      for path, demo, _ in episodes]).astype(np.float32)
        low, high = self.states.min(0), self.states.max(0)
        self.state_scale, self.state_offset = (high - low) / 2, (high + low) / 2
        self.states = (self.states - self.state_offset) / self.state_scale
        if buffer.restored:
            self.commands = json.loads(commands_path.read_text())
            assert len(self.commands) == buffer.num_episodes, "buffer and commands disagree"
            return buffer

        for path, demo, _ in tqdm(episodes, desc="T5 episodes -> buffer"):
            with h5py.File(path) as f:
                group = f[f"data/{demo}"]
                episode = {f"obs.{k}": decode(group[f"obs/{CAMERAS[k]}"]) for k in self._image_shapes}
                episode["action"] = group["actions"][:].astype(np.float32)
            buffer.add_episode(episode)
        self.commands = [instruction(path) for path, *_ in episodes]
        commands_path.write_text(json.dumps(self.commands, indent=0))
        return buffer

    def __getitem__(self, idx: int) -> dict:
        data = super().__getitem__(idx)
        start = self.sampler.indices[idx][0]
        episode = int(np.searchsorted(self.buffer.episode_ends, start, side="right"))
        ids, mask = self.tokens[self.commands[episode]]
        data["input_ids"], data["attention_mask"] = ids, mask
        data["obs"]["robot_states"] = torch.from_numpy(self.states[start:self.sampler.indices[idx][1]])
        return data

    def tokenize(self) -> None:
        tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.tokens = {}
        for c in set(self.commands):
            t = tok(c, padding="max_length", max_length=TOKENS, return_tensors="pt")
            self.tokens[c] = (t["input_ids"][0], t["attention_mask"][0])


def make_t5_dataset(name, hdf5_path_globs, buffer_path, shape_meta, seq_len, val_ratio=0.0,
                    subsample_ratio=1.0, flip_rgb=False):
    """Their make_robomimic_dataset, for T5Dataset: build the buffer once, then train / val sets."""
    kwargs = dict(name=name, hdf5_path_globs=hdf5_path_globs, buffer_path=buffer_path,
                  shape_meta=shape_meta, seq_len=seq_len, flip_rgb=flip_rgb)
    if not os.path.exists(buffer_path) and (not dist.is_initialized() or dist.get_rank() == 0):
        T5Dataset(**kwargs)
    if dist.is_initialized():
        dist.barrier()
    train_set = T5Dataset(**kwargs, val_ratio=val_ratio, subsample_ratio=subsample_ratio)
    train_set.tokenize()
    return train_set, train_set.get_validation_dataset()
