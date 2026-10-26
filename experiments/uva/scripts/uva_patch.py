"""UVA that knows where the arm is (joint readings) and, optionally, sees the wrist camera.

Why (experiments/uva/notes.md): our actions are absolute joint targets, so the model must know the
arm's current joints; on a real robot the readings (not past commands) carry contact and the
gripper stalling on the object. Their LIBERO setup feeds no robot state and one camera; b1 (top
only) missed grasps by a little, and the wrist view is what lines a grasp up.

What this adds, without editing their code (``add_inputs``):
- joints: a Linear(8 -> 768) maps the reading at each of the 4 past frames to one vector, copied
  over that frame's 256 positions;
- wrist (``wrist=True``): the wrist frames at the same 4 steps go through their frozen VAE
  (resized to 256, like the top camera) -> 16x16 patch codes per frame -> a Linear(16 -> 768), one
  vector per position. The same per-position fusion their own code uses for a second image
  (``proprioception_image_cond``, concatenated before ``proj_cond_x_layer``; a linear of a
  concatenation is a sum of linears);
- both are added right after their per-position projection (``proj_cond_x_layer``), and both
  layers start at zero -> at step 0 the model is exactly the LIBERO-10 one.
- imagine_wrist (``imagine_wrist=True``, needs ``wrist``): a second video head, a copy of their
  pretrained top-camera one (``diffloss``), learns the future wrist frames' VAE latents from the same
  decoder output ``z``, in the modes that predict video, with the same mask -- as their own
  ``predict_wrist_img`` head (``diffloss_wrist``), without the extra wrist input slots that would
  resize the pretrained ``proj_cond_x_layer``. Targets: the wrist frames at their 4 future steps
  (19, 23, 27, 31 of the 32-step window).
Inputs arrive in the obs dict -- ``joint_qpos`` (raw, scaled here by the normalizer's entry) and
``eye_in_hand_rgb`` ([0, 1] frames, upright) -- 32 steps in training, 16 at run time, and are taken
at the same steps as their 4 past frames (3, 7, 11, 15).

``scale_actions``: their loader leaves actions unscaled when ``abs_action: false`` ("already
normalized"); ours are radians in about [-2.4, 2.9], so map each dimension to [-1, 1] like their
pose actions.
"""

from __future__ import annotations

import copy
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from unified_video_action.common.normalize_util import array_to_stats, get_range_normalizer_from_stat
from unified_video_action.utils.data_utils import extract_latent_autoregressive

PAST_FRAME_STEPS = [3, 7, 11, 15]  # their select_frames: the 4 condition frames, train and eval
FUTURE_FRAME_STEPS = [19, 23, 27, 31]  # and the 4 frames it predicts, in training
VIDEO_MODES = ("video_model", "dynamic_model", "full_dynamic_model")  # their modes with a video loss
WRIST_KEY = "eye_in_hand_rgb"


class AddInputs(nn.Module):
    """Their per-position projection, plus the extra inputs' vectors set for this call."""

    def __init__(self, proj: nn.Linear):
        super().__init__()
        self.proj = proj
        self.pending: torch.Tensor | None = None  # [B, 4 * 256, 768]

    def forward(self, x):
        out = self.proj(x)
        if self.pending is not None:
            out = out + self.pending.to(out.dtype)
        return out


def _zero_linear(n_in: int, n_out: int) -> nn.Linear:
    layer = nn.Linear(n_in, n_out)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def wrist_tokens(policy, frames) -> torch.Tensor:
    """Wrist frames [B, 4, 3, H, W] in [0, 1] -> their VAE's patch codes [B, 4 * 256, 16], laid out as
    their own frame tokens (resized as resize_image, scaled as process_data)."""
    b = frames.shape[0]
    frames = F.interpolate(rearrange(frames.float(), "b t c h w -> (b t) c h w"), size=(256, 256),
                           mode="bilinear", align_corners=False)
    frames = rearrange(frames * 2 - 1, "(b t) c h w -> b c t h w", b=b)
    latents, _ = extract_latent_autoregressive(policy.vae_model, frames)  # [B, 4, 16, 16, 16]
    tokens = policy.model.patchify(rearrange(latents, "b t c h w -> (b t) c h w"))
    return rearrange(tokens, "(b t) s c -> b (t s) c", b=b)


def add_inputs(policy, *, wrist: bool = False, imagine_wrist: bool = False, joint_dim: int = 8) -> None:
    """Convert a UnifiedVideoActionPolicy in place (after loading pretrained weights)."""
    assert wrist or not imagine_wrist, "imagining the wrist needs the wrist camera"
    mar = policy.model
    width = mar.proj_cond_x_layer.out_features
    mar.joint_proj = _zero_linear(joint_dim, width)
    if wrist:
        mar.wrist_proj = _zero_linear(mar.token_embed_dim, width)
    mar.proj_cond_x_layer = AddInputs(mar.proj_cond_x_layer)
    if imagine_wrist:
        mar.diffloss_wrist = copy.deepcopy(mar.diffloss)  # starts as their pretrained video head
        mar.wrist_target = None  # [B, 4 * 256, 16], set per training batch
        forward_loss = mar.forward_loss

        def forward_loss_with_wrist(*args, **kwargs):
            loss, video_loss, act_loss = forward_loss(*args, **kwargs)
            if kwargs.get("task_mode") in VIDEO_MODES and mar.wrist_target is not None:
                wrist_loss = mar.diffloss_wrist(z=kwargs["z"], target=mar.wrist_target, mask=kwargs["mask"],
                                                text_latents=kwargs.get("text_latents"))
                loss, video_loss = loss + wrist_loss, video_loss + wrist_loss
            return loss, video_loss, act_loss

        mar.forward_loss = forward_loss_with_wrist

    def set_inputs(self, obs):
        """Pop joint_qpos [B, T, 8] (+ wrist frames [B, T, 3, H, W]) -> pending [B, 4 * 256, 768]."""
        joints = self.normalizer["joint_qpos"].normalize(obs.pop("joint_qpos"))[:, PAST_FRAME_STEPS]
        pending = repeat(self.model.joint_proj(joints.float()), "b t c -> b (t s) c", s=self.model.seq_len)
        if wrist:
            tokens = wrist_tokens(self, obs.pop(WRIST_KEY)[:, PAST_FRAME_STEPS])
            pending = pending + self.model.wrist_proj(tokens.float())
        self.model.proj_cond_x_layer.pending = pending

    compute_loss, predict_action = policy.compute_loss, policy.predict_action

    def compute_loss_with_inputs(self, batch, **kwargs):
        if imagine_wrist:  # the future wrist frames, before set_inputs takes the window
            mar.wrist_target = wrist_tokens(self, batch["obs"][WRIST_KEY][:, FUTURE_FRAME_STEPS])
        self.set_inputs(batch["obs"])
        try:
            return compute_loss(batch, **kwargs)
        finally:
            if imagine_wrist:
                mar.wrist_target = None

    def predict_action_with_inputs(self, obs_dict, language_goal=None):
        self.set_inputs(obs_dict)
        return predict_action(obs_dict, language_goal=language_goal)

    policy.set_inputs = types.MethodType(set_inputs, policy)
    policy.compute_loss = types.MethodType(compute_loss_with_inputs, policy)
    policy.predict_action = types.MethodType(predict_action_with_inputs, policy)


def decode_into_ram(dataset) -> None:
    """Their replay buffer keeps frames JPEG2000-compressed in Zarr and decodes 32 per sample;
    on the pod that kept the GPU at 0% (loader workers at 100% CPU). Decode every key once into
    plain arrays (pick_place_cube: ~260 MB) and let the sampler slice those."""
    dataset.sampler.replay_buffer = {k: np.asarray(dataset.replay_buffer[k][:]) for k in dataset.sampler.keys}


def skip_threadpool_limits() -> None:
    """Their __getitem__ calls threadpool_limits(1) per sample, which stats ~1,750 library paths;
    on the pod's /workspace that is 1.8 s of a 1.8 s sample (profile_loader.py). Make it a no-op;
    loader workers cap their threads once instead (``one_thread``)."""
    import unified_video_action.dataset.libero_replay_image_dataset as libero_dataset
    libero_dataset.threadpool_limits = lambda *args, **kwargs: None


def one_thread(worker_id: int) -> None:
    """DataLoader worker_init_fn: one thread per worker, as their per-sample limit intended."""
    torch.set_num_threads(1)


def scale_actions(normalizer, actions) -> None:
    """Replace their identity action normalizer with a per-dimension [-1, 1] range one."""
    normalizer["action"] = get_range_normalizer_from_stat(array_to_stats(actions))
