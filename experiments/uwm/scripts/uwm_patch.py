"""UWM that can see where things are: image patches + instruction words as cross-attention context.

Why (experiments/uwm/notes.md step 8): as built, each camera image reaches the transformer as ONE
pooled vector and the instruction as ONE pooled CLIP vector, both only through AdaLN. On T5 the
model learnt to ignore the colour word (swap probe ~0) and grab the cube in slot 1 -- "yellow" has
nothing spatial to attach to.

What this adds, from parts already in their code (none used by their configs):
- every block becomes their AdaLNHybridAttentionBlock: the same self-attention / MLP / AdaLN
  (pretrained weights load 1:1) plus a cross-attention to context tokens;
- context = the ResNet-18's 7x7 feature map before pooling (49 tokens per image, 2 cameras x 2
  frames = 196) + CLIP's per-word hidden states (16 tokens), each projected to 768, with learned
  position embeddings, layer-normed;
- each cross-attention's output projection starts at zero -> at step 0 the model is exactly the
  pretrained one; the patch path is learnt from there.
The pooled image / text vectors (and joints, if given) still feed AdaLN as before. Video-drop
works here too.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
from einops import rearrange

from models.common.adaln_attention import AdaLNHybridAttentionBlock

PATCHES_PER_IMAGE = 49  # ResNet-18 at 224 px: 7 x 7


def _resnet_map(resnet, x):
    """torchvision ResNet up to layer4 (before pooling): (N, 512, 7, 7)."""
    x = resnet.maxpool(resnet.relu(resnet.bn1(resnet.conv1(x))))
    return resnet.layer4(resnet.layer3(resnet.layer2(resnet.layer1(x))))


def add_patch_context(model) -> None:
    """Convert a UnifiedWorldModel in place (before loading pretrained weights)."""
    enc, net = model.obs_encoder, model.noise_pred_net
    dim = net.pos_embed.shape[-1]
    images = enc.num_views * enc.num_frames

    # hybrid blocks: pretrained self-attention / MLP / AdaLN keys keep their names
    old = net.blocks[0]
    net.blocks = nn.ModuleList([
        AdaLNHybridAttentionBlock(dim=dim, cond_dim=old.adaLN_modulation[-1].in_features,
                                  num_heads=old.attn.num_heads, mlp_ratio=4.0, qkv_bias=True)
        for _ in net.blocks
    ])
    for block in net.blocks:
        nn.init.zeros_(block.xattn.proj.weight)
        nn.init.zeros_(block.xattn.proj.bias)
        nn.init.zeros_(block.adaLN_modulation[-1].weight)  # their init for AdaLN; loaded over
        nn.init.zeros_(block.adaLN_modulation[-1].bias)

    net.patch_proj = nn.Linear(512, dim)
    net.word_proj = nn.Linear(512, dim)
    net.ctx_pos = nn.Parameter(torch.randn(1, images * PATCHES_PER_IMAGE + 16, dim) * 0.02)
    net.ctx_norm = nn.LayerNorm(dim)
    net.context = None

    def encode(curr_imgs, curr_obs_dict):
        """Their encode_curr_obs, plus the context tokens (stashed on the net for its forward)."""
        B = curr_imgs.shape[0]
        flat = enc.img_encoder.norm(rearrange(curr_imgs, "b v c t h w -> (b v t) c h w"))
        resnet = enc.img_encoder.model
        fmap = _resnet_map(resnet, flat)                                   # (BVT, 512, 7, 7)
        pooled = resnet.fc(torch.flatten(resnet.avgpool(fmap), 1))         # = resnet(flat)
        feats = rearrange(pooled, "(b n) c -> b (n c)", b=B)
        patches = rearrange(fmap, "(b n) c h w -> b (n h w) c", b=B)

        text = enc.text_encoder.language_model.text_model(
            input_ids=curr_obs_dict["input_ids"], attention_mask=curr_obs_dict["attention_mask"])
        words = text.last_hidden_state                                     # (B, 16, 512)
        pooled_text = enc.text_encoder.head(enc.text_encoder.language_model.text_projection(
            text.pooler_output))                                           # = get_text_features
        net.context = net.ctx_norm(
            torch.cat([net.patch_proj(patches), net.word_proj(words)], dim=1) + net.ctx_pos)
        low_dims = [curr_obs_dict[k].flatten(1) for k in enc.low_dim_keys if enc.use_low_dim]
        return torch.cat([feats, *low_dims, pooled_text], dim=-1)          # their order

    def encode_curr_obs(self, curr_obs_dict):
        return encode(self.apply_transform(curr_obs_dict), curr_obs_dict)

    def encode_curr_and_next_obs(self, curr_obs_dict, next_obs_dict):
        curr_imgs, next_imgs = self.apply_transform([curr_obs_dict, next_obs_dict])
        return encode(curr_imgs, curr_obs_dict), self.apply_vae(next_imgs)

    enc.encode_curr_obs = types.MethodType(encode_curr_obs, enc)
    enc.encode_curr_and_next_obs = types.MethodType(encode_curr_and_next_obs, enc)
    net.forward = types.MethodType(forward_with_context, net)
    net.drop_video = False


def forward_with_context(self, global_cond, action, action_t, next_obs, next_obs_t):
    """DualNoisePredictionNet.forward with hybrid blocks; video tokens left out if drop_video."""
    if action_t.dim() == 0:
        action_t = action_t.expand(action.shape[0]).to(dtype=torch.long, device=action.device)
    if next_obs_t.dim() == 0:
        next_obs_t = next_obs_t.expand(action.shape[0]).to(dtype=torch.long, device=action.device)
    a0, a1 = self.action_inds
    o0, o1 = self.next_obs_inds
    registers = self.registers.expand(action.shape[0], -1, -1)
    if self.drop_video:
        x = torch.cat((self.action_encoder(action), registers), dim=1)
        x = x + torch.cat((self.pos_embed[:, a0:a1], self.pos_embed[:, o1:]), dim=1)
    else:
        x = torch.cat((self.action_encoder(action), self.obs_patchifier(next_obs), registers), dim=1)
        x = x + self.pos_embed
    cond = torch.cat((global_cond, self.timestep_embedding(action_t, next_obs_t)), dim=-1)
    for block in self.blocks:
        x = block(x, self.context, cond)
    x = self.head(x, cond)
    actions = self.action_decoder(x[:, a0:a1])
    if self.drop_video:
        return actions, None
    return actions, self.obs_patchifier.unpatchify(x[:, o0:o1])
