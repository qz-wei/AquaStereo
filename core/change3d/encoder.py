import importlib.util
import os
from dinov2.models.vision_transformer import vit_base, vit_small


from functools import partial
import math
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint

from core.change3d.x3d import create_x3d
from torch.nn import functional as F
import torch.distributed as dist
import numpy as np


def _normalize_optional_path(path):
    if path is None:
        return None
    path = str(path)
    if path.lower() in ["none", "null", ""]:
        return None
    return path


def resolve_dino_pretrained_path(args):
    """
    Resolve DINOv2 pretrained path according to args.vit_size.

    Usage:
      --pretrained_dino ./pretrained
          vitb -> ./pretrained/dinov2_vitb14_pretrain.pth
          vits -> ./pretrained/dinov2_vits14_pretrain.pth

      --pretrained_dino ./pretrained/dinov2_vitb14_pretrain.pth
          use this file directly
    """
    dino_arg = _normalize_optional_path(getattr(args, "pretrained_dino", "./pretrained"))
    vit_size = str(getattr(args, "vit_size", "vitb") or "vitb").lower()

    if dino_arg is None:
        dino_arg = "./pretrained"

    if os.path.isdir(dino_arg):
        if vit_size == "vitb":
            return os.path.join(dino_arg, "dinov2_vitb14_pretrain.pth")
        elif vit_size == "vits":
            return os.path.join(dino_arg, "dinov2_vits14_pretrain.pth")
        else:
            raise ValueError(f"Unsupported vit_size={vit_size!r}. Expected 'vitb' or 'vits'.")

    return dino_arg


def _extract_state_dict(ckpt):
    """Extract a state_dict from common checkpoint formats."""
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("model", "model_state", "state_dict", "teacher", "student"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _clean_common_prefixes(state_dict):
    """Remove wrappers that often appear in saved checkpoints."""
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    changed = True
        cleaned[nk] = v
    return cleaned


def get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
  assert max_H%divider==0
  assert max_W%divider==0

  def round_by_divider(x):
    return int(np.ceil(x/divider)*divider)

  H_resize = round_by_divider(H)   
  W_resize = round_by_divider(W)
  if H_resize>max_H or W_resize>max_W:
    if H_resize>W_resize:
      W_resize = round_by_divider(W_resize*max_H/H_resize)
      H_resize = max_H
    else:
      H_resize = round_by_divider(H_resize*max_W/W_resize)
      W_resize = max_W
  return int(H_resize), int(W_resize)


class Encoder(nn.Module):


    def __init__(self, args: Any, embed_dims: List[int]) -> None:
        super().__init__()
        self.args = args
        self.num_perception_frame = int(args.num_perception_frame)


        self.x3d = create_x3d(input_clip_length=self.num_perception_frame + 2, depth_factor=5.0)
        self._load_x3d_pretrained()


        init_h = int(getattr(args, "in_height", getattr(args, "height", 512)))
        init_w = int(getattr(args, "in_width", getattr(args, "width", 512)))
        self.perception_frames = nn.Parameter(
            torch.randn(1, 3, self.num_perception_frame, init_h, init_w) * 0.02,
            requires_grad=True,
        )

        # DINO is always enabled.
        self.use_dino = True

        restore_ckpt = _normalize_optional_path(getattr(args, "restore_ckpt", None))
        dino_path = resolve_dino_pretrained_path(args)

        self.vit_size = str(getattr(args, "vit_size", "vitb") or "vitb").lower()

        vit_kwargs = dict(
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
        )

        if self.vit_size == "vits":
            dino_model = vit_small(**vit_kwargs).eval()
            dino_dim = 384
            dino_name = "DINOv2 ViT-S/14"
        elif self.vit_size == "vitb":
            dino_model = vit_base(**vit_kwargs).eval()
            dino_dim = 768
            dino_name = "DINOv2 ViT-B/14"
        else:
            raise ValueError(f"Unsupported vit_size={self.vit_size!r}. Expected 'vits' or 'vitb'.")

        print(f"[Encoder] Build {dino_name}, feature_dim={dino_dim}")

        if restore_ckpt is not None:
            print(f"[Encoder] restore_ckpt is set; skip external {dino_name} loading. DINO will be loaded from restore_ckpt.")
        else:
            if not os.path.isfile(dino_path):
                raise FileNotFoundError(f"[Encoder] {dino_name} pretrained not found: {dino_path}")

            print(f"[Encoder] Load {dino_name} pretrained: {dino_path}")
            dino_weights = torch.load(dino_path, map_location="cpu")
            msg = dino_model.load_state_dict(dino_weights, strict=False)
            print(f"[Encoder] Loaded {dino_name}: {msg}")

        for p in dino_model.parameters():
            p.requires_grad_(False)

        self.dvt_vitb14 = nn.ModuleList([dino_model])
        self.dino_proj = nn.Sequential(nn.Conv2d(dino_dim, 24, 1))

        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(),
            ) for dim in embed_dims
        ])

    def _load_x3d_pretrained(self) -> None:
        restore_ckpt = _normalize_optional_path(getattr(self.args, "restore_ckpt", None))
        pretrained = _normalize_optional_path(getattr(self.args, "pretrained_change3d", None))

        # restore_ckpt has the highest priority. When it is provided, the whole
        # model state, including encoder.x3d, should come from that checkpoint.
        if restore_ckpt is not None:
            print("[Encoder] restore_ckpt is set; skip external X3D loading. X3D will be loaded from restore_ckpt.")
            return

        if pretrained is None:
            raise ValueError(
                "X3D pretrained path is empty and --restore_ckpt is not provided. "
                "Pass --pretrained_change3d for training from external pretrain."
            )

        if not os.path.isfile(pretrained):
            raise FileNotFoundError(f"[Encoder] X3D pretrained not found: {pretrained}")

        ckpt = torch.load(pretrained, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model_state" in ckpt:
                state_dict = ckpt["model_state"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        cleaned = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("x3d."):
                nk = nk[len("x3d."):]
            if nk.startswith("encoder.x3d."):
                nk = nk[len("encoder.x3d."):]
            cleaned[nk] = v

        try:
            msg = self.x3d.load_state_dict(cleaned, strict=True)
            print(f"[Encoder] Loaded X3D pretrained: {pretrained}, {msg}")
        except RuntimeError as e:
            msg = self.x3d.load_state_dict(cleaned, strict=False)
            print(f"[Encoder] Loaded X3D pretrained with strict=False: {pretrained}, {msg}")
            print(f"[Encoder] strict=True failed with: {e}")
    def _resize_perception_frames(self, x: torch.Tensor) -> torch.Tensor:

        B, _, H, W = x.shape
        p = self.perception_frames.to(device=x.device, dtype=x.dtype)

        if p.shape[-2:] != (H, W):

            _, C, T, H0, W0 = p.shape
            p_2d = p.permute(0, 2, 1, 3, 4).reshape(-1, C, H0, W0)
            p_2d = F.interpolate(
                p_2d,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

            p = p_2d.reshape(1, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()


        return p.expand(B, -1, -1, -1, -1)

    @torch.cuda.amp.autocast()
    def _process_dino_frame(self, frame: torch.Tensor, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:

        if not self.use_dino:
            raise RuntimeError("_process_dino_frame was called while use_dino is disabled.")

        with torch.no_grad():
            H, W = frame.shape[-2:]
            resize_H, resize_W = get_resize_keep_aspect_ratio(H, W, divider=14)
            dino_input = F.interpolate(
                frame,
                size=(resize_H, resize_W),
                mode="bilinear",
                align_corners=False,
            )
            dino_feat = self.dvt_vitb14[0].forward_features(dino_input)["x_norm_patchtokens"]
            B, N, C = dino_feat.shape
            dino_feat = dino_feat.permute(0, 2, 1).reshape(
                B,
                C,
                resize_H // 14,
                resize_W // 14,
            )

            if target_hw is None:
                target_hw = (H, W)
            dino_feat = F.interpolate(
                dino_feat,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        return dino_feat

    def enhance(self, x: torch.Tensor, fc: nn.Module, x0_pre_post=None) -> torch.Tensor:

        middle_idx = x.shape[2] // 2

        pre_frame = x[:, :, 0]
        post_frame = x[:, :, self.num_perception_frame + 1]

        middle_frame = x[:, :, middle_idx]
        middlepre_frame = x[:, :, 1]
        middlepost_frame = x[:, :, self.num_perception_frame]

        enhanced_x = x.clone()

        if x0_pre_post is not None:
            raw_pre_frame, raw_post_frame = x0_pre_post
            target_hw = middle_frame.shape[-2:]

            dino_feat_pre = self.dino_proj(self._process_dino_frame(raw_pre_frame, target_hw=target_hw))
            dino_feat_post = self.dino_proj(self._process_dino_frame(raw_post_frame, target_hw=target_hw))
            semantic_diff = torch.abs(dino_feat_pre - dino_feat_post)
            enhanced_middle_frame = middle_frame + semantic_diff

            if self.num_perception_frame == 3:
                enhanced_x[:, :, 1] = middlepre_frame + dino_feat_pre
                enhanced_x[:, :, self.num_perception_frame] = middlepost_frame + dino_feat_post
        else:
            temporal_diff = torch.abs(pre_frame - post_frame)
            enhancement_features = fc(temporal_diff)
            enhanced_middle_frame = middle_frame + enhancement_features

        enhanced_x[:, :, middle_idx] = enhanced_middle_frame
        return enhanced_x

    def base_forward(self, x: torch.Tensor, output_final: bool = False) -> List[torch.Tensor]:

        if output_final:
            for i in range(5):
                x = self.x3d.blocks[i](x)
            return x[:, :, self.num_perception_frame]

        out = []
        for i in range(5):
            x0_pre_post = None
            if self.use_dino and i == 0:
                pre_frame = x[:, :, 0]
                post_frame = x[:, :, self.num_perception_frame + 1]
                x0_pre_post = [pre_frame, post_frame]

            x = self.x3d.blocks[i](x)
            x = self.enhance(x, self.fc[i], x0_pre_post)

            layer_feature = []
            for idx in range(self.num_perception_frame):
                layer_feature.append(x[:, :, idx + 1])
            out.append(layer_feature)

        return out

    def forward(self, x: torch.Tensor, y: torch.Tensor, output_final: bool = False) -> List[torch.Tensor]:
        """
        Forward pass with input and target frames.

        Args:
            x: [B, C, H, W]
            y: [B, C, H, W]
        """
        if x.ndim != 4 or y.ndim != 4:
            raise ValueError(f"Encoder expects x/y as [B, C, H, W], got x={tuple(x.shape)}, y={tuple(y.shape)}")
        if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
            raise ValueError(f"x/y batch or channel mismatch: x={tuple(x.shape)}, y={tuple(y.shape)}")
        if x.shape[-2:] != y.shape[-2:]:
            raise ValueError(
                f"x/y spatial size must be the same before Encoder, got x={tuple(x.shape)}, y={tuple(y.shape)}. "
                "For stereo/change feature extraction, please make left/right frames aligned before calling Encoder."
            )

        if self.use_dino:

            self.dvt_vitb14.eval()

        expand_percep_frames = self._resize_perception_frames(x)

        frames = torch.cat([
            x.unsqueeze(2),
            expand_percep_frames,
            y.unsqueeze(2),
        ], dim=2)

        features = self.base_forward(frames, output_final)
        return features


def load_unified_ckpt(model: nn.Module, ckpt_path: str, map_location: str = "cpu", strict: bool = False):

    ckpt = torch.load(ckpt_path, map_location=map_location)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state" in ckpt:
            state_dict = ckpt["model_state"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("model."):
            new_k = new_k[len("model."):]
        cleaned_state_dict[new_k] = v
    state_dict = cleaned_state_dict

    model_state = model.state_dict()
    for key in ("encoder.perception_frames", "perception_frames"):
        if key in state_dict and key in model_state and state_dict[key].shape != model_state[key].shape:
            src = state_dict[key]
            dst = model_state[key]
            _, C, T, H0, W0 = src.shape
            target_h, target_w = dst.shape[-2:]
            src_2d = src.permute(0, 2, 1, 3, 4).reshape(-1, C, H0, W0).float()
            src_2d = F.interpolate(src_2d, size=(target_h, target_w), mode="bilinear", align_corners=False)
            src = src_2d.reshape(1, T, C, target_h, target_w).permute(0, 2, 1, 3, 4).to(dtype=dst.dtype)
            state_dict[key] = src

    msg = model.load_state_dict(state_dict, strict=strict)
    print(f"Loaded unified checkpoint: {ckpt_path}")
    print(msg)
    return msg


