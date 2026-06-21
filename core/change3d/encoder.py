# Copyright (c) Duowang Zhu.
# All rights reserved.
import importlib.util
import os
from dinov2.models.vision_transformer import vit_base


from functools import partial
import math
import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union, Callable, Any

import torch
import torch.nn as nn
import torch.utils.checkpoint
from einops import rearrange, repeat

from core.change3d.x3d import create_x3d
from core.change3d.change_decoder import ChangeDecoder
from core.change3d.caption_decoder import CaptionDecoder
from core.change3d.utils import weight_init
from torch.nn import functional as F
import torch.distributed as dist
import numpy as np
def get_resize_keep_aspect_ratio(H, W, divider=16, max_H=1232, max_W=1232):
  assert max_H%divider==0
  assert max_W%divider==0

  def round_by_divider(x):
    return int(np.ceil(x/divider)*divider)

  H_resize = round_by_divider(H)   #!NOTE KITTI width=1242
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
    """
    Encoder model based on X3D architecture with feature enhancement capabilities.

    当前版本：
    1) 训练初始化时会加载 args.pretrained 对应的 X3D 预训练权重。
    2) DINOv2 vit_base 使用 nn.ModuleList 注册到模型中，因此会进入统一 ckpt 保存。
       但 DINOv2 backbone 默认冻结，只作为特征提取器使用。
    3) perception_frames 保持为 learnable parameter，forward 时动态插值到当前输入图片 H/W，
       因此不同分辨率输入不会因为 token 尺寸不一致而 cat 失败。
    """

    def __init__(self, args: Any, embed_dims: List[int]) -> None:
        super().__init__()
        self.args = args
        self.num_perception_frame = int(args.num_perception_frame)

        # Initialize X3D backbone.
        self.x3d = create_x3d(input_clip_length=self.num_perception_frame + 2, depth_factor=5.0)
        self._load_x3d_pretrained()

        # Learnable perception frames.
        # 这里仍然需要一个“基准尺寸”来保存参数；forward 会 resize 到当前输入尺寸。
        init_h = int(getattr(args, "in_height", getattr(args, "height", 224)))
        init_w = int(getattr(args, "in_width", getattr(args, "width", 224)))
        self.perception_frames = nn.Parameter(
            torch.randn(1, 3, self.num_perception_frame, init_h, init_w) * 0.02,
            requires_grad=True,
        )

        # Optional DINOv2 branch.
        self.use_dino = getattr(args, "use_dino", "none") != "none"
        if self.use_dino:
            print("dino_pth:", args.use_dino)
            dvt_weights = torch.load(args.use_dino, map_location="cpu")
            vit_kwargs = dict(
                img_size=518,
                patch_size=14,
                init_values=1.0,
                ffn_layer="mlp",
                block_chunks=0,
            )
            dvt_vitb14 = vit_base(**vit_kwargs).eval()
            dvt_vitb14.load_state_dict(
                dvt_weights,
                strict=False,
            )
            # 注册到模型中，这样 DINOv2 vit_base 会进入 state_dict 并保存到统一 ckpt。
            # 但默认冻结，只作为特征提取器使用。
            for p in dvt_vitb14.parameters():
                p.requires_grad_(False)
            self.dvt_vitb14 = nn.ModuleList([dvt_vitb14])
            self.dino_proj = nn.Sequential(nn.Conv2d(768, 24, 1))
        else:
            self.dvt_vitb14 = None
            self.dino_proj = None

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
        """
        Load X3D pretrained weights during training/model construction.

        支持常见格式：
        - {"model_state": state_dict}
        - {"state_dict": state_dict}
        - {"model": state_dict}
        - 直接就是 state_dict
        """
        pretrained = getattr(self.args, "pretrained", None)
        if pretrained is None or str(pretrained).lower() in ["", "none", "null"]:
            print("[Encoder] Skip X3D pretrained loading because args.pretrained is empty.")
            return
        if not os.path.isfile(pretrained):
            print(f"[Encoder] X3D pretrained not found, skip loading: {pretrained}")
            return

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

        # Remove common wrappers if present.
        cleaned = {}
        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("x3d."):
                nk = nk[len("x3d."):]
            cleaned[nk] = v

        try:
            msg = self.x3d.load_state_dict(cleaned, strict=True)
            print(f"[Encoder] Loaded X3D pretrained: {pretrained}, {msg}")
        except RuntimeError as e:
            msg = self.x3d.load_state_dict(cleaned, strict=False)
            print(f"[Encoder] Loaded X3D pretrained with strict=False: {pretrained}, {msg}")
            print(f"[Encoder] strict=True failed with: {e}")

    def _resize_perception_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize learnable perception frames to the current input image size.

        Args:
            x: [B, C, H, W]

        Returns:
            [B, 3, T, H, W]
        """
        B, _, H, W = x.shape
        p = self.perception_frames.to(device=x.device, dtype=x.dtype)

        if p.shape[-2:] != (H, W):
            # [1, C, T, H0, W0] -> [T, C, H0, W0]
            _, C, T, H0, W0 = p.shape
            p_2d = p.permute(0, 2, 1, 3, 4).reshape(-1, C, H0, W0)
            p_2d = F.interpolate(
                p_2d,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
            # [T, C, H, W] -> [1, C, T, H, W]
            p = p_2d.reshape(1, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()

        # expand 不复制参数；cat 时会自然生成实际输入序列。
        return p.expand(B, -1, -1, -1, -1)

    @torch.cuda.amp.autocast()
    def _process_dino_frame(self, frame: torch.Tensor, target_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        DINOv2 feature extraction.

        frame: [B, C, H, W]
        target_hw: 输出特征需要对齐到的空间尺寸；如果为 None，则对齐到 frame 自身尺寸。
        """
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
        """
        Enhance perception frames using temporal information from pre/post frames.

        x: [B, C, T, H, W], where T = num_perception_frame + 2.
        """
        middle_idx = x.shape[2] // 2

        pre_frame = x[:, :, 0]
        post_frame = x[:, :, self.num_perception_frame + 1]

        middle_frame = x[:, :, middle_idx]
        middlepre_frame = x[:, :, 1]
        middlepost_frame = x[:, :, self.num_perception_frame]

        enhanced_x = x.clone()

        if x0_pre_post is not None:
            # x0_pre_post 是进入 X3D block 之前的原始 pre/post frame。
            # 这里必须把 DINO 输出 resize 到当前 block 的 feature spatial size，
            # 否则输入图像尺寸变化或 X3D block 下采样后会发生 H/W mismatch。
            raw_pre_frame, raw_post_frame = x0_pre_post
            target_hw = middle_frame.shape[-2:]

            dino_feat_pre = self.dino_proj(self._process_dino_frame(raw_pre_frame, target_hw=target_hw))
            dino_feat_post = self.dino_proj(self._process_dino_frame(raw_post_frame, target_hw=target_hw))
            semantic_diff = torch.abs(dino_feat_pre - dino_feat_post)
            enhanced_middle_frame = middle_frame + semantic_diff

            # 原逻辑：num_perception_frame == 3 时，同时增强 pre/post perception frame。
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
        """
        Forward pass through X3D blocks with enhancement.

        x: [B, C, T, H, W]
        """
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
            # DINOv2 已注册为子模块，外部 model.to(device) 会自动移动它；
            # 这里保持 eval，避免 BN/Dropout 等状态变化。
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
    """
    Load one unified checkpoint for train/eval.

    Encoder 初始化时会先加载 args.pretrained 作为 X3D 初始化；
    eval 或 resume 时再用这个函数加载完整统一 ckpt 覆盖当前模型权重。
    如果 ckpt 里的 encoder.perception_frames 和当前模型构建尺寸不同，会先 resize 再加载。
    """
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

    # Remove common wrappers: module.xxx / model.xxx
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


class Trainer(nn.Module):
    """
    Complete model with encoder and decoder for video frame enhancement.
    """
    
    def __init__(self, args: Any) -> None:
        """
        Initialize the trainer with encoder and decoder.
        
        Args:
            args: Configuration arguments
        """
        super().__init__()
        self.args = args
    
        # Define embedding dimensions for each stage
        self.embed_dims = [24, 24, 48, 96]
        
        # Initialize encoder and decoder
        self.encoder = Encoder(args, self.embed_dims)
        
        # For binary change detection and change caption task
        if args.num_perception_frame == 1 and 'CD' in args.dataset:
            self.decoder = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
            # Initialize decoder weights
            weight_init(self.decoder)

        # For semantic change detection task
        elif args.num_perception_frame == 3:
            self.decoder_pre = ChangeDecoder(args, in_dim=self.embed_dims)
            self.decoder_post = ChangeDecoder(args, in_dim=self.embed_dims)
            self.decoder_change = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
            # Initialize decoder weights
            weight_init(self.decoder_pre)
            weight_init(self.decoder_post)
            weight_init(self.decoder_change)

        # For building damage assessment task
        elif args.num_perception_frame == 2:
            self.decoder_cls = ChangeDecoder(args, in_dim=self.embed_dims)
            self.decoder_loc = ChangeDecoder(args, in_dim=self.embed_dims, has_sigmoid=True)
            # Initialize decoder weights
            weight_init(self.decoder_cls)
            weight_init(self.decoder_loc)

        # For change caption task
        elif args.num_perception_frame == 1 and 'CC' in args.dataset:
            self.decoder = CaptionDecoder(args)
        else:
            assert False

    def update_bcd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]

        Returns:
            Predicted frame tensor
        """
        # Extract features using encoder
        features = self.encoder(x, y)

        # perception feature
        perception_change_feat = list(map(lambda x: x[0], features))

        # Generate prediction using decoder
        prediction = self.decoder(perception_change_feat)

        return prediction
    
    def update_scd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]

        Returns:
            Predicted frame tensor
        """
        # Extract features using encoder
        features = self.encoder(x, y)
        
        # Generate prediction using decoder
        perception_pre_feat = list(map(lambda x: x[0], features))
        perception_change_feat = list(map(lambda x: x[1], features))
        perception_post_feat = list(map(lambda x: x[2], features))

        pre_mask = self.decoder_pre(perception_pre_feat)
        post_mask = self.decoder_post(perception_post_feat)
        change_mask = self.decoder_change(perception_change_feat)

        return pre_mask, post_mask, change_mask
    
    def update_bda(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]

        Returns:
            Predicted frame tensor
        """
        # Extract features using encoder
        features = self.encoder(x, y)

        # perception feature
        perception_cls_feat = list(map(lambda x: x[0], features))
        perception_loc_feat = list(map(lambda x: x[1], features))

        # Generate prediction using decoder
        pred_cls = self.decoder_cls(perception_cls_feat)
        pred_loc = self.decoder_loc(perception_loc_feat)

        return pred_cls, pred_loc
    
    def update_cc(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete model.

        Args:
            x: Input frame tensor with shape [B, C, H, W]
            y: Target frame tensor with shape [B, C, H, W]

        Returns:
            Predicted frame tensor
        """
        # Extract features using encoder
        features = self.encoder(x, y, output_final=True)

        return features
