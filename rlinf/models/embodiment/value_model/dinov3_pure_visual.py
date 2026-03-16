# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen DINOv3 backbone with a lightweight visual aggregator."""

import logging
from typing import Literal

import torch
from torch import Tensor, nn
from transformers import AutoModel

from .paligemma_with_multi_expert import _requires_uniform_dtype

logger = logging.getLogger(__name__)


class Dinov3PureVisualModel(nn.Module):
    """Pure-visual value backbone built on frozen DINOv3 patch features."""

    def __init__(
        self,
        vision_encoder_path: str,
        hidden_size: int = 256,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        freeze_vision_encoder: bool = True,
        freeze_vlm: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        max_cameras: int = 8,
    ):
        super().__init__()
        if not freeze_vlm:
            raise ValueError(
                "Dinov3PureVisualModel requires freeze_vlm=True because the "
                "DINOv3 backbone is permanently frozen in this path."
            )

        self.freeze_vlm = True
        self.freeze_vision_encoder = True
        if not freeze_vision_encoder:
            logger.info(
                "Ignoring freeze_vision_encoder=False for dinov3_pure_visual; "
                "the vision backbone is always frozen."
            )

        logger.info("Loading DINOv3 visual encoder from %s", vision_encoder_path)
        self.vision_tower = AutoModel.from_pretrained(
            vision_encoder_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        vision_hidden = self._get_vision_hidden_size()
        self.hidden_size = hidden_size

        self.vision_proj = nn.Linear(vision_hidden, hidden_size, bias=True)
        self.camera_embedding = nn.Embedding(max_cameras, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=int(hidden_size * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.aggregator = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.output_norm = nn.LayerNorm(hidden_size)

        nn.init.normal_(self.vision_proj.weight, std=0.02)
        nn.init.zeros_(self.vision_proj.bias)

        self._apply_precision(precision)
        self._set_requires_grad()

    def _flatten_vision_features(self, feats: Tensor) -> Tensor:
        """Normalize DINO outputs into `[B, num_tokens, hidden]` format."""
        hidden_size = self._get_vision_hidden_size()
        if feats.ndim == 3:
            return feats
        if feats.ndim != 4:
            raise ValueError(f"Unsupported DINO feature shape: {tuple(feats.shape)}")

        if feats.shape[1] == hidden_size:
            return feats.flatten(2).transpose(1, 2)
        if feats.shape[-1] == hidden_size:
            return feats.flatten(1, 2)
        raise ValueError(
            f"Could not infer hidden dimension from DINO features: {tuple(feats.shape)}"
        )

    def _get_vision_hidden_size(self) -> int:
        """Return the channel width for DINO features across HF wrappers."""
        hidden_size = getattr(self.vision_tower.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.vision_tower.config, "num_features", None)
        if hidden_size is None:
            raise ValueError(
                "Could not determine DINOv3 hidden size from config. Expected "
                "`hidden_size` or `num_features` to be set."
            )
        return int(hidden_size)

    def embed_image(self, image: Tensor, camera_index: int) -> Tensor:
        """Embed a single camera image batch into patch tokens."""
        feats = self.vision_tower(pixel_values=image).last_hidden_state
        feats = self._flatten_vision_features(feats)
        feats = self.vision_proj(feats)
        camera_ids = torch.full(
            (feats.shape[0], 1),
            camera_index,
            dtype=torch.long,
            device=feats.device,
        )
        return feats + self.camera_embedding(camera_ids)

    def forward(
        self, images: list[Tensor], img_masks: list[Tensor], cls_emb: Tensor
    ) -> Tensor:
        """Aggregate multi-camera DINO patch tokens and return the CLS state."""
        tokens = []
        valid_masks = []
        for camera_index, (image, img_mask) in enumerate(
            zip(images, img_masks, strict=True)
        ):
            img_tokens = self.embed_image(image, camera_index=camera_index)
            tokens.append(img_tokens)
            valid_masks.append(img_mask[:, None].expand(img_tokens.shape[:2]))

        patch_tokens = torch.cat(tokens, dim=1)
        patch_valid = torch.cat(valid_masks, dim=1)
        full_tokens = torch.cat([cls_emb, patch_tokens], dim=1)
        full_valid = torch.cat(
            [
                torch.ones(cls_emb.shape[:2], dtype=torch.bool, device=cls_emb.device),
                patch_valid,
            ],
            dim=1,
        )

        aggregated = self.aggregator(
            full_tokens,
            src_key_padding_mask=~full_valid,
        )
        return self.output_norm(aggregated[:, 0, :])

    def _set_requires_grad(self):
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.vision_tower.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_tower.eval()
        return self

    def _apply_precision(self, precision: Literal["bfloat16", "float32"]):
        if precision == "float32":
            self.to(dtype=torch.float32)
            return
        if precision != "bfloat16":
            raise ValueError(f"Invalid precision: {precision}")

        self.to(dtype=torch.bfloat16)
        if _requires_uniform_dtype():
            logger.info(
                "Parameter sharding detected (FSDP/Zero-3): using uniform bfloat16"
            )
