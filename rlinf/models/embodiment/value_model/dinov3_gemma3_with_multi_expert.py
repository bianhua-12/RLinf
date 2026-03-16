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

"""DINOv3 + Gemma3 backbone with frozen encoders and independent experts."""

import logging
from typing import Literal

import torch
from torch import nn
from transformers import AutoModel, Gemma3ForCausalLM, GemmaForCausalLM
from transformers.cache_utils import DynamicCache
from transformers.models.auto import CONFIG_MAPPING

from .paligemma_with_multi_expert import _requires_uniform_dtype

logger = logging.getLogger(__name__)


class Dinov3Gemma3WithMultiExpert(nn.Module):
    """Frozen DINOv3 + Gemma3 backbone with trainable Gemma experts.

    DINOv3 and Gemma3 are always frozen for this backbone. A trainable projection
    maps DINOv3 image features into Gemma3 hidden space, then a Gemma expert
    consumes Gemma3's KV cache in a second stage forward.
    """

    def __init__(
        self,
        expert_configs: dict,
        vision_encoder_path: str,
        gemma3_path: str,
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        trainable_experts: list[str] | None = None,
    ):
        super().__init__()
        if not freeze_vlm:
            raise ValueError(
                "Dinov3Gemma3WithMultiExpert requires freeze_vlm=True because "
                "the DINOv3 and Gemma3 backbones are permanently frozen."
            )

        self.freeze_vision_encoder = True
        self.freeze_vlm = True
        self.expert_names = list(expert_configs.keys())
        self.trainable_experts = (
            trainable_experts if trainable_experts is not None else self.expert_names
        )

        if freeze_vision_encoder is False:
            logger.info(
                "Ignoring freeze_vision_encoder=False for dinov3_gemma3; "
                "vision encoder is always frozen."
            )

        logger.info(
            "Creating Dinov3Gemma3WithMultiExpert: experts=%s, vision=%s, gemma3=%s",
            self.expert_names,
            vision_encoder_path,
            gemma3_path,
        )

        logger.info("  Loading DINOv3 vision encoder from %s", vision_encoder_path)
        self.vision_tower = AutoModel.from_pretrained(
            vision_encoder_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        vision_hidden = self.vision_tower.config.hidden_size

        logger.info("  Loading Gemma3 from %s", gemma3_path)
        self.gemma3 = Gemma3ForCausalLM.from_pretrained(gemma3_path)
        gemma3_hidden = self.gemma3.config.hidden_size

        self.multi_modal_proj = nn.Linear(vision_hidden, gemma3_hidden, bias=True)
        nn.init.normal_(self.multi_modal_proj.weight, std=0.02)
        nn.init.zeros_(self.multi_modal_proj.bias)
        logger.info(
            "  Fresh projection: %s -> %s (randomly initialized)",
            vision_hidden,
            gemma3_hidden,
        )

        gemma3_head_dim = self.gemma3.config.head_dim
        self.experts = nn.ModuleDict()
        for name, expert_config in expert_configs.items():
            if expert_config.head_dim != gemma3_head_dim:
                raise ValueError(
                    f"Expert '{name}' has head_dim={expert_config.head_dim} but "
                    f"Gemma3 270M has head_dim={gemma3_head_dim}. They must match "
                    "for KV cache cross-attention."
                )
            expert_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=expert_config.head_dim,
                hidden_size=expert_config.width,
                intermediate_size=expert_config.mlp_dim,
                num_attention_heads=expert_config.num_heads,
                num_hidden_layers=expert_config.depth,
                num_key_value_heads=expert_config.num_kv_heads,
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="float32",
            )
            expert = GemmaForCausalLM(config=expert_config_hf)
            expert.model.embed_tokens = None
            self.experts[name] = expert

        self._apply_precision(precision)
        self._set_requires_grad()

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed images with DINOv3 and project them into Gemma3 hidden space."""
        feats = self.vision_tower(pixel_values=image).last_hidden_state
        return self.multi_modal_proj(feats)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using Gemma3 token embeddings."""
        return self.gemma3.model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: list | None = None,
        use_cache: bool | None = None,
        expert_name: str | None = None,
        **kwargs,
    ):
        """Run two-stage Gemma3 prefix + expert suffix forward."""
        del use_cache, kwargs
        prefix_embs, suffix_embs = inputs_embeds

        if suffix_embs is None:
            if past_key_values is None:
                past_key_values = DynamicCache()
            out = self.gemma3.model(
                inputs_embeds=prefix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            return [out.last_hidden_state, None], out.past_key_values

        if prefix_embs is None:
            expert_name = self._resolve_expert_name(expert_name)
            out = self.experts[expert_name].model(
                inputs_embeds=suffix_embs,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=False,
            )
            return [None, out.last_hidden_state], None

        raise ValueError(
            "Dinov3Gemma3WithMultiExpert does not support interleaved forward. "
            "Use two-stage forward via ValueCriticModel._forward_expert_two_stage."
        )

    def _set_requires_grad(self):
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        self.vision_tower.eval()

        for param in self.gemma3.parameters():
            param.requires_grad = False
        self.gemma3.eval()

        for name in self.expert_names:
            if name not in self.trainable_experts:
                for param in self.experts[name].parameters():
                    param.requires_grad = False
                self.experts[name].eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.vision_tower.eval()
        self.gemma3.eval()
        for name in self.expert_names:
            if name not in self.trainable_experts:
                self.experts[name].eval()
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
            return

        params_to_keep_float32 = [
            "multi_modal_proj",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(sel in name for sel in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _resolve_expert_name(self, expert_name: str | None) -> str:
        if expert_name is not None:
            if expert_name not in self.expert_names:
                raise ValueError(
                    f"Unknown expert: {expert_name}. Available: {self.expert_names}"
                )
            return expert_name
        if len(self.expert_names) == 1:
            return self.expert_names[0]
        raise ValueError(
            "expert_name must be specified when multiple experts exist: "
            f"{self.expert_names}"
        )
