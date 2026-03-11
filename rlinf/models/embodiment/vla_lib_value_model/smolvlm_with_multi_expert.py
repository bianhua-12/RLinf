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

"""SmolVLM backbone with independent expert modules for value modeling."""

import copy
import logging
from typing import Literal

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    SmolVLMForConditionalGeneration,
)

from .paligemma_with_multi_expert import _requires_uniform_dtype

logger = logging.getLogger(__name__)


def resolve_torch_dtype(
    precision: Literal["bfloat16", "float16", "float32"],
) -> torch.dtype:
    """Map the shared value-model precision contract to a torch dtype."""
    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "float16":
        return torch.float16
    if precision == "float32":
        return torch.float32
    raise ValueError(f"Invalid precision: {precision}")


def get_num_key_value_heads(attention: nn.Module) -> int:
    """Infer KV head count across transformers attention implementations."""
    num_kv_heads = getattr(attention, "num_key_value_heads", None)
    if num_kv_heads is not None:
        return int(num_kv_heads)

    head_dim = getattr(attention, "head_dim", None)
    if head_dim is None:
        raise AttributeError("Attention module is missing head_dim")
    out_features = getattr(attention.k_proj, "out_features", None)
    if out_features is None:
        raise AttributeError("Attention module is missing k_proj.out_features")
    if out_features % head_dim != 0:
        raise ValueError(
            f"k_proj out_features ({out_features}) must be divisible by head_dim ({head_dim})"
        )
    return out_features // head_dim


def get_real_image_slot_mask(pixel_values: torch.Tensor) -> torch.Tensor:
    """Match transformers SmolVLM logic for filtering all-zero padded images."""
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(1)
    if pixel_values.dim() != 5:
        raise ValueError(
            "SmolVLM pixel values must have shape [B, N, C, H, W] or [B, C, H, W]"
        )

    flat_pixel_values = pixel_values.reshape(
        pixel_values.shape[0] * pixel_values.shape[1], *pixel_values.shape[2:]
    )
    num_values_per_image = flat_pixel_values.shape[1:].numel()
    real_image_slots = (
        flat_pixel_values.eq(0.0).sum(dim=(-1, -2, -3)) != num_values_per_image
    )
    if not torch.any(real_image_slots):
        real_image_slots[0] = True
    return real_image_slots.view(pixel_values.shape[0], pixel_values.shape[1])


def scatter_image_hidden_states_back_to_slots(
    image_hidden_states: torch.Tensor,
    real_image_slot_mask: torch.Tensor,
) -> torch.Tensor:
    """Restore dropped SmolVLM image slots so downstream code keeps fixed ordering."""
    if real_image_slot_mask.dim() != 2:
        raise ValueError("real_image_slot_mask must have shape [B, N]")
    if image_hidden_states.dim() != 3:
        raise ValueError("image_hidden_states must have shape [R, T, D]")

    batch_size, num_image_slots = real_image_slot_mask.shape
    sequence_length = image_hidden_states.shape[1]
    hidden_size = image_hidden_states.shape[2]
    expected_real_slots = int(real_image_slot_mask.sum().item())
    if image_hidden_states.shape[0] != expected_real_slots:
        raise ValueError(
            "SmolVLM image embedding count does not match the number of non-zero image "
            f"slots. Got {image_hidden_states.shape[0]} embeddings for "
            f"{expected_real_slots} real slots."
        )

    restored = image_hidden_states.new_zeros(
        batch_size * num_image_slots,
        sequence_length,
        hidden_size,
    )
    restored[real_image_slot_mask.reshape(-1)] = image_hidden_states
    return restored.view(batch_size, num_image_slots * sequence_length, hidden_size)


def normalize_attention_mask_for_eager_attention(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert RLinf additive masks into the bool mask expected by eager attention."""
    if attention_mask.dtype == torch.bool:
        return attention_mask
    if torch.is_floating_point(attention_mask):
        return attention_mask >= 0
    return attention_mask != 0


def apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    max_wavelength: int = 10_000,
) -> torch.Tensor:
    """Apply RoPE to tensors shaped ``[B, L, H, D]``."""
    half_dim = x.shape[-1] // 2
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(
        half_dim, dtype=torch.float32, device=x.device
    )
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]

    sin = torch.sin(radians)
    cos = torch.cos(radians)

    x1, x2 = x.split(half_dim, dim=-1)
    out = torch.empty_like(x)
    out[..., :half_dim] = x1 * cos - x2 * sin
    out[..., half_dim:] = x2 * cos + x1 * sin
    return out.to(dtype)


def get_intermediate_size(
    hidden_dim: int,
    ffn_dim_multiplier: float = 4.0,
    multiple_of: int = 256,
) -> int:
    """Match the upstream SmolVLA expert MLP sizing rule."""
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


class SmolVLMExpert(nn.Module):
    """Wrapper that normalizes expert access to ``.model``."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model


class SmolVLMWithMultiExpert(nn.Module):
    """SmolVLM backbone with one or more lightweight language experts."""

    def __init__(
        self,
        expert_names: list[str],
        smolvlm_path: str,
        *,
        precision: Literal["bfloat16", "float16", "float32"] = "bfloat16",
        load_vlm_weights: bool = True,
        freeze_vision_encoder: bool = False,
        freeze_vlm: bool = False,
        attention_mode: str = "cross_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = 16,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.75,
    ):
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_vlm = freeze_vlm
        self.attention_mode = attention_mode
        self.expert_names = expert_names
        self.trainable_experts = expert_names
        self.self_attn_every_n_layers = self_attn_every_n_layers
        vlm_dtype = resolve_torch_dtype(precision)

        if load_vlm_weights:
            logger.info("Loading SmolVLM weights from %s", smolvlm_path)
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                smolvlm_path,
                torch_dtype=vlm_dtype,
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(smolvlm_path)
            self.vlm = SmolVLMForConditionalGeneration(config=config)

        if num_vlm_layers > 0:
            self.get_vlm_model().text_model.layers = (
                self.get_vlm_model().text_model.layers[:num_vlm_layers]
            )
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config

        expert_config = copy.deepcopy(config.text_config)
        hidden_size = expert_config.hidden_size
        expert_config.hidden_size = int(hidden_size * expert_width_multiplier)
        expert_config.intermediate_size = get_intermediate_size(
            expert_config.hidden_size
        )
        expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            if self.num_vlm_layers % num_expert_layers != 0:
                raise ValueError(
                    "num_vlm_layers must be divisible by num_expert_layers for SmolVLM experts"
                )
            expert_config.num_hidden_layers = num_expert_layers
        self.num_expert_layers = expert_config.num_hidden_layers
        self.expert_hidden_size = expert_config.hidden_size

        self.experts = nn.ModuleDict()
        for name in expert_names:
            expert_model = AutoModel.from_config(expert_config)
            if hasattr(expert_model, "embed_tokens"):
                expert_model.embed_tokens = None
            self.experts[name] = SmolVLMExpert(expert_model)

        if "cross" in attention_mode:
            self._reshape_cross_attention_projections()

        self._apply_precision(precision)
        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads
        self._set_requires_grad()

    def get_vlm_model(self) -> nn.Module:
        """Return the SmolVLM multimodal model without LM head."""
        return self.vlm.model

    def _reshape_cross_attention_projections(self) -> None:
        """Project cached VLM KV into expert KV dimensions."""
        for expert in self.experts.values():
            for layer_idx in range(len(expert.model.layers)):
                if (
                    self.self_attn_every_n_layers > 0
                    and layer_idx % self.self_attn_every_n_layers == 0
                ):
                    continue
                layer = expert.model.layers[layer_idx]
                expert_num_kv_heads = get_num_key_value_heads(layer.self_attn)
                layer.self_attn.cross_k_proj = nn.Linear(
                    self.config.text_config.num_key_value_heads
                    * self.config.text_config.head_dim,
                    expert_num_kv_heads * layer.self_attn.head_dim,
                    bias=self.config.text_config.attention_bias,
                )
                layer.self_attn.cross_v_proj = nn.Linear(
                    self.config.text_config.num_key_value_heads
                    * self.config.text_config.head_dim,
                    expert_num_kv_heads * layer.self_attn.head_dim,
                    bias=self.config.text_config.attention_bias,
                )

    def _set_requires_grad(self) -> None:
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for param in self.get_vlm_model().vision_model.parameters():
                param.requires_grad = False

        if self.freeze_vlm:
            self.vlm.eval()
            for param in self.vlm.parameters():
                param.requires_grad = False

    def _apply_precision(
        self, precision: Literal["bfloat16", "float16", "float32"]
    ) -> None:
        """Apply the shared precision contract while staying FSDP-safe."""
        dtype = resolve_torch_dtype(precision)
        self.to(dtype=dtype)

        if precision == "float32":
            return

        if _requires_uniform_dtype():
            logger.info(
                "Parameter sharding detected (FSDP/Zero-3): using uniform %s",
                precision,
            )
            return

        logger.info(
            "Applying mixed precision for SmolVLM: %s backbone with fp32 norms",
            precision,
        )
        params_to_keep_float32 = [
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def train(self, mode: bool = True):
        """Preserve frozen backbone modules in eval mode."""
        super().train(mode)
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
        if self.freeze_vlm:
            self.vlm.eval()
        return self

    def embed_image(
        self,
        image: torch.Tensor,
        pixel_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode one or more images into SmolVLM text space."""
        if image.dim() == 4:
            image = image.unsqueeze(1)
        if pixel_attention_mask is not None and pixel_attention_mask.dim() == 3:
            pixel_attention_mask = pixel_attention_mask.unsqueeze(1)
        real_image_slot_mask = get_real_image_slot_mask(image)
        image_hidden_states = self.get_vlm_model().get_image_features(
            pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
            pixel_attention_mask=pixel_attention_mask,
        )
        return scatter_image_hidden_states_back_to_slots(
            image_hidden_states=image_hidden_states,
            real_image_slot_mask=real_image_slot_mask,
        )

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed text tokens with SmolVLM text embeddings."""
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def get_model_layers(self, expert_name: str) -> list[list[nn.Module | None]]:
        """Map expert layers onto the VLM layer schedule."""
        vlm_layers = []
        expert_layers = []
        expert_model = self.experts[expert_name].model
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = expert_model.layers[expert_index]
            vlm_layers.append(self.get_vlm_model().text_model.layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def eager_attention_forward(
        self,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> torch.Tensor:
        """Shared eager attention used by upstream SmolVLA."""
        num_groups = self.num_attention_heads // self.num_key_value_heads
        sequence_length = key_states.shape[1]
        attention_mask = normalize_attention_mask_for_eager_attention(attention_mask)

        key_states = key_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            num_groups,
            head_dim,
        )
        key_states = key_states.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads * num_groups,
            head_dim,
        )
        value_states = value_states[:, :, :, None, :].expand(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            num_groups,
            head_dim,
        )
        value_states = value_states.reshape(
            batch_size,
            sequence_length,
            self.num_key_value_heads * num_groups,
            head_dim,
        )

        query_states = query_states.to(dtype=torch.float32).transpose(1, 2)
        key_states = key_states.to(dtype=torch.float32).transpose(1, 2)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights *= head_dim**-0.5
        big_neg = torch.finfo(attn_weights.dtype).min
        masked_attn = torch.where(attention_mask[:, None, :, :], attn_weights, big_neg)
        probs = nn.functional.softmax(masked_attn, dim=-1).to(dtype=value_states.dtype)
        attn_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))
        attn_output = attn_output.permute(0, 2, 1, 3)
        return attn_output.reshape(
            batch_size,
            -1,
            self.num_key_value_heads * num_groups * head_dim,
        )

    def forward_attn_layer(
        self,
        model_layers: list[list[nn.Module | None]],
        inputs_embeds: list[torch.Tensor | None],
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values: dict | None = None,
    ) -> tuple[list[torch.Tensor], dict | None]:
        """Run one layer with shared self-attention."""
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_states.append(
                layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            )
            key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape))
            value_states.append(
                layer.self_attn.v_proj(hidden_states).view(hidden_shape)
            )

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        position_ids = position_ids[:, :seq_len]
        attention_mask = attention_mask[:, :seq_len, :seq_len]

        query_states = apply_rope(query_states, position_ids)
        key_states = apply_rope(key_states, position_ids)

        if use_cache and past_key_values is None:
            past_key_values = {}
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat(
                    [past_key_values[layer_idx]["key_states"], key_states], dim=1
                )
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states], dim=1
                )

        att_output = self.eager_attention_forward(
            attention_mask,
            batch_size,
            head_dim,
            query_states,
            key_states,
            value_states,
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers: list[list[nn.Module | None]],
        inputs_embeds: list[torch.Tensor | None],
        layer_idx: int,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        head_dim: int,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values: dict | None = None,
    ) -> tuple[list[torch.Tensor | None], dict | None]:
        """Run one layer with VLM self-attn + expert cross-attn."""
        att_outputs: list[torch.Tensor | None] = []

        if len(inputs_embeds) == 2 and not past_key_values:
            seq_len = inputs_embeds[0].shape[1]
            prefix_pos = position_ids[:, :seq_len]
            expert_pos = position_ids[:, seq_len:]
            prefix_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]
            hidden_states = layer.input_layernorm(inputs_embeds[0])
            hidden_shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_states = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_states = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states = apply_rope(query_states, prefix_pos)
            key_states = apply_rope(key_states, prefix_pos)
            att_outputs.append(
                self.eager_attention_forward(
                    prefix_mask,
                    batch_size,
                    head_dim,
                    query_states,
                    key_states,
                    value_states,
                )
            )
        else:
            expert_pos = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden = expert_layer.input_layernorm(inputs_embeds[1])
            expert_shape = (
                *expert_hidden.shape[:-1],
                -1,
                expert_layer.self_attn.head_dim,
            )
            expert_hidden = expert_hidden.to(
                dtype=expert_layer.self_attn.q_proj.weight.dtype
            )
            expert_query = expert_layer.self_attn.q_proj(expert_hidden).view(
                expert_shape
            )

            flat_key = key_states.to(
                dtype=expert_layer.self_attn.k_proj.weight.dtype
            ).view(
                *key_states.shape[:2],
                -1,
            )
            cross_k_proj = getattr(expert_layer.self_attn, "cross_k_proj")
            expert_key = cross_k_proj(flat_key).view(
                *flat_key.shape[:-1],
                -1,
                expert_layer.self_attn.head_dim,
            )
            flat_value = value_states.to(
                dtype=expert_layer.self_attn.v_proj.weight.dtype
            ).view(
                *value_states.shape[:2],
                -1,
            )
            cross_v_proj = getattr(expert_layer.self_attn, "cross_v_proj")
            expert_value = cross_v_proj(flat_value).view(
                *flat_value.shape[:-1],
                -1,
                expert_layer.self_attn.head_dim,
            )

            expert_pos = expert_pos - torch.min(expert_pos, dim=1, keepdim=True).values
            expert_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key.shape[1]
            ]
            expert_query = apply_rope(expert_query, expert_pos)
            att_outputs.append(
                self.eager_attention_forward(
                    expert_mask,
                    batch_size,
                    head_dim,
                    expert_query,
                    expert_key,
                    expert_value,
                )
            )
        else:
            att_outputs.append(None)

        return att_outputs, past_key_values

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: dict | None = None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool | None = None,
        expert_name: str | None = None,
        fill_kv_cache: bool = True,
        **kwargs,
    ) -> tuple[list[torch.Tensor | None], dict | None]:
        """Run SmolVLM and one expert with upstream-compatible attention."""
        del kwargs
        if attention_mask is None or position_ids is None or inputs_embeds is None:
            raise ValueError(
                "attention_mask, position_ids and inputs_embeds are required"
            )
        if attention_mask.dim() == 4:
            attention_mask = attention_mask[:, 0, :, :]
        if expert_name is None:
            if len(self.expert_names) != 1:
                raise ValueError(
                    "expert_name must be specified for multi-expert SmolVLM"
                )
            expert_name = self.expert_names[0]

        model_layers = self.get_model_layers(expert_name)
        batch_size = next(
            hidden.shape[0] for hidden in inputs_embeds if hidden is not None
        )
        head_dim = self.vlm.config.text_config.head_dim
        num_layers = self.num_vlm_layers

        for layer_idx in range(num_layers):
            use_cross_attn = (
                not fill_kv_cache
                and "cross" in self.attention_mode
                and not (
                    self.self_attn_every_n_layers > 0
                    and layer_idx % self.self_attn_every_n_layers == 0
                )
            )
            if use_cross_attn:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=bool(use_cache),
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=bool(use_cache),
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                if hidden_states is None:
                    outputs_embeds.append(None)
                    continue
                if layer is None:
                    outputs_embeds.append(hidden_states)
                    continue

                end = start + hidden_states.shape[1]
                if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                    att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                att_out = att_output[:, start:end]
                out_emb = layer.self_attn.o_proj(att_out)
                out_emb = out_emb + hidden_states
                after_residual = out_emb.clone()
                out_emb = layer.post_attention_layernorm(out_emb)
                out_emb = layer.mlp(out_emb)
                out_emb = out_emb + after_residual
                outputs_embeds.append(out_emb)
                start = end if len(att_outputs) == 1 else 0
            inputs_embeds = outputs_embeds

        outputs = []
        norms = [
            self.get_vlm_model().text_model.norm,
            self.experts[expert_name].model.norm,
        ]
        for hidden_states, norm in zip(inputs_embeds, norms, strict=True):
            outputs.append(None if hidden_states is None else norm(hidden_states))
        return outputs, past_key_values
