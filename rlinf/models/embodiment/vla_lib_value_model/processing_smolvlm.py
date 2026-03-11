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

"""SmolVLM-specific processor shim for value-model training and inference."""

from typing import Any, Optional

import numpy as np
import torch
from transformers import AutoProcessor, BatchFeature, PreTrainedTokenizerBase

from .processing import PI05Processor


class SmolVLMImageProcessorAdapter:
    """Adapter that accepts RLinf's multi-camera tensor dict format."""

    def __init__(self, base_image_processor: Any):
        self.base_image_processor = base_image_processor

    def __call__(
        self,
        images: dict[str, torch.Tensor],
        image_masks: Optional[dict[str, torch.Tensor]] = None,
        return_tensors: Optional[str] = "pt",
        train: bool = False,
    ) -> BatchFeature:
        del train
        if not images:
            raise ValueError("SmolVLM image processor requires at least one image")

        sorted_keys = sorted(images)
        batch_size = next(iter(images.values())).shape[0]
        nested_images: list[list[np.ndarray]] = []
        image_presence: list[list[bool]] = []

        for batch_idx in range(batch_size):
            sample_images = []
            sample_presence = []
            for key in sorted_keys:
                image = images[key][batch_idx]
                is_present = True
                if image_masks and key in image_masks:
                    mask_value = image_masks[key][batch_idx]
                    if isinstance(mask_value, torch.Tensor):
                        is_present = bool(mask_value.item())
                    else:
                        is_present = bool(mask_value)
                sample_presence.append(is_present)
                sample_images.append(
                    _image_to_uint8_numpy(image, zero_fill=not is_present)
                )
            nested_images.append(sample_images)
            image_presence.append(sample_presence)

        encoding = self.base_image_processor(
            nested_images,
            return_tensors=return_tensors,
        )
        encoding["image_masks"] = torch.tensor(image_presence, dtype=torch.bool)
        return encoding


def _image_to_uint8_numpy(
    image: torch.Tensor, *, zero_fill: bool = False
) -> np.ndarray:
    """Convert one camera image to HWC uint8 while preserving fixed camera slots."""
    if image.dim() == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    image = image.detach().cpu()
    if image.dtype != torch.uint8:
        if image.max() <= 1.0 and image.min() >= -1.0:
            image = (image.clamp(-1.0, 1.0) + 1.0) * 127.5
        image = image.clamp(0, 255).to(torch.uint8)
    image_np = image.numpy()
    if zero_fill:
        return np.zeros_like(image_np)
    return image_np


class SmolVLMProcessor(PI05Processor):
    """Processor that reuses PI05 tokenization with SmolVLM image preprocessing."""

    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_token_len: int = 200,
        tokenizer_name_or_path: Optional[str] = None,
        smolvlm_path: Optional[str] = None,
        **kwargs,
    ):
        processor_source = smolvlm_path or tokenizer_name_or_path
        if processor_source is None:
            raise ValueError(
                "SmolVLMProcessor requires smolvlm_path or tokenizer_name_or_path"
            )

        base_processor = AutoProcessor.from_pretrained(processor_source)
        if tokenizer is None:
            tokenizer = base_processor.tokenizer

        image_processor = SmolVLMImageProcessorAdapter(base_processor.image_processor)
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_token_len=max_token_len,
            tokenizer_name_or_path=tokenizer_name_or_path,
            **kwargs,
        )
        self.smolvlm_path = processor_source


def get_value_model_processor(
    *,
    backbone_variant: str,
    max_token_len: int,
    tokenizer_name_or_path: Optional[str] = None,
    smolvlm_path: Optional[str] = None,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    discrete_state_input: bool = False,
    exclude_cot_from_kv_cache: bool = False,
) -> PI05Processor:
    """Build the correct processor for the configured value-model backbone."""
    if backbone_variant == "smolvlm":
        return SmolVLMProcessor(
            tokenizer=tokenizer,
            max_token_len=max_token_len,
            tokenizer_name_or_path=tokenizer_name_or_path,
            smolvlm_path=smolvlm_path,
            discrete_state_input=discrete_state_input,
            exclude_cot_from_kv_cache=exclude_cot_from_kv_cache,
        )

    return PI05Processor(
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        tokenizer_name_or_path=tokenizer_name_or_path,
        discrete_state_input=discrete_state_input,
        exclude_cot_from_kv_cache=exclude_cot_from_kv_cache,
    )
