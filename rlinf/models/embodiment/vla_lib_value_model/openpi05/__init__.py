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

"""
OpenPI 0.5 Extended Model Package.

PI0.5 with support for three forwarding modes:
- VLA: Standard flow matching action prediction
- VLM: Pure VLM with cross-entropy loss
- VLM+VLA: Reasoning text generation + action prediction
"""

# Re-export image processor from PI0
from ..openpi.processing_pi0 import PI0ImageProcessor
from .configuration_pi05 import ForwardMode, PI05Config
from .data_collator_pi05 import PI05DataCollator
from .modeling_critic import (
    CriticOutput,
    PI05CriticConfig,
    PI05QCritic,
    PI05ValueCritic,
    QCriticModel,
    ValueCriticModel,
)
from .modeling_pi05 import PI05FlowMatching, PI05ForConditionalGeneration, PI05Output
from .paligemma_with_multi_expert import PaliGemmaWithMultiExpertModel
from .paligemma_with_multi_expert_800m import PaliGemmaWithMultiExpert800M
from .processing_pi05 import PI05Processor
from .static_kv_cache import StaticKVCache, left_to_right_align

__all__ = [
    # Configuration
    "PI05Config",
    "PI05CriticConfig",
    "ForwardMode",
    # Model
    "PI05ForConditionalGeneration",
    "PI05FlowMatching",
    "PI05Output",
    "PaliGemmaWithMultiExpertModel",
    "PaliGemmaWithMultiExpert800M",
    # Critic models
    "PI05ValueCritic",
    "PI05QCritic",
    "CriticOutput",
    "ValueCriticModel",
    "QCriticModel",
    # Processing
    "PI05Processor",
    "PI0ImageProcessor",
    # Data
    "PI05DataCollator",
    # Cache utilities
    "StaticKVCache",
    "left_to_right_align",
]
