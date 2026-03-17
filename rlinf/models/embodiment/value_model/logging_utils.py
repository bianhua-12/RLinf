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

"""Utilities for standardized value-model metric logging."""

from __future__ import annotations

from typing import Any

import torch

VALUE_LOG_METRIC_KEYS: tuple[str, ...] = (
    "loss",
    "expert_loss",
    "predicted_value_mean",
    "predicted_value_std",
    "target_value_mean",
    "target_value_std",
    "cat_acc_best",
    "cat_acc_neighbor",
    "cat_mae",
    "mse",
    "regression_mae",
    "value_mae",
)


def _to_float_metric(value: Any) -> float:
    """Convert metric scalars to Python floats."""
    if isinstance(value, torch.Tensor):
        return value.detach().item()
    return float(value)


def build_value_metric_dict(
    result: Any,
    loss: torch.Tensor,
    target_values: torch.Tensor | None,
) -> dict[str, float]:
    """Build a standardized metric dict for value-model train/eval steps.

    The returned dict always contains the union of categorical and regression
    metric keys. Metrics that are not produced by the current loss type are
    emitted as ``NaN`` so downstream loggers receive a stable schema.

    Args:
        result: Model forward result with value-model metric attributes.
        loss: Scalar loss tensor returned by the model.
        target_values: Optional supervision tensor for target statistics.

    Returns:
        A metric dictionary containing all keys in ``VALUE_LOG_METRIC_KEYS``.
    """
    metrics = {key: float("nan") for key in VALUE_LOG_METRIC_KEYS}
    metrics["loss"] = _to_float_metric(loss)

    if getattr(result, "expert_loss", None) is not None:
        metrics["expert_loss"] = _to_float_metric(result.expert_loss)

    predicted_values = getattr(result, "predicted_values", None)
    if predicted_values is not None:
        metrics["predicted_value_mean"] = predicted_values.detach().mean().item()
        metrics["predicted_value_std"] = predicted_values.detach().std().item()

    if target_values is not None:
        metrics["target_value_mean"] = target_values.detach().mean().item()
        metrics["target_value_std"] = target_values.detach().std().item()

    if getattr(result, "cat_acc_best", None) is not None:
        metrics["cat_acc_best"] = _to_float_metric(result.cat_acc_best)
    if getattr(result, "cat_acc_neighbor", None) is not None:
        metrics["cat_acc_neighbor"] = _to_float_metric(result.cat_acc_neighbor)
    if getattr(result, "cat_mae", None) is not None:
        metrics["cat_mae"] = _to_float_metric(result.cat_mae)
        metrics["value_mae"] = metrics["cat_mae"]

    if getattr(result, "mse", None) is not None:
        metrics["mse"] = _to_float_metric(result.mse)
    if getattr(result, "regression_mae", None) is not None:
        metrics["regression_mae"] = _to_float_metric(result.regression_mae)
        metrics["value_mae"] = metrics["regression_mae"]

    return metrics
