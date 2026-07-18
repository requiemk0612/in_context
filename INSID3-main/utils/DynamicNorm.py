"""Dynamic normalization and masked global-token attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def dynamic_normalization(
    similarity_proxy: torch.Tensor,
    value_global: torch.Tensor,
    positive_count: torch.Tensor,
    window_count: int,
    lambda1: float,
    lambda2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply row-wise dynamic normalization and return features and attention.

    Negative normalized logits are masked before softmax. If this would mask a
    complete row, the maximum-similarity key is retained for that row.
    """

    # error handling
    if not isinstance(similarity_proxy, torch.Tensor) or not isinstance(value_global, torch.Tensor):
        raise TypeError("similarity_proxy and value_global must be torch.Tensor instances")
    if not isinstance(positive_count, torch.Tensor):
        raise TypeError("positive_count must be a torch.Tensor")
    if similarity_proxy.ndim != 2 or value_global.ndim != 2:
        raise ValueError("similarity_proxy and value_global must be two-dimensional")
    if positive_count.ndim != 1 or positive_count.shape[0] != similarity_proxy.shape[0]:
        raise ValueError("positive_count must have one entry per query row")
    if similarity_proxy.shape[0] == 0 or similarity_proxy.shape[1] == 0:
        raise ValueError("similarity_proxy must contain at least one query and key")
    if similarity_proxy.shape[1] != value_global.shape[0]:
        raise ValueError("the similarity key dimension must match value_global tokens")
    if not similarity_proxy.is_floating_point() or not value_global.is_floating_point():
        raise TypeError("similarity_proxy and value_global must have floating-point dtypes")
    if similarity_proxy.device != value_global.device or positive_count.device != similarity_proxy.device:
        raise ValueError("similarities, values, and counts must be on the same device")
    if similarity_proxy.dtype != value_global.dtype:
        raise ValueError("similarity_proxy and value_global must have the same dtype")
    if isinstance(window_count, bool) or int(window_count) != window_count or window_count <= 0:
        raise ValueError("window_count must be a positive integer")
    if not math.isfinite(float(lambda1)) or not math.isfinite(float(lambda2)):
        raise ValueError("lambda1 and lambda2 must be finite")
    if (positive_count < 0).any():
        raise ValueError("positive_count must be non-negative")

    row_mean = similarity_proxy.mean(dim=1, keepdim=True)
    u = similarity_proxy.new_tensor(
        1.0 + float(lambda1) * math.log1p(int(window_count))
    )
    # Empty positive sets can occur for rho == 1. Proxy Anchor retains the
    # query in that case; clamp to one keeps DynamicNorm finite as well.
    safe_count = positive_count.to(similarity_proxy.dtype).clamp_min(1)
    w = 1.0 + float(lambda2) / safe_count
    logits = w[:, None] * (similarity_proxy - u * row_mean)

    keep = logits >= 0
    empty = ~keep.any(dim=1)
    if empty.any():
        fallback = similarity_proxy[empty].argmax(dim=1, keepdim=True)
        keep[empty] = False
        empty_rows = keep[empty]
        empty_rows.scatter_(1, fallback, True)
        keep[empty] = empty_rows

    masked_logits = logits.masked_fill(~keep, -torch.inf)
    # Float32 softmax avoids fp16/bfloat16 underflow; the result is cast back
    # before multiplying the original V_global values.
    attention = torch.softmax(masked_logits.float(), dim=1).to(value_global.dtype)
    output = attention @ value_global
    return output, attention


class DynamicNormalization(nn.Module):
    """Configurable module wrapper around :func:`dynamic_normalization`."""

    def __init__(self, lambda1: float, lambda2: float) -> None:
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(
        self,
        similarity_proxy: torch.Tensor,
        value_global: torch.Tensor,
        positive_count: torch.Tensor,
        window_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return dynamic_normalization(
            similarity_proxy,
            value_global,
            positive_count,
            window_count,
            self.lambda1,
            self.lambda2,
        )


DynamicNorm = DynamicNormalization
dynamic_norm = dynamic_normalization
