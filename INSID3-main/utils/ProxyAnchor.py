"""Iterative Proxy Anchor queries for global sliding-window tokens."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def proxy_anchor(
    query: torch.Tensor,
    key_global: torch.Tensor,
    rho: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iteratively replace each query by its normalized positive-key mean.

    Returns ``(Q_proxy, S_proxy, positive_count)``. If a query has no key with
    cosine similarity strictly greater than ``rho``, that query is retained.
    The returned count is from the final aggregation round (or from the input
    query when ``iterations == 0``).
    """

    # error handling
    if not isinstance(query, torch.Tensor) or not isinstance(key_global, torch.Tensor):
        raise TypeError("query and key_global must be torch.Tensor instances")
    if query.ndim != 2 or key_global.ndim != 2:
        raise ValueError("query and key_global must have shape (tokens, channels)")
    if query.shape[0] == 0 or key_global.shape[0] == 0:
        raise ValueError("query and key_global must contain at least one token")
    if query.shape[1] != key_global.shape[1]:
        raise ValueError("query and key_global channel dimensions must match")
    if not query.is_floating_point() or not key_global.is_floating_point():
        raise TypeError("query and key_global must have floating-point dtypes")
    if query.device != key_global.device:
        raise ValueError("query and key_global must be on the same device")
    if query.dtype != key_global.dtype:
        raise ValueError("query and key_global must have the same dtype")
    if not -1.0 <= float(rho) <= 1.0:
        raise ValueError("rho must be in [-1, 1]")
    if isinstance(iterations, bool) or int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")

    keys = F.normalize(key_global, p=2, dim=1)
    proxy = F.normalize(query, p=2, dim=1)
    positive_count: torch.Tensor | None = None

    for _ in range(int(iterations)):
        similarity = proxy @ keys.transpose(0, 1)
        positives = similarity > float(rho)
        positive_count = positives.sum(dim=1)
        nonempty = positive_count > 0
        if nonempty.any():
            summed = positives[nonempty].to(keys.dtype) @ keys
            means = summed / positive_count[nonempty, None].to(keys.dtype)
            updated = proxy.clone()
            updated[nonempty] = F.normalize(means, p=2, dim=1)
            proxy = updated

    similarity_proxy = proxy @ keys.transpose(0, 1)
    if positive_count is None:
        positive_count = (similarity_proxy > float(rho)).sum(dim=1)
    return proxy, similarity_proxy, positive_count


class ProxyAnchor(nn.Module):
    """Configurable module wrapper around :func:`proxy_anchor`."""

    def __init__(self, rho: float, iterations: int) -> None:
        super().__init__()
        self.rho = rho
        self.iterations = iterations

    def forward(
        self,
        query: torch.Tensor,
        key_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return proxy_anchor(query, key_global, self.rho, self.iterations)
