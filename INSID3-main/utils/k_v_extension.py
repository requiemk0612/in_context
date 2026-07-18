"""Key/value extension over ordered sliding-window patch features."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _check_feature(feature: torch.Tensor, name: str) -> None:
    if not isinstance(feature, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if feature.ndim != 2:
        raise ValueError(f"{name} must have shape (tokens, channels), got {tuple(feature.shape)}")
    if feature.shape[0] == 0 or feature.shape[1] == 0:
        raise ValueError(f"{name} must not contain an empty dimension")
    if not feature.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {feature.dtype}")


def build_global_key_value(
    window_features: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build K_global and V_global without deduplicating overlaps.

    Each input must be a flattened final-layer patch feature tensor with shape
    ``(N_l, C)``. Input order and token order within every window are retained.
    """
    if not isinstance(window_features, Sequence) or len(window_features) == 0:
        raise ValueError("window_features must be a non-empty sequence")

    first = window_features[0]
    _check_feature(first, "window_features[0]")
    channels = first.shape[1]
    device = first.device
    dtype = first.dtype
    for index, feature in enumerate(window_features[1:], start=1):
        _check_feature(feature, f"window_features[{index}]")
        if feature.shape[1] != channels:
            raise ValueError("all window features must have the same channel dimension")
        if feature.device != device:
            raise ValueError("all window features must be on the same device")
        if feature.dtype != dtype:
            raise ValueError("all window features must have the same dtype")

    value_global = torch.cat(list(window_features), dim=0).contiguous()
    key_global = F.normalize(value_global, p=2, dim=1)
    return key_global, value_global


def key_value_extension(
    window_feature: torch.Tensor,
    key_global: torch.Tensor,
    value_global: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``Q_l`` and raw cosine similarities ``S_l`` for one window.

    This function deliberately applies no temperature, softmax, or residual.
    ``value_global`` is accepted and validated here so a mismatched key/value
    bank fails before attention is evaluated by a later stage.
    """
    _check_feature(window_feature, "window_feature")
    _check_feature(key_global, "key_global")
    _check_feature(value_global, "value_global")


    if key_global.shape != value_global.shape:
        raise ValueError("key_global and value_global must have identical shapes")
    if window_feature.shape[1] != key_global.shape[1]:
        raise ValueError("query and global key/value channel dimensions must match")
    if window_feature.device != key_global.device or value_global.device != key_global.device:
        raise ValueError("query, keys, and values must be on the same device")
    if window_feature.dtype != key_global.dtype or value_global.dtype != key_global.dtype:
        raise ValueError("query, keys, and values must have the same dtype")

    query = F.normalize(window_feature, p=2, dim=1)
    # Re-normalization makes the public function safe when called with a bank
    # not produced by build_global_key_value, while preserving the exact KVE
    # formula for a bank that already is normalized.
    keys = F.normalize(key_global, p=2, dim=1)  #TODO:检测这里的Key是不是重复正则化了？
    similarity = query @ keys.transpose(0, 1)
    return query, similarity


class KeyValueExtension(nn.Module):
    """Stateless ``nn.Module`` wrapper for :func:`key_value_extension`."""

    def forward(
        self,
        window_feature: torch.Tensor,
        key_global: torch.Tensor,
        value_global: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return key_value_extension(window_feature, key_global, value_global)


# Compatibility with the name used by the GLA-CLIP reference implementation.
KV_Extension = KeyValueExtension
k_v_extension = key_value_extension
