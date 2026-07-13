"""Window generation, token coordinates, feature fusion, and score stitching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Window:
    window_id: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return asdict(self)


def _starts(length: int, crop: int, stride: int) -> list[int]:
    if crop <= 0 or stride <= 0:
        raise ValueError("crop and stride must be positive")
    if stride > crop:
        raise ValueError("stride cannot exceed crop because it would leave coverage holes")
    if length <= crop:
        return [0]
    values = list(range(0, length - crop + 1, stride))
    edge = length - crop
    if values[-1] != edge:
        values.append(edge)
    return values


def make_windows(height: int, width: int, crop: int, stride: int) -> list[Window]:
    windows: list[Window] = []
    for y1 in _starts(height, crop, stride):
        for x1 in _starts(width, crop, stride):
            windows.append(Window(len(windows), x1, y1, min(x1 + crop, width), min(y1 + crop, height)))
    return windows


def coverage_map(windows: Iterable[Window], height: int, width: int) -> torch.Tensor:
    coverage = torch.zeros((height, width), dtype=torch.int16)
    for window in windows:
        coverage[window.y1:window.y2, window.x1:window.x2] += 1
    return coverage


def token_centers(window: Window, feat_h: int, feat_w: int, device=None) -> torch.Tensor:
    ys = window.y1 + (torch.arange(feat_h, device=device, dtype=torch.float32) + 0.5) * window.height / feat_h
    xs = window.x1 + (torch.arange(feat_w, device=device, dtype=torch.float32) + 0.5) * window.width / feat_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def edge_distances(window: Window, coords: torch.Tensor) -> torch.Tensor:
    x, y = coords[:, 0], coords[:, 1]
    return torch.stack((x - window.x1, window.x2 - x, y - window.y1, window.y2 - y), dim=1).amin(dim=1)


def coordinate_keys(coords: torch.Tensor, quantum: float = 1.0) -> torch.Tensor:
    if quantum <= 0:
        raise ValueError("coordinate quantum must be positive")
    return torch.round(coords / quantum).to(torch.int64)


def blend_weights(height: int, width: int, mode: str, device, dtype) -> torch.Tensor:
    if mode == "uniform":
        return torch.ones((height, width), device=device, dtype=dtype)
    if mode == "hann":
        wy = torch.hann_window(height + 2, periodic=False, device=device, dtype=dtype)[1:-1]
        wx = torch.hann_window(width + 2, periodic=False, device=device, dtype=dtype)[1:-1]
        return (wy[:, None] * wx[None, :]).clamp_min(1e-4)
    if mode == "center":
        yy = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        xx = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        return 1.0 - torch.maximum((yy[:, None] - 0.5).abs() * 2, (xx[None, :] - 0.5).abs() * 2)
    raise ValueError(f"Unknown stitch mode: {mode}")


def stitch_scores(
    scores: list[torch.Tensor],
    windows: list[Window],
    image_hw: tuple[int, int],
    mode: str = "uniform",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(scores) != len(windows) or not scores:
        raise ValueError("scores and windows must be non-empty and equally sized")
    height, width = image_hw
    device, dtype = scores[0].device, scores[0].dtype
    total = torch.zeros((height, width), device=device, dtype=dtype)
    weights = torch.zeros_like(total)
    winner = torch.full_like(total, -torch.inf) if mode == "center" else None
    variance_sum = torch.zeros_like(total)
    variance_sq = torch.zeros_like(total)
    count = torch.zeros_like(total)
    for score, window in zip(scores, windows):
        if score.ndim != 2:
            raise ValueError(f"Each score must be HxW, got shape {tuple(score.shape)}")
        resized = F.interpolate(score[None, None].float(), (window.height, window.width), mode="bilinear", align_corners=False)[0, 0].to(dtype)
        weight = blend_weights(window.height, window.width, mode, device, dtype)
        region = (slice(window.y1, window.y2), slice(window.x1, window.x2))
        if mode == "center":
            current = winner[region]
            take = weight > current
            total[region] = torch.where(take, resized, total[region])
            winner[region] = torch.maximum(current, weight)
            weights[region] = 1
        else:
            total[region] += resized * weight
            weights[region] += weight
        variance_sum[region] += resized
        variance_sq[region] += resized.square()
        count[region] += 1
    if (weights <= 0).any():
        raise RuntimeError("Window stitch left coverage holes")
    stitched = total if mode == "center" else total / weights
    variance = (variance_sq / count.clamp_min(1) - (variance_sum / count.clamp_min(1)).square()).clamp_min(0)
    return stitched, count, variance


def fuse_feature_windows(
    features: list[torch.Tensor],
    windows: list[Window],
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse aligned feature tokens onto a global token canvas for SW-Early."""
    if not features:
        raise ValueError("features cannot be empty")
    if len(features) != len(windows):
        raise ValueError("features and windows must be equally sized")
    c, fh, fw = features[0].shape
    if any(feature.shape != features[0].shape for feature in features):
        raise ValueError("All feature windows must share a CxHxW shape")
    step_y = windows[0].height / fh
    step_x = windows[0].width / fw
    gh = max(1, round(image_hw[0] / step_y))
    gw = max(1, round(image_hw[1] / step_x))
    total = torch.zeros((c, gh * gw), device=features[0].device, dtype=features[0].dtype)
    count = torch.zeros((gh * gw,), device=features[0].device, dtype=features[0].dtype)
    for feature, window in zip(features, windows):
        coords = token_centers(window, fh, fw, feature.device)
        gx = torch.round(coords[:, 0] / step_x - 0.5).long().clamp(0, gw - 1)
        gy = torch.round(coords[:, 1] / step_y - 0.5).long().clamp(0, gh - 1)
        index = gy * gw + gx
        total.index_add_(1, index, feature.reshape(c, -1))
        count.index_add_(0, index, torch.ones_like(index, dtype=count.dtype))
    if (count == 0).any():
        # Interpolate rare holes caused by non-integral physical token spacing.
        canvas = (total / count.clamp_min(1)).reshape(c, gh, gw)
        valid = (count > 0).reshape(1, 1, gh, gw).float()
        smooth = F.avg_pool2d(canvas[None], 3, 1, 1)[0]
        canvas = torch.where(valid[0].bool(), canvas, smooth)
    else:
        canvas = (total / count).reshape(c, gh, gw)
    return F.normalize(canvas, dim=0), count.reshape(gh, gw)
