"""Sliding-window extraction, feature-grid restoration, and mask fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from utils.clustering import agglomerative_clustering, compute_cluster_prototypes
from utils.data import downsample_mask
from utils.refinement import upsample_mask


@dataclass(frozen=True)
class SlidingWindow:
    """A row-major window in the transformed target-image canvas."""

    index: int
    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def width(self) -> int:
        return self.right - self.left


@dataclass
class WindowPatchFeatures:
    """Final patch features and spatial metadata for one target window."""

    window: SlidingWindow
    image: torch.Tensor
    flat_features: torch.Tensor
    grid_size: tuple[int, int]
    original_box: tuple[float, float, float, float]
    original_token_coords: torch.Tensor


@dataclass
class ReferenceFeatureContext:
    """Reference state shared by every target window in an episode."""

    debiased_features: torch.Tensor
    masks_for_matching: torch.Tensor
    prototype: torch.Tensor
    channels: int


def _window_starts(length: int, crop_size: int, stride: int) -> list[int]:
    if length <= 0:
        raise ValueError("image dimensions must be positive")
    if crop_size <= 0 or stride <= 0:
        raise ValueError("crop_size and stride must be positive")
    if stride > crop_size:
        raise ValueError("stride must not exceed crop_size because it leaves coverage holes")
    if length <= crop_size:
        return [0]

    last = length - crop_size
    starts = list(range(0, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_sliding_windows(
    image_size: tuple[int, int],
    crop_size: int,
    stride: int,
) -> list[SlidingWindow]:
    """Generate row-major overlapping windows with edge-aligned final crops."""
    height, width = image_size
    windows: list[SlidingWindow] = []
    for top in _window_starts(height, crop_size, stride):
        bottom = min(top + crop_size, height)
        for left in _window_starts(width, crop_size, stride):
            right = min(left + crop_size, width)
            windows.append(SlidingWindow(
                index=len(windows),
                top=top,
                left=left,
                bottom=bottom,
                right=right,
            ))
    return windows


def _original_coordinates(
    window: SlidingWindow,
    grid_size: tuple[int, int],
    canvas_size: tuple[int, int],
    original_size: tuple[int, int],
    device: torch.device,
) -> tuple[tuple[float, float, float, float], torch.Tensor]:
    grid_h, grid_w = grid_size
    canvas_h, canvas_w = canvas_size
    original_h, original_w = original_size
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError("patch-grid dimensions must be positive")
    if original_h <= 0 or original_w <= 0:
        raise ValueError("original-image dimensions must be positive")

    ys = window.top + (
        torch.arange(grid_h, device=device, dtype=torch.float32) + 0.5
    ) * window.height / grid_h
    xs = window.left + (
        torch.arange(grid_w, device=device, dtype=torch.float32) + 0.5
    ) * window.width / grid_w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    token_coords = torch.stack((yy, xx), dim=-1).reshape(-1, 2)
    token_coords[:, 0] *= original_h / canvas_h
    token_coords[:, 1] *= original_w / canvas_w

    original_box = (
        window.top * original_h / canvas_h,
        window.left * original_w / canvas_w,
        window.bottom * original_h / canvas_h,
        window.right * original_w / canvas_w,
    )
    return original_box, token_coords


@torch.no_grad()
def extract_sliding_window_features(
    model: torch.nn.Module,
    target_canvas: torch.Tensor,
    windows: list[SlidingWindow],
    original_size: tuple[int, int],
    batch_size: int = 4,
    expected_channels: int | None = None,
) -> list[WindowPatchFeatures]:
    """Batch-extract final VFM patch features while preserving window order."""
    if target_canvas.ndim != 4 or target_canvas.shape[0] != 1:
        raise ValueError("target_canvas must have shape (1, C, H, W)")
    if not windows:
        raise ValueError("windows must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    canvas_size = tuple(target_canvas.shape[-2:])
    records: list[WindowPatchFeatures] = []
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start:start + batch_size]
        crops = [
            target_canvas[:, :, window.top:window.bottom, window.left:window.right]
            for window in batch_windows
        ]
        if len({tuple(crop.shape[-2:]) for crop in crops}) != 1:
            raise ValueError("windows in one extraction batch must have equal shapes")

        crop_batch = torch.cat(crops, dim=0)
        feature_batch = model._extract_features(crop_batch.unsqueeze(0))[0]
        if feature_batch.shape[0] != len(batch_windows):
            raise ValueError("VFM returned an unexpected number of window features")

        for window, crop, feature in zip(batch_windows, crops, feature_batch):
            if feature.ndim != 3 or not feature.is_floating_point():
                raise ValueError("each VFM window feature must be a floating CxHxW tensor")
            if feature.device != target_canvas.device:
                raise ValueError("window features and target_canvas must share a device")
            channels, grid_h, grid_w = feature.shape
            if expected_channels is not None and channels != expected_channels:
                raise ValueError("target and reference feature channel dimensions do not match")

            flat_features = feature.permute(1, 2, 0).reshape(-1, channels).contiguous()
            original_box, token_coords = _original_coordinates(
                window, (grid_h, grid_w), canvas_size, original_size, feature.device
            )
            records.append(WindowPatchFeatures(
                window=window,
                image=crop,
                flat_features=flat_features,
                grid_size=(grid_h, grid_w),
                original_box=original_box,
                original_token_coords=token_coords,
            ))

    if [record.window.index for record in records] != list(range(len(windows))):
        raise RuntimeError("window feature order was not preserved")
    first = records[0].flat_features
    if any(
        record.flat_features.device != first.device
        or record.flat_features.dtype != first.dtype
        for record in records
    ):
        raise ValueError("all window features must share device and dtype")
    return records


@torch.no_grad()
def prepare_reference_context(model: torch.nn.Module) -> ReferenceFeatureContext:
    """Prepare references exactly as INSID3.predict_mask does before matching."""
    if model._ref_images is None or model._ref_masks is None:
        raise RuntimeError("reference images and masks must be initialized")

    features = model._extract_features(model._ref_images.unsqueeze(0))
    normalized = F.normalize(features, p=2, dim=2)
    debiased = model._debias_features(normalized)
    _, shots, channels, feat_h, feat_w = debiased.shape
    masks_for_matching = model._ref_masks.unsqueeze(1)

    prototypes = []
    for shot in range(shots):
        # Use the official helper, including its bilinear/nearest/center
        # fallback behavior, rather than implementing mask reduction here.
        mask_shot = downsample_mask(
            masks_for_matching[shot:shot + 1], feat_h, feat_w
        )
        foreground = debiased[0, shot, :, mask_shot]
        if foreground.shape[1] > 0:
            prototypes.append(foreground.mean(dim=1))
    prototype = F.normalize(
        torch.stack(prototypes).mean(dim=0), p=2, dim=0
    ).unsqueeze(1)
    return ReferenceFeatureContext(
        debiased_features=debiased,
        masks_for_matching=masks_for_matching,
        prototype=prototype,
        channels=channels,
    )


def restore_patch_grid(
    flat_features: torch.Tensor,
    grid_size: tuple[int, int],
) -> torch.Tensor:
    """Restore an ordered ``(tokens, channels)`` tensor to ``(C, H, W)``."""
    if flat_features.ndim != 2:
        raise ValueError("flat_features must have shape (tokens, channels)")
    grid_h, grid_w = grid_size
    if flat_features.shape[0] != grid_h * grid_w:
        raise ValueError("flat feature count does not match the patch grid")
    return flat_features.reshape(
        grid_h, grid_w, flat_features.shape[1]
    ).permute(2, 0, 1).contiguous()


@torch.no_grad()
def predict_window_mask_from_features(
    model: torch.nn.Module,
    raw_features: torch.Tensor,
    target_window: torch.Tensor,
    reference: ReferenceFeatureContext,
) -> torch.Tensor:
    """Apply INSID3's official post-feature stages to one window feature grid.

    INSID3 has no public entry point for a precomputed target feature whose
    grid differs from the reference grid. This adapter only supplies that
    missing glue and delegates mask reduction, candidate localization,
    clustering, aggregation, and final refinement to the project primitives.
    """
    if raw_features.ndim != 3:
        raise ValueError("raw_features must have shape (C, H, W)")
    if target_window.ndim != 4 or target_window.shape[0] != 1:
        raise ValueError("target_window must have shape (1, C, H, W)")
    channels, feat_h, feat_w = raw_features.shape
    if channels != reference.channels or feat_h <= 0 or feat_w <= 0:
        raise ValueError("target and reference feature dimensions are incompatible")

    target = F.normalize(raw_features.unsqueeze(0), p=2, dim=1)
    target_debiased = model._debias_features(target.unsqueeze(1))[:, 0]
    similarity_maps = [
        torch.einsum(
            "bchw,bcxy->bhwxy",
            reference.debiased_features[:, shot],
            target_debiased,
        )
        for shot in range(reference.debiased_features.shape[1])
    ]
    candidates = model._locate_candidates(
        similarity_maps,
        reference.masks_for_matching,
        target_debiased,
        reference.prototype,
        feat_h,
        feat_w,
    )

    saved_original_size = model._orig_tgt_size
    model._orig_tgt_size = tuple(target_window.shape[-2:])
    try:
        if candidates.sum() == 0:
            return model._finalize_mask(candidates, target_window)

        target_flat = target[0].reshape(channels, -1).permute(1, 0)
        labels = agglomerative_clustering(
            target_flat, model.tau
        ).reshape(feat_h, feat_w)
        cluster_count = int(labels.max().item()) + 1
        target_debiased_flat = target_debiased[0].reshape(
            channels, -1
        ).permute(1, 0)
        prototypes = compute_cluster_prototypes(
            target_debiased_flat, labels.reshape(-1), cluster_count
        )
        prediction = model._seed_and_aggregate(
            candidates,
            labels,
            prototypes,
            cluster_count,
            reference.prototype,
            target,
            target_debiased,
            feat_h,
            feat_w,
        )
        return model._finalize_mask(prediction, target_window)
    finally:
        model._orig_tgt_size = saved_original_size


def fuse_window_predictions(
    predictions: list[torch.Tensor],
    windows: list[SlidingWindow],
    canvas_size: tuple[int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Average overlapping window predictions by pixel position."""
    if not predictions or len(predictions) != len(windows):
        raise ValueError("predictions and windows must be non-empty and equally sized")
    canvas_h, canvas_w = canvas_size
    first = predictions[0]
    total = torch.zeros((canvas_h, canvas_w), device=first.device, dtype=torch.float32)
    count = torch.zeros_like(total)

    for prediction, window in zip(predictions, windows):
        if prediction.ndim != 2:
            raise ValueError("each window prediction must be two-dimensional")
        if prediction.device != first.device:
            raise ValueError("all window predictions must share a device")
        if prediction.shape != (window.height, window.width):
            prediction = F.interpolate(
                prediction[None, None].float(),
                size=(window.height, window.width),
                mode="nearest",
            )[0, 0]
        region = (slice(window.top, window.bottom), slice(window.left, window.right))
        total[region] += prediction.float()
        count[region] += 1

    if (count == 0).any():
        raise RuntimeError("sliding-window fusion left uncovered target pixels")
    fused = total / count > 0.5
    if tuple(fused.shape) == tuple(original_size):
        return fused
    # Match INSID3's official mask resizing behavior.
    return upsample_mask(fused, original_size[0], original_size[1])


@torch.no_grad()
def segment_feature_windows(
    model: torch.nn.Module,
    records: list[WindowPatchFeatures],
    enhanced_features: list[torch.Tensor],
    reference: ReferenceFeatureContext,
    canvas_size: tuple[int, int],
    original_size: tuple[int, int],
) -> torch.Tensor:
    """Restore, segment, and fuse an ordered set of enhanced window features."""
    if not records or len(records) != len(enhanced_features):
        raise ValueError("records and enhanced_features must be non-empty and equally sized")

    predictions = []
    for record, flat_features in zip(records, enhanced_features):
        feature_grid = restore_patch_grid(flat_features, record.grid_size)
        predictions.append(predict_window_mask_from_features(
            model, feature_grid, record.image, reference
        ))
    return fuse_window_predictions(
        predictions,
        [record.window for record in records],
        canvas_size,
        original_size,
    )
