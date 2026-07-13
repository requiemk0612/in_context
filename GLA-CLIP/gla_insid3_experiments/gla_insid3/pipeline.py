"""Four replayable interfaces around the unmodified INSID3 model."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .aligner import AlignerConfig, align_feature_windows
from .windows import Window, fuse_feature_windows, stitch_scores, token_centers


@dataclass
class ReferenceState:
    images: torch.Tensor
    masks: torch.Tensor
    raw_features: torch.Tensor
    debiased_features: torch.Tensor
    prototype: torch.Tensor


@dataclass
class WindowFeatures:
    windows: list[Window]
    crops: list[Image.Image]
    raw: list[torch.Tensor]
    debiased: list[torch.Tensor]
    coordinates: list[torch.Tensor]


def _external_data_utils():
    return importlib.import_module("utils.data")


def _external_clustering():
    return importlib.import_module("utils.clustering")


@torch.no_grad()
def prepare_reference(model, images: list[Image.Image], masks: list[torch.Tensor], device: str) -> ReferenceState:
    if len(images) != len(masks) or not images:
        raise ValueError("Reference images and masks must be non-empty and equally sized")
    image_tensors = torch.stack([model._transform(image) for image in images]).to(device)
    mask_tensors = torch.stack([mask.bool() for mask in masks]).to(device)
    fmaps = model._extract_features(image_tensors.unsqueeze(0))
    raw = F.normalize(fmaps, p=2, dim=2)
    debiased = model._debias_features(raw)
    _, shots, _, h, w = debiased.shape
    downsample_mask = _external_data_utils().downsample_mask
    prototypes = []
    for shot in range(shots):
        mask = F.interpolate(mask_tensors[shot][None, None].float(), (model.image_size, model.image_size), mode="nearest") > 0.5
        down = downsample_mask(mask, h, w)
        foreground = debiased[0, shot, :, down]
        if foreground.shape[1]:
            prototypes.append(foreground.mean(dim=1))
    if not prototypes:
        raise RuntimeError("Reference foreground vanished at feature resolution")
    prototype = F.normalize(torch.stack(prototypes).mean(dim=0), dim=0).unsqueeze(1)
    return ReferenceState(image_tensors, mask_tensors, raw, debiased, prototype)


@torch.no_grad()
def I1_extract_windows(model, target: Image.Image, windows: list[Window], device: str, batch_size: int = 2) -> WindowFeatures:
    crops = [target.crop((w.x1, w.y1, w.x2, w.y2)) for w in windows]
    raw_features: list[torch.Tensor] = []
    debiased_features: list[torch.Tensor] = []
    for start in range(0, len(crops), batch_size):
        tensors = torch.stack([model._transform(crop) for crop in crops[start:start + batch_size]]).to(device)
        raw_batch = F.normalize(model._extract_features(tensors.unsqueeze(0)), dim=2)
        deb_batch = model._debias_features(raw_batch)
        raw_features.extend([item for item in raw_batch[0]])
        debiased_features.extend([item for item in deb_batch[0]])
    coordinates = [
        token_centers(window, feature.shape[1], feature.shape[2], feature.device)
        for window, feature in zip(windows, raw_features)
    ]
    return WindowFeatures(windows, crops, raw_features, debiased_features, coordinates)


@torch.no_grad()
def I2_align_features(state: WindowFeatures, config: AlignerConfig):
    aligned, diagnostics = align_feature_windows(state.debiased, state.coordinates, config)
    return aligned, diagnostics


def _cluster(raw_feature: torch.Tensor, tau: float) -> torch.Tensor:
    c, h, w = raw_feature.shape
    flat = raw_feature.reshape(c, -1).T
    return _external_clustering().agglomerative_clustering(flat, tau).reshape(h, w)


@torch.no_grad()
def I3_reason_per_window(
    model,
    reference: ReferenceState,
    raw_feature: torch.Tensor,
    semantic_feature: torch.Tensor,
    cluster_labels: torch.Tensor | None = None,
    candidate_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Expose candidate, cluster, seed, and continuous aggregation states."""
    c, h, w = raw_feature.shape
    feat_tgt = raw_feature.unsqueeze(0)
    feat_sem = semantic_feature.unsqueeze(0)
    prototype = reference.prototype.to(device=semantic_feature.device, dtype=semantic_feature.dtype)
    sim_fwd = torch.einsum("bchw,cd->bhw", feat_sem, prototype).squeeze(0)
    forward_mask = sim_fwd > 0
    if forward_mask.sum() == 0:
        forward_mask = sim_fwd > torch.quantile(sim_fwd, 0.9)

    shots = reference.debiased_features.shape[1]
    votes = torch.zeros((h, w), dtype=torch.int32, device=semantic_feature.device)
    nn_indices = []
    backward_memberships = []
    downsample_mask = _external_data_utils().downsample_mask
    tgt_flat = semantic_feature.flatten(1).T
    for shot in range(shots):
        ref = reference.debiased_features[0, shot].to(semantic_feature.dtype)
        ref_flat = ref.flatten(1).T
        best_idx = (tgt_flat @ ref_flat.T).argmax(dim=1).reshape(h, w)
        rh, rw = ref.shape[1:]
        mask_input = F.interpolate(reference.masks[shot][None, None].float(), (model.image_size, model.image_size), mode="nearest") > 0.5
        ref_mask = downsample_mask(mask_input, rh, rw)
        member = ref_mask.reshape(-1)[best_idx]
        votes += member.to(torch.int32)
        nn_indices.append(best_idx)
        backward_memberships.append(member)
    backward_mask = votes >= math.ceil(shots / 2)
    candidate = forward_mask & backward_mask
    if candidate_override is not None:
        candidate = candidate_override.to(candidate.device).bool()

    if cluster_labels is None:
        cluster_labels = _cluster(raw_feature, model.tau)
    cluster_labels = cluster_labels.to(raw_feature.device)
    k_count = int(cluster_labels.max().item()) + 1
    compute_prototypes = _external_clustering().compute_cluster_prototypes
    semantic_flat = semantic_feature.reshape(c, -1).T
    semantic_prototypes = compute_prototypes(semantic_flat, cluster_labels.reshape(-1), k_count)
    raw_flat = raw_feature.reshape(c, -1).T
    raw_prototypes = compute_prototypes(raw_flat, cluster_labels.reshape(-1), k_count)

    matched = candidate & (cluster_labels >= 0)
    area_weights = torch.zeros(k_count, device=raw_feature.device, dtype=raw_feature.dtype)
    seed_id = -1
    if matched.any():
        matched_ids, matched_pixels = cluster_labels[matched].unique(return_counts=True)
        all_ids, all_pixels = cluster_labels[cluster_labels >= 0].unique(return_counts=True)
        area_by_id = torch.zeros(k_count, device=raw_feature.device, dtype=raw_feature.dtype)
        area_by_id[all_ids] = all_pixels.to(raw_feature.dtype)
        area_weights[matched_ids] = matched_pixels.to(raw_feature.dtype) / area_by_id[matched_ids].clamp_min(1)
        matched_cross = (prototype.T @ semantic_prototypes[matched_ids].T).squeeze(0)
        seed_id = int(matched_ids[matched_cross.argmax()].item())

    foreground_similarity = sim_fwd
    cross_similarity = torch.zeros(k_count, device=raw_feature.device, dtype=raw_feature.dtype)
    for cluster_id in range(k_count):
        pixels = cluster_labels == cluster_id
        if pixels.any():
            cross_similarity[cluster_id] = foreground_similarity[pixels].mean()
    if seed_id >= 0:
        intra_similarity = raw_prototypes @ raw_prototypes[seed_id]
        area_weights[seed_id] = 1
    else:
        intra_similarity = torch.zeros_like(cross_similarity)
    combined = cross_similarity * intra_similarity * area_weights
    continuous_score = torch.zeros((h, w), device=raw_feature.device, dtype=raw_feature.dtype)
    valid = cluster_labels >= 0
    continuous_score[valid] = combined[cluster_labels[valid]]
    hard_mask = continuous_score > model.merge_threshold
    return {
        "raw_feat": raw_feature,
        "debiased_feat": semantic_feature,
        "sim_fwd": sim_fwd,
        "forward_mask": forward_mask,
        "nn_ref_index": torch.stack(nn_indices),
        "backward_membership": torch.stack(backward_memberships),
        "backward_mask": backward_mask,
        "candidate_mask": candidate,
        "cluster_labels": cluster_labels,
        "cluster_prototypes": semantic_prototypes,
        "seed_id": seed_id,
        "cross_sim": cross_similarity,
        "intra_sim": intra_similarity,
        "area_weights": area_weights,
        "combined_score": combined,
        "continuous_score": continuous_score,
        "pre_crf_mask": hard_mask,
    }


def limit_early_resolution(raw: torch.Tensor, semantic: torch.Tensor, max_tokens: int):
    _, h, w = raw.shape
    if h * w <= max_tokens:
        return raw, semantic
    scale = math.sqrt(max_tokens / (h * w))
    size = (max(1, int(h * scale)), max(1, int(w * scale)))
    raw = F.normalize(F.interpolate(raw[None], size, mode="bilinear", align_corners=False)[0], dim=0)
    semantic = F.normalize(F.interpolate(semantic[None], size, mode="bilinear", align_corners=False)[0], dim=0)
    return raw, semantic


@torch.no_grad()
def run_early_reasoning(
    model,
    reference: ReferenceState,
    state: WindowFeatures,
    image_hw: tuple[int, int],
    max_tokens: int = 4096,
) -> dict[str, Any]:
    raw, raw_coverage = fuse_feature_windows(state.raw, state.windows, image_hw)
    semantic, _ = fuse_feature_windows(state.debiased, state.windows, image_hw)
    raw, semantic = limit_early_resolution(raw, semantic, max_tokens)
    result = I3_reason_per_window(model, reference, raw, semantic)
    result["early_feature_coverage"] = raw_coverage
    return result


@torch.no_grad()
def I4_stitch_and_refine(
    model,
    target: Image.Image,
    results: list[dict[str, Any]],
    windows: list[Window],
    stitch_mode: str,
    global_crf: bool = False,
    per_window_crf: bool = False,
) -> dict[str, torch.Tensor]:
    image_hw = (target.height, target.width)
    score_inputs = [result["continuous_score"] for result in results]
    if per_window_crf:
        if not hasattr(model, "_crf"):
            raise RuntimeError("Model must be constructed with mask_refiner='crf' for per-window CRF")
        refinement = importlib.import_module("utils.refinement")
        score_inputs = []
        for crop, result in zip([target.crop((w.x1, w.y1, w.x2, w.y2)) for w in windows], results):
            image_tensor = model._transform(crop).unsqueeze(0).to(result["continuous_score"].device)
            image_tensor = F.interpolate(image_tensor, model._crf_size, mode="bilinear", align_corners=False)
            initial = F.interpolate(result["pre_crf_mask"][None, None].float(), model._crf_size, mode="nearest")[0, 0] > 0.5
            refined = refinement.crf_refine(model._crf, model._crf_band_px, model._crf_p_core, image_tensor, initial)
            score_inputs.append(refined.float())
    score, coverage, variance = stitch_scores(
        score_inputs, windows, image_hw, stitch_mode
    )
    pre_crf = score > model.merge_threshold
    post_crf = pre_crf
    if global_crf:
        if not hasattr(model, "_crf"):
            raise RuntimeError("Model must be constructed with mask_refiner='crf' for global CRF")
        refinement = importlib.import_module("utils.refinement")
        crf_hw = model._crf_size
        image_tensor = model._transform(target).unsqueeze(0).to(score.device)
        image_tensor = F.interpolate(image_tensor, crf_hw, mode="bilinear", align_corners=False)
        mask_crf = F.interpolate(pre_crf[None, None].float(), crf_hw, mode="nearest")[0, 0] > 0.5
        refined = refinement.crf_refine(model._crf, model._crf_band_px, model._crf_p_core, image_tensor, mask_crf)
        post_crf = F.interpolate(refined[None, None].float(), image_hw, mode="nearest")[0, 0] > 0.5
    return {
        "stitched_score": score,
        "coverage": coverage,
        "score_variance": variance,
        "pre_crf_mask": pre_crf,
        "post_crf_mask": post_crf,
    }


def canonicalize_tensor_observations(
    tensors: list[torch.Tensor],
    coordinates: list[torch.Tensor],
    quantum: float = 1.0,
) -> list[torch.Tensor]:
    """Replace repeated coordinate observations by their mean (D1/D2/D5 replay)."""
    shapes = [tensor.shape for tensor in tensors]
    flattened = []
    for tensor in tensors:
        flattened.append(tensor.flatten(1).T if tensor.ndim == 3 else tensor.reshape(-1, 1))
    values = torch.cat(flattened, dim=0)
    coords = torch.cat(coordinates, dim=0).to(values.device)
    keys = torch.round(coords / quantum).long()
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    count = torch.zeros((int(inverse.max()) + 1,), device=values.device, dtype=values.dtype)
    sums = torch.zeros((len(count), values.shape[1]), device=values.device, dtype=values.dtype)
    sums.index_add_(0, inverse, values)
    count.index_add_(0, inverse, torch.ones_like(inverse, dtype=values.dtype))
    canonical = sums / count[:, None].clamp_min(1)
    replaced = canonical[inverse]
    outputs, start = [], 0
    for shape, flat in zip(shapes, flattened):
        item = replaced[start:start + len(flat)]
        start += len(flat)
        if len(shape) == 3:
            outputs.append(F.normalize(item.T.reshape(shape), dim=0))
        else:
            outputs.append(item.reshape(shape))
    return outputs


def canonicalize_binary_observations(masks: list[torch.Tensor], coordinates: list[torch.Tensor], quantum: float = 1.0):
    means = canonicalize_tensor_observations([mask.float() for mask in masks], coordinates, quantum)
    return [item >= 0.5 for item in means]


def canonicalize_cluster_observations(
    raw_features: list[torch.Tensor],
    coordinates: list[torch.Tensor],
    tau: float,
    quantum: float = 1.0,
    max_tokens: int = 16000,
) -> list[torch.Tensor]:
    """D4 replay: cluster canonical coordinate-mean features once, then map labels back."""
    flattened = [feature.flatten(1).T for feature in raw_features]
    features = torch.cat(flattened, dim=0)
    coords = torch.cat(coordinates, dim=0).to(features.device)
    keys = torch.round(coords / quantum).long()
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    n = int(inverse.max().item()) + 1
    if n > max_tokens:
        raise RuntimeError(f"D4 canonical clustering has {n} tokens, above --d4-max-tokens={max_tokens}")
    sums = torch.zeros((n, features.shape[1]), device=features.device, dtype=features.dtype)
    count = torch.zeros((n,), device=features.device, dtype=features.dtype)
    sums.index_add_(0, inverse, features)
    count.index_add_(0, inverse, torch.ones_like(inverse, dtype=features.dtype))
    canonical_features = F.normalize(sums / count[:, None].clamp_min(1), dim=1)
    labels = _external_clustering().agglomerative_clustering(canonical_features, tau)
    mapped = labels[inverse]
    output, start = [], 0
    for feature in raw_features:
        length = feature.shape[1] * feature.shape[2]
        output.append(mapped[start:start + length].reshape(feature.shape[1:]))
        start += length
    return output
