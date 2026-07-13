"""Window consistency, seam, and binary segmentation metrics."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .windows import Window, coordinate_keys


def _groups(coordinates: list[torch.Tensor], quantum: float = 1.0):
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for window_id, coords in enumerate(coordinates):
        keys = coordinate_keys(coords.detach().cpu(), quantum)
        for token_id, key in enumerate(keys.tolist()):
            groups.setdefault((key[0], key[1]), []).append((window_id, token_id))
    return [items for items in groups.values() if len(items) > 1]


def overlap_feature_consistency(
    features: list[torch.Tensor],
    coordinates: list[torch.Tensor],
    quantum: float = 1.0,
) -> dict[str, float]:
    groups = _groups(coordinates, quantum)
    flattened = [feature.flatten(1).T for feature in features]
    cosine = []
    for observations in groups:
        for (wa, ia), (wb, ib) in itertools.combinations(observations, 2):
            if wa == wb:
                continue
            cosine.append(float(F.cosine_similarity(
                flattened[wa][ia][None], flattened[wb][ib][None]
            ).item()))
    return {
        "overlap_pairs": len(cosine),
        "ofc": float(np.mean(cosine)) if cosine else math.nan,
    }


def overlap_metrics(
    features: list[torch.Tensor],
    similarity_scores: list[torch.Tensor],
    output_scores: list[torch.Tensor],
    candidates: list[torch.Tensor],
    forward_masks: list[torch.Tensor],
    backward_memberships: list[torch.Tensor],
    nn_indices: list[torch.Tensor],
    cluster_labels: list[torch.Tensor],
    coordinates: list[torch.Tensor],
    quantum: float = 1.0,
    decision_threshold: float = 0.2,
    ccd_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    groups = _groups(coordinates, quantum)
    flat_features = [feature.flatten(1).T for feature in features]
    flat_similarity = [score.reshape(-1) for score in similarity_scores]
    flat_output = [score.reshape(-1) for score in output_scores]
    flat_candidates = [mask.reshape(-1) for mask in candidates]
    flat_forward = [mask.reshape(-1) for mask in forward_masks]
    flat_membership = [membership.reshape(membership.shape[0], -1) for membership in backward_memberships]
    flat_nn = [index.reshape(index.shape[0], -1) for index in nn_indices]
    flat_clusters = [label.reshape(-1) for label in cluster_labels]
    cosine, similarity_abs, output_abs, output_binary = [], [], [], []
    candidate_flip, forward_flip, membership_flip, nn_flip = [], [], [], []
    rank_pairs: dict[tuple[int, int], tuple[list[float], list[float]]] = {}
    for observations in groups:
        for (wa, ia), (wb, ib) in itertools.combinations(observations, 2):
            if wa == wb:
                continue
            cosine.append(float(F.cosine_similarity(flat_features[wa][ia][None], flat_features[wb][ib][None]).item()))
            sim_a, sim_b = flat_similarity[wa][ia], flat_similarity[wb][ib]
            out_a, out_b = flat_output[wa][ia], flat_output[wb][ib]
            similarity_abs.append(float((sim_a - sim_b).abs().item()))
            output_abs.append(float((out_a - out_b).abs().item()))
            output_binary.append(float(((out_a > decision_threshold) != (out_b > decision_threshold)).item()))
            candidate_flip.append(float((flat_candidates[wa][ia] != flat_candidates[wb][ib]).item()))
            forward_flip.append(float((flat_forward[wa][ia] != flat_forward[wb][ib]).item()))
            membership_flip.append(float(
                (flat_membership[wa][:, ia] != flat_membership[wb][:, ib]).float().mean().item()
            ))
            nn_flip.append(float((flat_nn[wa][:, ia] != flat_nn[wb][:, ib]).float().mean().item()))
            key = (min(wa, wb), max(wa, wb))
            pair = rank_pairs.setdefault(key, ([], []))
            if wa < wb:
                pair[0].append(float(sim_a.item())); pair[1].append(float(sim_b.item()))
            else:
                pair[0].append(float(sim_b.item())); pair[1].append(float(sim_a.item()))

    rng = random.Random(seed)
    ccd_values = []
    pairable = [g for g in groups if len({w for w, _ in g}) >= 2]
    for _ in range(min(ccd_samples, len(pairable) * max(len(pairable) - 1, 0) // 2)):
        ga, gb = rng.sample(pairable, 2)
        common = sorted({w for w, _ in ga} & {w for w, _ in gb})
        if len(common) < 2:
            continue
        wa, wb = rng.sample(common, 2)
        ia = next(i for w, i in ga if w == wa); ja = next(i for w, i in gb if w == wa)
        ib = next(i for w, i in ga if w == wb); jb = next(i for w, i in gb if w == wb)
        ccd_values.append(float((flat_clusters[wa][ia] == flat_clusters[wa][ja]) != (flat_clusters[wb][ib] == flat_clusters[wb][jb])))

    def mean(values):
        return float(np.mean(values)) if values else math.nan

    def rankdata(values: list[float]) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        order = np.argsort(array, kind="mergesort")
        ranks = np.empty(len(array), dtype=np.float64)
        start = 0
        while start < len(array):
            stop = start + 1
            while stop < len(array) and array[order[stop]] == array[order[start]]:
                stop += 1
            ranks[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return ranks

    rank_correlations = []
    for left, right in rank_pairs.values():
        if len(left) < 2:
            continue
        ranks_left, ranks_right = rankdata(left), rankdata(right)
        if ranks_left.std() == 0 or ranks_right.std() == 0:
            continue
        rank_correlations.append(float(np.corrcoef(ranks_left, ranks_right)[0, 1]))
    return {
        "overlap_pairs": len(cosine),
        "ofc": mean(cosine),
        "cwsd": mean(similarity_abs),
        "similarity_rank_correlation": mean(rank_correlations),
        "cwod_score_mae": mean(output_abs),
        "cwod_binary": mean(output_binary),
        "forward_flip_rate": mean(forward_flip),
        "backward_membership_flip_rate": mean(membership_flip),
        "candidate_flip_rate": mean(candidate_flip),
        "backward_nn_change_rate": mean(nn_flip),
        "cluster_coassociation_disagreement": mean(ccd_values),
    }


def binary_metrics(prediction: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor | None = None) -> dict[str, float]:
    pred, gt = prediction.bool(), target.bool().to(prediction.device)
    if pred.shape != gt.shape:
        raise ValueError(f"prediction and target shapes differ: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    valid = torch.ones_like(gt) if ignore is None else ~ignore.bool().to(gt.device)
    tp = (pred & gt & valid).sum().item()
    fp = (pred & ~gt & valid).sum().item()
    fn = (~pred & gt & valid).sum().item()
    tn = (~pred & ~gt & valid).sum().item()
    eps = 1e-12
    return {
        "fg_iou": tp / (tp + fp + fn + eps),
        "bg_iou": tn / (tn + fp + fn + eps),
        "miou_binary": 0.5 * (tp / (tp + fp + fn + eps) + tn / (tn + fp + fn + eps)),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "foreground_recall": tp / (tp + fn + eps),
    }


def boundary_fscore(
    prediction: torch.Tensor,
    target: torch.Tensor,
    tolerance: int = 2,
    ignore: torch.Tensor | None = None,
) -> float:
    def boundary(mask):
        x = mask.float()[None, None]
        eroded = -F.max_pool2d(-x, 3, 1, 1)
        return (x > eroded)[0, 0]
    pb, gb = boundary(prediction), boundary(target.to(prediction.device))
    if ignore is not None:
        invalid = ignore.bool().to(prediction.device)
        invalid = F.max_pool2d(
            invalid.float()[None, None], tolerance * 2 + 3, 1, tolerance + 1
        )[0, 0] > 0
        pb &= ~invalid
        gb &= ~invalid
    pd = F.max_pool2d(pb.float()[None, None], tolerance * 2 + 1, 1, tolerance)[0, 0] > 0
    gd = F.max_pool2d(gb.float()[None, None], tolerance * 2 + 1, 1, tolerance)[0, 0] > 0
    precision = (pb & gd).sum().float() / pb.sum().clamp_min(1)
    recall = (gb & pd).sum().float() / gb.sum().clamp_min(1)
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-12)).item())


def seam_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    windows: list[Window],
    band: int = 3,
    ignore: torch.Tensor | None = None,
    max_boundary_distance: int = 32,
) -> dict[str, float]:
    pred, gt = prediction.bool(), target.bool().to(prediction.device)
    h, w = gt.shape
    valid = torch.ones_like(gt) if ignore is None else ~ignore.bool().to(gt.device)
    x_lines = sorted({x for window in windows for x in (window.x1, window.x2) if 0 < x < w})
    y_lines = sorted({y for window in windows for y in (window.y1, window.y2) if 0 < y < h})
    total = fg_total = bg_total = errors = fg_errors = bg_errors = 0
    for x in x_lines:
        same = (gt[:, x - 1] == gt[:, x]) & valid[:, x - 1] & valid[:, x]
        err = pred[:, x - 1] != pred[:, x]
        fg = same & gt[:, x]
        bg = same & ~gt[:, x]
        total += same.sum().item(); errors += (same & err).sum().item()
        fg_total += fg.sum().item(); fg_errors += (fg & err).sum().item()
        bg_total += bg.sum().item(); bg_errors += (bg & err).sum().item()
    for y in y_lines:
        same = (gt[y - 1] == gt[y]) & valid[y - 1] & valid[y]
        err = pred[y - 1] != pred[y]
        fg = same & gt[y]
        bg = same & ~gt[y]
        total += same.sum().item(); errors += (same & err).sum().item()
        fg_total += fg.sum().item(); fg_errors += (fg & err).sum().item()
        bg_total += bg.sum().item(); bg_errors += (bg & err).sum().item()
    seam_band = torch.zeros_like(gt)
    for x in x_lines:
        seam_band[:, max(0, x - band):min(w, x + band + 1)] = True
    for y in y_lines:
        seam_band[max(0, y - band):min(h, y + band + 1), :] = True
    seam_band &= valid
    control_band = (~seam_band) & valid
    pixel_error = (pred != gt).float()

    # Match the control error to the seam pixels' approximate distance from a
    # real GT boundary. This avoids treating difficult object boundaries as a
    # sliding-window artifact.
    x = gt.float()[None, None]
    gt_boundary = ((x > -F.max_pool2d(-x, 3, 1, 1))[0, 0]) & valid
    distance_bin = torch.full_like(gt, max_boundary_distance + 1, dtype=torch.long)
    reached = gt_boundary.clone()
    distance_bin[reached] = 0
    for distance in range(1, max_boundary_distance + 1):
        dilated = F.max_pool2d(reached.float()[None, None], 3, 1, 1)[0, 0] > 0
        ring = dilated & ~reached & valid
        distance_bin[ring] = distance
        reached = dilated
    seam_count = int(seam_band.sum().item())
    matched_control_sum = torch.tensor(0.0, device=gt.device)
    matched_weight = 0
    seam_distances = torch.unique(distance_bin[seam_band]).tolist() if seam_count else []
    for distance in seam_distances:
        seam_at_distance = seam_band & (distance_bin == distance)
        control_at_distance = control_band & (distance_bin == distance)
        count_at_distance = int(seam_at_distance.sum().item())
        if control_at_distance.any():
            matched_control_sum += pixel_error[control_at_distance].mean() * count_at_distance
            matched_weight += count_at_distance
    seam_error = pixel_error[seam_band].mean() if seam_count else torch.tensor(float("nan"), device=gt.device)
    if matched_weight:
        control_error = matched_control_sum / matched_weight
    elif control_band.any():
        control_error = pixel_error[control_band].mean()
    else:
        control_error = torch.tensor(float("nan"), device=gt.device)
    return {
        "ber": 100 * errors / max(total, 1),
        "ber_fg": 100 * fg_errors / max(fg_total, 1),
        "ber_bg": 100 * bg_errors / max(bg_total, 1),
        "seam_excess_error": float((seam_error - control_error).item()),
    }


def summarize_attention(diagnostics: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    result = {}
    for key in (
        "positive_count", "proxy_drift", "dn_u", "dn_w", "mask_ratio",
        "attention_entropy", "outer_positive_ratio", "outer_attention_mass",
    ):
        values = torch.cat([item[key].float().cpu() for item in diagnostics])
        result[f"attention_{key}_mean"] = float(values.mean().item())
    return result
