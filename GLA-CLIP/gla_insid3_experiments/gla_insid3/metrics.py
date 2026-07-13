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


def overlap_metrics(
    features: list[torch.Tensor],
    scores: list[torch.Tensor],
    candidates: list[torch.Tensor],
    nn_indices: list[torch.Tensor],
    cluster_labels: list[torch.Tensor],
    coordinates: list[torch.Tensor],
    quantum: float = 1.0,
    ccd_samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    groups = _groups(coordinates, quantum)
    flat_features = [feature.flatten(1).T for feature in features]
    flat_scores = [score.reshape(-1) for score in scores]
    flat_candidates = [mask.reshape(-1) for mask in candidates]
    flat_nn = [index[0].reshape(-1) for index in nn_indices]
    flat_clusters = [label.reshape(-1) for label in cluster_labels]
    cosine, score_abs, score_binary, candidate_flip, nn_flip = [], [], [], [], []
    for observations in groups:
        for (wa, ia), (wb, ib) in itertools.combinations(observations, 2):
            cosine.append(float(F.cosine_similarity(flat_features[wa][ia][None], flat_features[wb][ib][None]).item()))
            a, b = flat_scores[wa][ia], flat_scores[wb][ib]
            score_abs.append(float((a - b).abs().item()))
            score_binary.append(float(((a > 0.2) != (b > 0.2)).item()))
            candidate_flip.append(float((flat_candidates[wa][ia] != flat_candidates[wb][ib]).item()))
            nn_flip.append(float((flat_nn[wa][ia] != flat_nn[wb][ib]).item()))

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
    return {
        "overlap_pairs": len(cosine),
        "ofc": mean(cosine),
        "cwsd": mean(score_abs),
        "cwod_binary": mean(score_binary),
        "candidate_flip_rate": mean(candidate_flip),
        "backward_nn_change_rate": mean(nn_flip),
        "cluster_coassociation_disagreement": mean(ccd_values),
    }


def binary_metrics(prediction: torch.Tensor, target: torch.Tensor, ignore: torch.Tensor | None = None) -> dict[str, float]:
    pred, gt = prediction.bool(), target.bool().to(prediction.device)
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


def boundary_fscore(prediction: torch.Tensor, target: torch.Tensor, tolerance: int = 2) -> float:
    def boundary(mask):
        x = mask.float()[None, None]
        eroded = -F.max_pool2d(-x, 3, 1, 1)
        return (x > eroded)[0, 0]
    pb, gb = boundary(prediction), boundary(target.to(prediction.device))
    pd = F.max_pool2d(pb.float()[None, None], tolerance * 2 + 1, 1, tolerance)[0, 0] > 0
    gd = F.max_pool2d(gb.float()[None, None], tolerance * 2 + 1, 1, tolerance)[0, 0] > 0
    precision = (pb & gd).sum().float() / pb.sum().clamp_min(1)
    recall = (gb & pd).sum().float() / gb.sum().clamp_min(1)
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-12)).item())


def seam_metrics(prediction: torch.Tensor, target: torch.Tensor, windows: list[Window], band: int = 3) -> dict[str, float]:
    pred, gt = prediction.bool(), target.bool().to(prediction.device)
    h, w = gt.shape
    x_lines = sorted({window.x1 for window in windows if 0 < window.x1 < w})
    y_lines = sorted({window.y1 for window in windows if 0 < window.y1 < h})
    total = fg_total = bg_total = errors = fg_errors = bg_errors = 0
    for x in x_lines:
        same = gt[:, x - 1] == gt[:, x]
        err = pred[:, x - 1] != pred[:, x]
        fg = same & gt[:, x]
        bg = same & ~gt[:, x]
        total += same.sum().item(); errors += (same & err).sum().item()
        fg_total += fg.sum().item(); fg_errors += (fg & err).sum().item()
        bg_total += bg.sum().item(); bg_errors += (bg & err).sum().item()
    for y in y_lines:
        same = gt[y - 1] == gt[y]
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
    pixel_error = pred != gt
    seam_error = pixel_error[seam_band].float().mean() if seam_band.any() else torch.tensor(float("nan"), device=gt.device)
    control_error = pixel_error[~seam_band].float().mean() if (~seam_band).any() else torch.tensor(float("nan"), device=gt.device)
    return {
        "ber": 100 * errors / max(total, 1),
        "ber_fg": 100 * fg_errors / max(fg_total, 1),
        "ber_bg": 100 * bg_errors / max(bg_total, 1),
        "seam_excess_error": float((seam_error - control_error).item()),
    }


def summarize_attention(diagnostics: list[dict[str, torch.Tensor]]) -> dict[str, float]:
    result = {}
    for key in ("positive_count", "proxy_drift", "dn_u", "dn_w", "mask_ratio", "attention_entropy"):
        values = torch.cat([item[key].float().cpu() for item in diagnostics])
        result[f"attention_{key}_mean"] = float(values.mean().item())
    return result
