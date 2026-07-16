"""DINO-only Key-Value Extension, Proxy Anchor, and Dynamic Normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .windows import coordinate_keys


@dataclass(frozen=True)
class AlignerConfig:
    kve: bool = False
    proxy: bool = False
    dynamic_norm: bool = False
    proxy_rho: float = 0.6
    proxy_iters: int = 2
    dn_lambda1: float = 0.3
    dn_lambda2: float = 30.0
    fixed_beta: float = 1.2
    fixed_gamma: float = 3.0
    dn_cutoff: float = 0.0
    token_bank: str = "duplicate"
    coordinate_quantum: float = 1.0
    topk: int = 256
    query_chunk: int = 128
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.proxy_rho <= 1.0:
            raise ValueError("proxy_rho must be in [-1, 1]")
        if self.proxy_iters < 0:
            raise ValueError("proxy_iters must be non-negative")
        if self.fixed_gamma <= 0 or self.temperature <= 0:
            raise ValueError("fixed_gamma and temperature must be positive")
        if self.coordinate_quantum <= 0 or self.query_chunk <= 0 or self.topk <= 0:
            raise ValueError("coordinate_quantum, query_chunk, and topk must be positive")
        if self.token_bank not in {"duplicate", "deduplicated", "topk", "region"}:
            raise ValueError(f"Unsupported token bank: {self.token_bank}")


def _deduplicate_bank(
    features: torch.Tensor,
    coords: torch.Tensor,
    quantum: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = coordinate_keys(coords, quantum)
    _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    n = int(inverse.max().item()) + 1
    bank = torch.zeros((n, features.shape[1]), device=features.device, dtype=features.dtype)
    count = torch.zeros((n,), device=features.device, dtype=features.dtype)
    bank.index_add_(0, inverse, features)
    count.index_add_(0, inverse, torch.ones_like(inverse, dtype=features.dtype))
    bank = F.normalize(bank / count[:, None].clamp_min(1), dim=1)
    centers = torch.zeros((n, 2), device=coords.device, dtype=coords.dtype)
    centers.index_add_(0, inverse, coords)
    centers /= count[:, None].to(coords.dtype).clamp_min(1)
    return bank, centers, inverse


def _proxy_queries(
    q: torch.Tensor,
    bank: torch.Tensor,
    cfg: AlignerConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    proxy = q
    positive_count = torch.ones((q.shape[0],), device=q.device, dtype=q.dtype)
    drift = torch.zeros_like(positive_count)
    for _ in range(cfg.proxy_iters if cfg.proxy else 0):
        similarity = proxy @ bank.T
        positives = similarity > cfg.proxy_rho
        # The most similar token is a deterministic fallback and normally is self.
        empty = positives.sum(dim=1) == 0
        if empty.any():
            positives[empty, similarity[empty].argmax(dim=1)] = True
        positive_count = positives.sum(dim=1).to(q.dtype)
        updated = positives.to(bank.dtype) @ bank
        updated = F.normalize(updated / positive_count[:, None].clamp_min(1), dim=1)
        drift = 1 - (updated * q).sum(dim=1)
        proxy = updated
    positives = (proxy @ bank.T) > cfg.proxy_rho
    empty = positives.sum(dim=1) == 0
    if empty.any():
        similarity = proxy[empty] @ bank.T
        positives[empty, similarity.argmax(dim=1)] = True
    positive_count = positives.sum(dim=1).to(q.dtype)
    return proxy, positive_count, drift, positives


def _attend(
    q: torch.Tensor,
    bank: torch.Tensor,
    values: torch.Tensor,
    bank_is_inner: torch.Tensor,
    window_count: int,
    cfg: AlignerConfig,
):
    q_proxy, positive_count, drift, positives = _proxy_queries(q, bank, cfg)
    similarity = q_proxy @ bank.T
    raw_mean = similarity.mean(dim=1, keepdim=True)
    u = 1.0 + cfg.dn_lambda1 * torch.log1p(torch.tensor(float(window_count), device=q.device, dtype=q.dtype))
    w = 1.0 + cfg.dn_lambda2 / positive_count.clamp_min(1)
    if cfg.dynamic_norm:
        logits = w[:, None] * (similarity - u * raw_mean)
        norm_shift = torch.full_like(positive_count, float(u.item()))
        norm_scale = w
    else:
        logits = cfg.fixed_gamma * (similarity - cfg.fixed_beta * raw_mean)
        norm_shift = torch.full_like(positive_count, cfg.fixed_beta)
        norm_scale = torch.full_like(positive_count, cfg.fixed_gamma)
    logits = logits / max(cfg.temperature, 1e-6)
    keep = logits >= cfg.dn_cutoff
    # Match GLA's dynamic threshold fallback: at least the row maximum survives.
    keep.scatter_(1, logits.argmax(dim=1, keepdim=True), True)
    if cfg.token_bank == "topk" and bank.shape[0] > cfg.topk:
        top = logits.topk(cfg.topk, dim=1).indices
        top_mask = torch.zeros_like(keep)
        top_mask.scatter_(1, top, True)
        keep &= top_mask
    # DN is defined as threshold -> -inf -> softmax for every token-bank mode.
    # Previously the threshold was only applied in the top-k branch, so the
    # default duplicate/deduplicated banks merely reported a mask ratio without
    # actually masking attention.
    masked_logits = logits.masked_fill(~keep, -torch.inf)
    attention = torch.softmax(masked_logits.float(), dim=1).to(values.dtype)
    output = F.normalize(attention @ values, dim=1)
    entropy = -(attention.float() * attention.float().clamp_min(1e-12).log()).sum(dim=1)
    outer = ~bank_is_inner.bool()
    outer_positive = (positives & outer[None]).sum(dim=1).to(q.dtype)
    diagnostics = {
        "positive_count": positive_count.detach(),
        "proxy_drift": drift.detach(),
        "dn_u": norm_shift.detach(),
        "dn_w": norm_scale.detach(),
        "mask_ratio": (1 - keep.float().mean(dim=1)).detach(),
        "attention_entropy": entropy.detach(),
        "attention_top1_mass": attention.float().amax(dim=1).detach(),
        "attention_effective_tokens": entropy.exp().detach(),
        "feature_drift": (1 - (output * q).sum(dim=1)).detach(),
        "outer_positive_ratio": (outer_positive / positive_count.clamp_min(1)).detach(),
        "outer_attention_mass": attention[:, outer].sum(dim=1).detach(),
    }
    return output, diagnostics


@torch.no_grad()
def align_feature_windows(
    features: list[torch.Tensor],
    coordinates: list[torch.Tensor],
    config: AlignerConfig,
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Align CxHxW debiased target features while preserving window shapes."""
    if len(features) != len(coordinates) or not features:
        raise ValueError("features and coordinates must be non-empty and equally sized")
    flattened = [F.normalize(feature.flatten(1).T, dim=1) for feature in features]
    global_features = torch.cat(flattened, dim=0)
    global_coords = torch.cat(coordinates, dim=0).to(global_features.device)
    global_window_ids = torch.cat([
        torch.full((len(feature),), window_id, device=global_features.device, dtype=torch.long)
        for window_id, feature in enumerate(flattened)
    ])
    if config.token_bank == "region":
        raise NotImplementedError("region bank is reserved for the region-level extension; use duplicate/deduplicated/topk")
    if config.token_bank == "deduplicated":
        global_bank, _, bank_inverse = _deduplicate_bank(
            global_features, global_coords, config.coordinate_quantum
        )
    else:
        global_bank = global_features
        bank_inverse = None

    outputs: list[torch.Tensor] = []
    all_diagnostics: list[dict[str, torch.Tensor]] = []
    for window_id, (query, feature) in enumerate(zip(flattened, features)):
        bank = global_bank if config.kve else query
        values = bank
        if not config.kve:
            bank_is_inner = torch.ones(len(bank), device=query.device, dtype=torch.bool)
        elif bank_inverse is None:
            bank_is_inner = global_window_ids == window_id
        else:
            inner_count = torch.zeros(len(global_bank), device=query.device, dtype=torch.int32)
            inner_count.index_add_(
                0, bank_inverse,
                (global_window_ids == window_id).to(torch.int32),
            )
            bank_is_inner = inner_count > 0
        chunks, diagnostic_chunks = [], []
        for start in range(0, query.shape[0], config.query_chunk):
            stop = min(start + config.query_chunk, query.shape[0])
            out, diag = _attend(
                query[start:stop], bank, values, bank_is_inner,
                len(features) if config.kve else 1, config,
            )
            chunks.append(out)
            diagnostic_chunks.append(diag)
        aligned = torch.cat(chunks, dim=0).T.reshape_as(feature)
        diagnostics = {
            key: torch.cat([item[key] for item in diagnostic_chunks], dim=0)
            for key in diagnostic_chunks[0]
        }
        diagnostics["window_id"] = torch.tensor(window_id)
        outputs.append(aligned)
        all_diagnostics.append(diagnostics)
    return outputs, all_diagnostics


FACTORIAL = {
    "A0": (False, False, False),
    "A1": (True, False, False),
    "A2": (False, True, False),
    "A3": (False, False, True),
    "A4": (True, True, False),
    "A5": (True, False, True),
    "A6": (False, True, True),
    "A7": (True, True, True),
}


def factorial_config(method: str, base: AlignerConfig) -> AlignerConfig:
    if method not in FACTORIAL:
        raise KeyError(f"Unknown factorial method: {method}")
    kve, proxy, dynamic_norm = FACTORIAL[method]
    values = dict(base.__dict__)
    values.update(kve=kve, proxy=proxy, dynamic_norm=dynamic_norm)
    return AlignerConfig(**values)
