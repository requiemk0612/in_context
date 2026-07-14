#!/usr/bin/env python3
"""Run fixed-manifest GLA-INSID3 baselines, factorials, and checkpoint replay."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gla_insid3.aligner import AlignerConfig, FACTORIAL, factorial_config
from gla_insid3.bootstrap import build_model, implementation_manifest, write_json
from gla_insid3.data import ISAIDStore, generate_manifest, load_manifest
from gla_insid3.metrics import (
    binary_metrics,
    boundary_fscore,
    overlap_feature_consistency,
    overlap_metrics,
    seam_metrics,
    summarize_attention,
)
from gla_insid3.pipeline import (
    I1_extract_windows,
    I2_align_features,
    I3_reason_per_window,
    I4_stitch_and_refine,
    canonicalize_binary_observations,
    canonicalize_cluster_observations,
    canonicalize_tensor_observations,
    prepare_reference,
    run_early_reasoning,
)
from gla_insid3.windows import Window, make_windows


DEFAULT_INSID3 = "/data2/cld/in_context/INSID3-main"
DEFAULT_DATA = "/data/lky/data/rs_seg"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
GLA_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_OUTPUT = str(EXPERIMENT_ROOT / "outputs" / "mvp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", choices=("manifest", "run"), default="run")
    parser.add_argument("--insid3-root", default=DEFAULT_INSID3)
    parser.add_argument("--data-root", default=DEFAULT_DATA)
    parser.add_argument("--manifest", required=True, help="JSONL episode manifest (created here, never under the dataset)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument(
        "--episode-limit", type=int, default=0,
        help="Run only the first N manifest episodes; 0 runs all (use 1 for a real smoke test)",
    )
    parser.add_argument("--window-crop", type=int, default=512)
    parser.add_argument("--window-stride", type=int, default=256)
    parser.add_argument("--include-single-window-targets", action="store_true")
    parser.add_argument("--methods", default="B0,B1,B2,B3,A1,A2,A3,A7")
    parser.add_argument("--replays", default="D1,D3,D4,D5", help="B1 replay interventions; empty disables")
    parser.add_argument("--model-size", choices=("small", "base", "large"), default="large")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--svd-comps", type=int, default=500)
    parser.add_argument(
        "--min-reference-tokens", type=int, default=10,
        help="Require this many foreground tokens per reference on the DINO feature grid",
    )
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--merge-thresh", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-batch-size", type=int, default=2)
    parser.add_argument("--early-max-tokens", type=int, default=4096)
    parser.add_argument("--d4-max-tokens", type=int, default=16000)
    parser.add_argument("--token-bank", choices=("duplicate", "deduplicated", "topk"), default="duplicate")
    parser.add_argument("--coordinate-quantum", type=float, default=1.0)
    parser.add_argument("--proxy-rho", type=float, default=0.6)
    parser.add_argument("--proxy-iters", type=int, default=2)
    parser.add_argument("--dn-lambda1", type=float, default=0.3)
    parser.add_argument("--dn-lambda2", type=float, default=30.0)
    parser.add_argument("--fixed-beta", type=float, default=1.2)
    parser.add_argument("--fixed-gamma", type=float, default=3.0)
    parser.add_argument("--dn-cutoff", type=float, default=0.0)
    parser.add_argument("--attention-temperature", type=float, default=1.0)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--enable-crf", action="store_true", help="Required for B4/B5; CRF is applied after stitch for B4")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--skip-duplicate-control", action="store_true")
    parser.add_argument("--duplicate-tolerance", type=float, default=1e-5)
    parser.add_argument("--resume", action="store_true", help="Skip method/episode records already in metrics.jsonl")
    return parser.parse_args()


def artifact_path(value: str | Path) -> Path:
    """Resolve every writable artifact under GLA-CLIP, never under the dataset."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = EXPERIMENT_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(GLA_ROOT)
    except ValueError as exc:
        raise ValueError(f"Artifact path must remain under {GLA_ROOT}: {path}") from exc
    return path


def set_determinism(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _reason_windows(model, reference, raw, semantic, clusters=None, candidates=None):
    results = []
    for index, (raw_feature, semantic_feature) in enumerate(zip(raw, semantic)):
        results.append(I3_reason_per_window(
            model, reference, raw_feature, semantic_feature,
            None if clusters is None else clusters[index],
            None if candidates is None else candidates[index],
        ))
    return results


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def _checkpoint_payload(results, aligned=None, diagnostics=None, include_features: bool = True):
    keys = (
        "raw_feat", "debiased_feat", "sim_fwd", "forward_mask", "nn_ref_index",
        "backward_membership", "backward_mask", "candidate_mask", "cluster_labels",
        "cluster_prototypes", "seed_id", "cross_sim", "intra_sim", "area_weights",
        "combined_score", "continuous_score", "pre_crf_mask", "early_feature_coverage",
        "early_fused_feature_hw", "early_reasoning_feature_hw", "early_max_tokens",
        "early_was_resized",
    )
    if not include_features:
        keys = tuple(key for key in keys if key not in {"raw_feat", "debiased_feat"})
    payload = [{key: result[key] for key in keys if key in result} for result in results]
    return _to_cpu({"windows": payload, "aligned": aligned, "attention": diagnostics})


def _resolution_metadata(method: str, model, metric_state, results, source_state) -> dict[str, Any]:
    reasoning_shapes = [tuple(result["continuous_score"].shape) for result in results]
    metadata: dict[str, Any] = {
        "method": method,
        "encoder_input_hw": [int(model.image_size), int(model.image_size)],
        "encoder_patch_size": 16,
        "source_num_windows": len(source_state.windows),
        "source_window_feature_hw": list(source_state.raw[0].shape[-2:]),
        "reasoning_feature_hw": [list(shape) for shape in reasoning_shapes],
        "reasoning_tokens_per_map": [int(height * width) for height, width in reasoning_shapes],
    }
    if method == "B3":
        metadata.update({
            "early_fused_feature_hw": list(results[0]["early_fused_feature_hw"]),
            "early_max_tokens": int(results[0]["early_max_tokens"]),
            "early_was_resized": bool(results[0]["early_was_resized"]),
        })
    metadata["method_num_windows"] = len(metric_state.windows) if method != "B3" else 1
    return metadata


@torch.no_grad()
def duplicate_control(model, target, window: Window, device: str) -> dict[str, float]:
    duplicate_windows = [
        Window(0, window.x1, window.y1, window.x2, window.y2),
        Window(1, window.x1, window.y1, window.x2, window.y2),
    ]
    state = I1_extract_windows(model, target, duplicate_windows, device, batch_size=2)
    return {
        "raw_max_abs": float((state.raw[0] - state.raw[1]).abs().max().item()),
        "debiased_max_abs": float((state.debiased[0] - state.debiased[1]).abs().max().item()),
    }


def _method_result(method: str, model, reference, target, state, args, base_results):
    attention = None
    semantics = state.debiased
    raw = state.raw
    clusters = [result["cluster_labels"] for result in base_results]
    stitch_mode, global_crf, per_window_crf = "uniform", False, False

    if method == "B0":
        full_window = [Window(0, 0, 0, target.width, target.height)]
        full_state = I1_extract_windows(model, target, full_window, args.device, 1)
        results = _reason_windows(model, reference, full_state.raw, full_state.debiased)
        stitched = I4_stitch_and_refine(model, target, results, full_window, "uniform")
        return results, full_state, stitched, attention
    if method == "B2":
        stitch_mode = "hann"
    elif method == "B3":
        result = run_early_reasoning(model, reference, state, (target.height, target.width), args.early_max_tokens)
        full_window = [Window(0, 0, 0, target.width, target.height)]
        stitched = I4_stitch_and_refine(model, target, [result], full_window, "uniform")
        return [result], state, stitched, attention
    elif method == "B4":
        if not args.enable_crf:
            raise ValueError("B4 requires --enable-crf")
        global_crf = True
    elif method == "B5":
        if not args.enable_crf:
            raise ValueError("B5 requires --enable-crf")
        per_window_crf = True
    elif method in FACTORIAL:
        base_cfg = AlignerConfig(
            proxy_rho=args.proxy_rho, proxy_iters=args.proxy_iters,
            dn_lambda1=args.dn_lambda1, dn_lambda2=args.dn_lambda2,
            fixed_beta=args.fixed_beta, fixed_gamma=args.fixed_gamma,
            dn_cutoff=args.dn_cutoff, token_bank=args.token_bank,
            coordinate_quantum=args.coordinate_quantum, topk=args.topk,
            query_chunk=args.query_chunk, temperature=args.attention_temperature,
        )
        semantics, attention = I2_align_features(state, factorial_config(method, base_cfg))
    elif method.startswith("R-"):
        stage = method[2:]
        if stage == "D1":
            raw = canonicalize_tensor_observations(state.raw, state.coordinates, args.coordinate_quantum)
            stacked = torch.stack(raw).unsqueeze(0)
            semantics = [item for item in model._debias_features(stacked)[0]]
            results = _reason_windows(model, reference, raw, semantics)
        elif stage == "D2":
            semantics = canonicalize_tensor_observations(state.debiased, state.coordinates, args.coordinate_quantum)
            results = _reason_windows(model, reference, raw, semantics, clusters)
        elif stage == "D3":
            candidates = canonicalize_binary_observations(
                [item["candidate_mask"] for item in base_results], state.coordinates, args.coordinate_quantum
            )
            results = _reason_windows(model, reference, raw, semantics, clusters, candidates)
        elif stage == "D4":
            canonical_clusters = canonicalize_cluster_observations(
                raw, state.coordinates, model.tau, args.coordinate_quantum, args.d4_max_tokens
            )
            results = _reason_windows(model, reference, raw, semantics, canonical_clusters)
        elif stage == "D5":
            canonical_scores = canonicalize_tensor_observations(
                [item["continuous_score"] for item in base_results], state.coordinates, args.coordinate_quantum
            )
            results = [dict(item, continuous_score=score, pre_crf_mask=score > model.merge_threshold)
                       for item, score in zip(base_results, canonical_scores)]
        else:
            raise KeyError(f"Unsupported replay stage: {stage}")
        stitched = I4_stitch_and_refine(model, target, results, state.windows, "uniform")
        return results, state, stitched, attention
    elif method != "B1":
        raise KeyError(f"Unknown method: {method}")

    results = _reason_windows(model, reference, raw, semantics, clusters)
    stitched = I4_stitch_and_refine(
        model, target, results, state.windows, stitch_mode,
        global_crf=global_crf, per_window_crf=per_window_crf,
    )
    return results, state, stitched, attention


def _evaluate_method(
    method, episode, model, reference, target, target_mask, target_ignore,
    state, args, base_results,
):
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results, metric_state, stitched, attention = _method_result(
        method, model, reference, target, state, args, base_results
    )
    elapsed = time.perf_counter() - started
    prediction = stitched["post_crf_mask"]
    pre_crf_prediction = stitched["pre_crf_mask"]
    source_state = metric_state if method == "B0" else state
    record: dict[str, Any] = {
        "episode_id": episode.episode_id,
        "fold": episode.fold,
        "class_id": episode.class_id,
        "method": method,
        "elapsed_seconds": elapsed,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available() and args.device.startswith("cuda") else 0.0
        ),
        "num_windows": len(metric_state.windows) if method != "B3" else len(state.windows),
        "coverage_min": float(stitched["coverage"].min().item()),
        "coverage_max": float(stitched["coverage"].max().item()),
        "score_variance_mean": float(stitched["score_variance"].mean().item()),
        "target_foreground_fraction": episode.target_foreground_fraction,
        "target_windows_with_foreground": episode.target_windows_with_foreground,
        "reference_foreground_tokens": reference.foreground_token_counts,
        "min_reference_tokens": args.min_reference_tokens,
        "encoder_input_hw": [int(model.image_size), int(model.image_size)],
        "encoder_patch_size": 16,
        "source_num_windows": len(source_state.windows),
        "source_window_pixel_hw": [
            [window.height, window.width] for window in source_state.windows
        ],
        "source_window_feature_hw": list(source_state.raw[0].shape[-2:]),
        "reasoning_num_maps": len(results),
        "reasoning_feature_hw": [list(item["continuous_score"].shape) for item in results],
        "reasoning_tokens_per_map": [int(item["continuous_score"].numel()) for item in results],
        "forward_positive_tokens_per_map": [
            int(item["forward_mask"].sum().item()) for item in results
        ],
        "backward_positive_tokens_per_map": [
            int(item["backward_mask"].sum().item()) for item in results
        ],
        "candidate_tokens_per_map": [
            int(item["candidate_mask"].sum().item()) for item in results
        ],
    }
    if method == "B3":
        record.update({
            "early_fused_feature_hw": list(results[0]["early_fused_feature_hw"]),
            "early_max_tokens": int(results[0]["early_max_tokens"]),
            "early_was_resized": bool(results[0]["early_was_resized"]),
        })
    post_metrics = binary_metrics(prediction, target_mask, target_ignore)
    post_metrics["boundary_fscore"] = boundary_fscore(
        prediction, target_mask, ignore=target_ignore
    )
    post_metrics.update(seam_metrics(
        prediction, target_mask, state.windows, ignore=target_ignore
    ))
    pre_metrics = binary_metrics(pre_crf_prediction, target_mask, target_ignore)
    pre_metrics["boundary_fscore"] = boundary_fscore(
        pre_crf_prediction, target_mask, ignore=target_ignore
    )
    pre_metrics.update(seam_metrics(
        pre_crf_prediction, target_mask, state.windows, ignore=target_ignore
    ))
    record.update(post_metrics)
    record.update({f"post_crf_{key}": value for key, value in post_metrics.items()})
    record.update({f"pre_crf_{key}": value for key, value in pre_metrics.items()})
    if method != "B0" and method != "B3":
        common = dict(
            similarity_scores=[item["sim_fwd"] for item in results],
            output_scores=[item["continuous_score"] for item in results],
            candidates=[item["candidate_mask"] for item in results],
            forward_masks=[item["forward_mask"] for item in results],
            backward_memberships=[item["backward_membership"] for item in results],
            nn_indices=[item["nn_ref_index"] for item in results],
            cluster_labels=[item["cluster_labels"] for item in results],
            coordinates=state.coordinates,
            quantum=args.coordinate_quantum,
            decision_threshold=model.merge_threshold,
            seed=args.seed,
        )
        raw_consistency = overlap_feature_consistency(
            state.raw, state.coordinates, args.coordinate_quantum
        )
        semantic_consistency = overlap_metrics(features=[item["debiased_feat"] for item in results], **common)
        record.update({f"raw_{key}": value for key, value in raw_consistency.items() if key in ("ofc",)})
        record.update(semantic_consistency)
    if attention:
        record.update(summarize_attention(attention))
    return record, results, stitched, attention, metric_state


def existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            keys.add((item["episode_id"], item["method"]))
    return keys


def run(args: argparse.Namespace) -> None:
    set_determinism(args.seed)
    if args.window_batch_size <= 0 or args.early_max_tokens <= 0 or args.d4_max_tokens <= 0:
        raise ValueError("window-batch-size, early-max-tokens, and d4-max-tokens must be positive")
    if args.duplicate_tolerance < 0:
        raise ValueError("duplicate-tolerance must be non-negative")
    if args.min_reference_tokens < 0:
        raise ValueError("min-reference-tokens must be non-negative")
    if args.episode_limit < 0:
        raise ValueError("episode-limit must be non-negative")
    output_dir = artifact_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    if metrics_path.exists() and not args.resume:
        raise FileExistsError(
            f"{metrics_path} already exists; use --resume or choose a new --output-dir"
        )
    done = existing_keys(metrics_path) if args.resume else set()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    methods += [f"R-{item.strip()}" for item in args.replays.split(",") if item.strip()]
    if not methods:
        raise ValueError("At least one method or replay stage must be selected")
    if len(methods) != len(set(methods)):
        raise ValueError(f"Duplicate methods were selected: {methods}")
    allowed = {"B0", "B1", "B2", "B3", "B4", "B5", *FACTORIAL}
    allowed.update({"R-D1", "R-D2", "R-D3", "R-D4", "R-D5"})
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; allowed: {sorted(allowed)}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({args.device}) but torch.cuda.is_available() is false")
    manifest_path = artifact_path(args.manifest)
    episodes = load_manifest(manifest_path)
    if args.episode_limit:
        episodes = episodes[:args.episode_limit]
    store = ISAIDStore(args.data_root)
    model = build_model(
        args.insid3_root, model_size=args.model_size, image_size=args.image_size,
        svd_components=args.svd_comps, tau=args.tau, merge_threshold=args.merge_thresh,
        mask_refiner="crf" if args.enable_crf else "bilinear",
        resize_to_orig_size=True, device=args.device,
    ).to(args.device).eval()
    write_json(output_dir / "implementation_manifest.json", implementation_manifest(
        args.insid3_root, Path(__file__).parent, vars(args)
    ))

    for episode_index, episode in enumerate(episodes):
        set_determinism(args.seed + episode_index)
        target = store.load_image(episode.target_image_id)
        target_mask, target_ignore = store.target_masks(episode.target_image_id, episode.class_id)
        target_mask = target_mask.to(args.device)
        target_ignore = target_ignore.to(args.device)
        reference_images = [store.load_image(item) for item in episode.reference_image_ids]
        reference_masks = [store.binary_mask(item, episode.class_id) for item in episode.reference_image_ids]
        reference = prepare_reference(model, reference_images, reference_masks, args.device)
        insufficient = [
            (image_id, count)
            for image_id, count in zip(
                episode.reference_image_ids, reference.foreground_token_counts
            )
            if count < args.min_reference_tokens
        ]
        if insufficient:
            details = ", ".join(f"{image_id}={count}" for image_id, count in insufficient)
            raise RuntimeError(
                f"Episode {episode.episode_id} has undersized reference foreground "
                f"({details} tokens; required >= {args.min_reference_tokens}). "
                "Regenerate the manifest with the same --min-reference-tokens."
            )
        windows = make_windows(target.height, target.width, episode.window_crop, episode.window_stride)
        if episode_index == 0 and not args.skip_duplicate_control:
            control = duplicate_control(model, target, windows[0], args.device)
            write_json(output_dir / "duplicate_control.json", control)
            if max(control.values()) > args.duplicate_tolerance:
                raise RuntimeError(
                    f"Duplicate control exceeded tolerance {args.duplicate_tolerance}: {control}"
                )
        state = I1_extract_windows(model, target, windows, args.device, args.window_batch_size)
        checkpoint_dir = output_dir / "checkpoints" / episode.episode_id
        if args.save_checkpoints:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            reference_path = checkpoint_dir / "reference.pt"
            extraction_path = checkpoint_dir / "window_extraction.pt"
            if not (args.resume and reference_path.exists()):
                torch.save(_to_cpu({
                    "episode": asdict(episode),
                    "reference_image_ids": episode.reference_image_ids,
                    "masks": reference.masks,
                    "raw_features": reference.raw_features,
                    "debiased_features": reference.debiased_features,
                    "prototype": reference.prototype,
                    "foreground_token_counts": reference.foreground_token_counts,
                    "min_reference_tokens": args.min_reference_tokens,
                    "encoder_input_hw": [int(model.image_size), int(model.image_size)],
                    "feature_hw": list(reference.raw_features.shape[-2:]),
                }), reference_path)
            if not (args.resume and extraction_path.exists()):
                torch.save(_to_cpu({
                    "episode": asdict(episode),
                    "target_image_id": episode.target_image_id,
                    "target_image_hw": [target.height, target.width],
                    "encoder_input_hw_per_window": [int(model.image_size), int(model.image_size)],
                    "encoder_patch_size": 16,
                    "window_specs": [window.to_dict() for window in windows],
                    "window_source_hw": [[window.height, window.width] for window in windows],
                    "feature_hw": [list(feature.shape[-2:]) for feature in state.raw],
                    "raw_features": state.raw,
                    "debiased_features": state.debiased,
                    "token_coordinates_xy": state.coordinates,
                }), extraction_path)
        # B1 reasoning is the common structural/candidate checkpoint for all variants.
        base_results = _reason_windows(model, reference, state.raw, state.debiased)
        for method in methods:
            if (episode.episode_id, method) in done:
                continue
            record, results, stitched, attention, metric_state = _evaluate_method(
                method, episode, model, reference, target, target_mask, target_ignore,
                state, args, base_results,
            )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")
            print(json.dumps(record, ensure_ascii=False, allow_nan=True), flush=True)
            if args.save_checkpoints:
                reasoning_windows = (
                    [Window(0, 0, 0, target.width, target.height)]
                    if method in {"B0", "B3"} else metric_state.windows
                )
                source_windows = (
                    reasoning_windows if method == "B0" else state.windows
                )
                include_features = method not in {"B1", "B2", "B4", "B5"}
                torch.save(_to_cpu({
                    "episode": asdict(episode),
                    "method": method,
                    "source_window_specs": [window.to_dict() for window in source_windows],
                    "reasoning_window_specs": [window.to_dict() for window in reasoning_windows],
                    "resolution": _resolution_metadata(
                        method, model, metric_state, results,
                        metric_state if method == "B0" else state,
                    ),
                    "states": _checkpoint_payload(
                        results, None, attention, include_features=include_features
                    ),
                    "stitched": stitched,
                }), checkpoint_dir / f"{method}.pt")


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        manifest_path = artifact_path(args.manifest)
        episodes = generate_manifest(
            ISAIDStore(args.data_root), manifest_path, args.fold, args.shots,
            args.num_episodes, args.window_crop, args.window_stride, args.seed,
            cross_window_only=not args.include_single_window_targets,
            min_reference_tokens=args.min_reference_tokens,
            encoder_image_size=args.image_size,
            reference_feature_grid=args.image_size // 16,
        )
        print(f"Wrote {len(episodes)} episodes to {manifest_path}")
    else:
        run(args)


if __name__ == "__main__":
    main()
