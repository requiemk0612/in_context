#!/usr/bin/env python3
"""Quickly extract matching, collapse, and segmentation diagnostics from JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_METHODS = ("B0", "B1", "B3")
SCRIPT_DIR = Path(__file__).resolve().parent
GLA_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", help="Path to metrics.jsonl")
    parser.add_argument(
        "--methods", default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods, or 'all' (default: B0,B1,B3)",
    )
    parser.add_argument(
        "--only-problems", action="store_true",
        help="Show rows with an empty match/prediction or a saturated forward gate",
    )
    parser.add_argument(
        "--saturation-ratio", type=float, default=0.95,
        help="Forward-positive fraction treated as saturated (default: 0.95)",
    )
    parser.add_argument("--json-output", default=None, help="Optional summary JSON under GLA-CLIP/")
    return parser.parse_args()


def numeric(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [float(value) for value in values if isinstance(value, (int, float))]


def average(values: list[float]) -> float | None:
    return fmean(values) if values else None


def minimum(values: list[float]) -> float | None:
    return min(values) if values else None


def display(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return "-" if not math.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def writable_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = GLA_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(GLA_ROOT)
    except ValueError as exc:
        raise ValueError(f"JSON output must remain under {GLA_ROOT}: {path}") from exc
    return path


def load_rows(path: Path, methods: set[str] | None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if methods is None or item.get("method") in methods:
                rows.append(item)
    if not rows:
        selected = "all methods" if methods is None else sorted(methods)
        raise ValueError(f"No selected methods {selected} found in {path}")
    return rows


def extract(row: dict[str, Any], saturation_ratio: float = 0.95) -> dict[str, Any]:
    reference = numeric(row.get("reference_foreground_tokens"))
    reference_ratios = numeric(row.get("reference_foreground_ratios"))
    if not reference_ratios and reference:
        grid_tokens = row.get("reference_grid_tokens")
        if isinstance(grid_tokens, (int, float)) and grid_tokens > 0:
            reference_ratios = [value / float(grid_tokens) for value in reference]
    forward = numeric(row.get("forward_positive_tokens_per_map"))
    backward = numeric(row.get("backward_positive_tokens_per_map"))
    candidate = numeric(row.get("candidate_tokens_per_map"))
    reasoning_tokens = numeric(row.get("reasoning_tokens_per_map"))
    forward_fraction = numeric(row.get("forward_positive_fraction_per_map"))
    if not forward_fraction and forward and len(forward) == len(reasoning_tokens):
        forward_fraction = [
            count / total if total else 0.0
            for count, total in zip(forward, reasoning_tokens)
        ]
    backward_hit = numeric(row.get("backward_foreground_hit_fraction_per_map"))
    margin = numeric(row.get("nn_foreground_margin_mean_per_map"))
    margin_positive = numeric(row.get("nn_foreground_margin_positive_fraction_per_map"))
    nn_unique = numeric(row.get("nn_unique_reference_tokens_per_map"))
    nn_dominant = numeric(row.get("nn_dominant_reference_token_fraction_per_map"))
    semantic_dispersion = numeric(row.get("semantic_spatial_dispersion_per_map"))
    semantic_input_cosine = numeric(row.get("semantic_input_cosine_mean_per_map"))
    forward_saturated_maps = sum(value >= saturation_ratio for value in forward_fraction)
    return {
        "episode": row.get("episode_id", "?"),
        "method": row.get("method", "?"),
        "reference_tokens": [int(value) for value in reference],
        "reference_min": minimum(reference),
        "reference_ratio_min": minimum(reference_ratios),
        "target_fg": row.get("target_foreground_fraction"),
        "maps": max(len(forward), len(backward), len(candidate)),
        "forward_mean": average(forward),
        "forward_ratio": average(forward_fraction),
        "forward_saturated_maps": forward_saturated_maps,
        "backward_mean": average(backward),
        "backward_hit": average(backward_hit),
        "candidate_mean": average(candidate),
        "backward_zero_maps": sum(value == 0 for value in backward),
        "candidate_zero_maps": sum(value == 0 for value in candidate),
        "backward_all_zero": bool(backward) and sum(backward) == 0,
        "candidate_all_zero": bool(candidate) and sum(candidate) == 0,
        "margin_mean": average(margin),
        "margin_positive": average(margin_positive),
        "nn_unique": average(nn_unique),
        "nn_dominant": average(nn_dominant),
        "semantic_dispersion": average(semantic_dispersion),
        "semantic_input_cosine": average(semantic_input_cosine),
        "prediction_fg": row.get("post_crf_foreground_fraction"),
        "empty_prediction": row.get("empty_prediction"),
        "fg_iou": row.get("fg_iou"),
        "dice": row.get("dice"),
        "boundary_fscore": row.get("boundary_fscore"),
        "raw_ofc": row.get("raw_ofc"),
        "semantic_ofc": row.get("ofc"),
        "cwsd": row.get("cwsd"),
        "cwod": row.get("cwod_score_mae"),
        "attention_entropy": row.get(
            "attention_entropy_mean", row.get("attention_attention_entropy_mean")
        ),
        "attention_top1": row.get(
            "attention_top1_mass_mean", row.get("attention_attention_top1_mass_mean")
        ),
        "attention_effective_tokens": row.get(
            "attention_effective_tokens_mean",
            row.get("attention_attention_effective_tokens_mean"),
        ),
        "feature_drift": row.get("attention_feature_drift_mean"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    result = {}
    for method, items in sorted(grouped.items()):
        total_maps = sum(item["maps"] for item in items)
        nonempty = [item for item in items if item["empty_prediction"] is False]
        result[method] = {
            "episodes": len(items),
            "nonempty_episodes": len(nonempty),
            "reference_minimum": minimum([
                item["reference_min"] for item in items if item["reference_min"] is not None
            ]),
            "reference_ratio_minimum": minimum([
                item["reference_ratio_min"] for item in items
                if item["reference_ratio_min"] is not None
            ]),
            "forward_saturated_map_rate": (
                sum(item["forward_saturated_maps"] for item in items) / total_maps
                if total_maps else None
            ),
            "backward_all_zero_rate": fmean(item["backward_all_zero"] for item in items),
            "candidate_all_zero_rate": fmean(item["candidate_all_zero"] for item in items),
            "backward_zero_map_rate": (
                sum(item["backward_zero_maps"] for item in items) / total_maps
                if total_maps else None
            ),
            "candidate_zero_map_rate": (
                sum(item["candidate_zero_maps"] for item in items) / total_maps
                if total_maps else None
            ),
            "empty_prediction_rate": fmean(
                bool(item["empty_prediction"]) for item in items
            ),
            "mean_margin": average([
                item["margin_mean"] for item in items if item["margin_mean"] is not None
            ]),
            "mean_margin_positive": average([
                item["margin_positive"] for item in items
                if item["margin_positive"] is not None
            ]),
            "mean_fg_iou": average([
                float(item["fg_iou"]) for item in items
                if isinstance(item["fg_iou"], (int, float)) and math.isfinite(item["fg_iou"])
            ]),
            "mean_dice": average([
                float(item["dice"]) for item in items
                if isinstance(item["dice"], (int, float)) and math.isfinite(item["dice"])
            ]),
            "mean_fg_iou_nonempty": average([
                float(item["fg_iou"]) for item in nonempty
                if isinstance(item["fg_iou"], (int, float)) and math.isfinite(item["fg_iou"])
            ]),
        }
    return result


def print_rows(rows: list[dict[str, Any]]) -> None:
    columns = (
        "episode", "method", "reference_tokens", "reference_ratio_min", "target_fg",
        "maps", "forward_ratio", "forward_saturated_maps", "backward_mean",
        "backward_hit", "candidate_mean", "backward_zero_maps",
        "candidate_zero_maps", "margin_mean", "margin_positive", "nn_unique",
        "nn_dominant", "prediction_fg", "fg_iou", "dice", "boundary_fscore",
        "raw_ofc", "semantic_ofc", "cwsd", "cwod", "semantic_dispersion",
        "semantic_input_cosine",
        "attention_entropy", "attention_top1", "feature_drift",
    )
    print("\t".join(columns))
    for row in rows:
        print("\t".join(display(row[column]) for column in columns))


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    print("\n# summary")
    columns = (
        "method", "episodes", "nonempty_episodes", "reference_minimum", "reference_ratio_minimum",
        "forward_saturated_map_rate", "backward_all_zero_rate",
        "candidate_all_zero_rate", "backward_zero_map_rate",
        "candidate_zero_map_rate", "empty_prediction_rate", "mean_margin",
        "mean_margin_positive", "mean_fg_iou", "mean_dice", "mean_fg_iou_nonempty",
    )
    print("\t".join(columns))
    for method, values in summary.items():
        row = {"method": method, **values}
        print("\t".join(display(row[column]) for column in columns))


def main() -> None:
    args = parse_args()
    if not 0.0 < args.saturation_ratio <= 1.0:
        raise ValueError("--saturation-ratio must be in (0, 1]")
    requested = {item.strip() for item in args.methods.split(",") if item.strip()}
    methods = None if requested == {"all"} else requested
    all_rows = [
        extract(row, args.saturation_ratio)
        for row in load_rows(Path(args.metrics).expanduser().resolve(), methods)
    ]
    extracted = all_rows
    if args.only_problems:
        extracted = [
            row for row in extracted
            if (
                row["backward_all_zero"] or row["candidate_all_zero"]
                or row["forward_saturated_maps"] > 0
                or bool(row["empty_prediction"])
            )
        ]
    extracted.sort(key=lambda row: (row["episode"], row["method"]))
    summary = summarize(all_rows)
    print_rows(extracted)
    print_summary(summary)
    if args.json_output:
        output = writable_path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"rows": all_rows, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
