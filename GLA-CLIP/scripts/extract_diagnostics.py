#!/usr/bin/env python3
"""Quickly extract B0/B1/B3 diagnostics from an experiment metrics.jsonl."""

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
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--only-problems", action="store_true",
        help="Show only rows where backward or candidate tokens are all zero",
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


def load_rows(path: Path, methods: set[str]) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if item.get("method") in methods:
                rows.append(item)
    if not rows:
        raise ValueError(f"No selected methods {sorted(methods)} found in {path}")
    return rows


def extract(row: dict[str, Any]) -> dict[str, Any]:
    reference = numeric(row.get("reference_foreground_tokens"))
    forward = numeric(row.get("forward_positive_tokens_per_map"))
    backward = numeric(row.get("backward_positive_tokens_per_map"))
    candidate = numeric(row.get("candidate_tokens_per_map"))
    return {
        "episode": row.get("episode_id", "?"),
        "method": row.get("method", "?"),
        "reference_tokens": [int(value) for value in reference],
        "reference_min": minimum(reference),
        "maps": max(len(forward), len(backward), len(candidate)),
        "forward_mean": average(forward),
        "backward_mean": average(backward),
        "candidate_mean": average(candidate),
        "backward_zero_maps": sum(value == 0 for value in backward),
        "candidate_zero_maps": sum(value == 0 for value in candidate),
        "backward_all_zero": bool(backward) and sum(backward) == 0,
        "candidate_all_zero": bool(candidate) and sum(candidate) == 0,
        "fg_iou": row.get("fg_iou"),
        "dice": row.get("dice"),
        "boundary_fscore": row.get("boundary_fscore"),
        "raw_ofc": row.get("raw_ofc"),
        "semantic_ofc": row.get("ofc"),
        "cwsd": row.get("cwsd"),
        "cwod": row.get("cwod_score_mae"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    result = {}
    for method, items in sorted(grouped.items()):
        total_maps = sum(item["maps"] for item in items)
        result[method] = {
            "episodes": len(items),
            "reference_minimum": minimum([
                item["reference_min"] for item in items if item["reference_min"] is not None
            ]),
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
            "mean_fg_iou": average([
                float(item["fg_iou"]) for item in items
                if isinstance(item["fg_iou"], (int, float)) and math.isfinite(item["fg_iou"])
            ]),
            "mean_dice": average([
                float(item["dice"]) for item in items
                if isinstance(item["dice"], (int, float)) and math.isfinite(item["dice"])
            ]),
        }
    return result


def print_rows(rows: list[dict[str, Any]]) -> None:
    columns = (
        "episode", "method", "reference_tokens", "maps", "forward_mean",
        "backward_mean", "candidate_mean", "backward_zero_maps",
        "candidate_zero_maps", "fg_iou", "dice", "boundary_fscore",
        "raw_ofc", "semantic_ofc", "cwsd", "cwod",
    )
    print("\t".join(columns))
    for row in rows:
        print("\t".join(display(row[column]) for column in columns))


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    print("\n# summary")
    columns = (
        "method", "episodes", "reference_minimum", "backward_all_zero_rate",
        "candidate_all_zero_rate", "backward_zero_map_rate",
        "candidate_zero_map_rate", "mean_fg_iou", "mean_dice",
    )
    print("\t".join(columns))
    for method, values in summary.items():
        row = {"method": method, **values}
        print("\t".join(display(row[column]) for column in columns))


def main() -> None:
    args = parse_args()
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    all_rows = [extract(row) for row in load_rows(Path(args.metrics).expanduser().resolve(), methods)]
    extracted = all_rows
    if args.only_problems:
        extracted = [
            row for row in extracted
            if row["backward_all_zero"] or row["candidate_all_zero"]
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
