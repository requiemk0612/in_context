#!/usr/bin/env python3
"""Aggregate episode metrics and compute paired bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


IDENTIFIERS = {"episode_id", "method", "fold", "class_id"}
LOWER_IS_BETTER = {
    "elapsed_seconds", "peak_memory_mb", "score_variance_mean", "ber", "ber_fg",
    "ber_bg", "seam_excess_error", "cwsd", "cwod_score_mae", "cwod_binary",
    "forward_flip_rate", "backward_membership_flip_rate", "candidate_flip_rate",
    "backward_nn_change_rate", "cluster_coassociation_disagreement",
}
EXPERIMENT_ROOT = Path(__file__).resolve().parent
GLA_ROOT = EXPERIMENT_ROOT.parent


def writable_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = EXPERIMENT_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(GLA_ROOT)
    except ValueError as exc:
        raise ValueError(f"Summary output must remain under {GLA_ROOT}: {path}") from exc
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="metrics.jsonl from run_experiment.py")
    parser.add_argument("--baseline", default="B1")
    parser.add_argument("--output", default=None, help="Defaults to summary.json beside metrics")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def lower_is_better(metric: str) -> bool:
    return metric in LOWER_IS_BETTER or any(
        metric.endswith(f"_{name}") for name in LOWER_IS_BETTER
    )


def interval(values: list[float], samples: int, rng: np.random.Generator) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"n": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    if len(array) == 1 or samples <= 0:
        mean = float(array.mean())
        return {"n": len(array), "mean": mean, "ci95_low": mean, "ci95_high": mean}
    chunk = max(1, min(samples, 2048))
    boot_means = []
    remaining = samples
    while remaining:
        current = min(chunk, remaining)
        indices = rng.integers(0, len(array), size=(current, len(array)))
        boot_means.append(array[indices].mean(axis=1))
        remaining -= current
    bootstrap = np.concatenate(boot_means)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "n": len(array), "mean": float(array.mean()),
        "ci95_low": float(low), "ci95_high": float(high),
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    keys = [(item["episode_id"], item["method"]) for item in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate episode/method rows found; use a clean output directory")
    return records


def summarize(records: list[dict[str, Any]], baseline: str, samples: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[record["method"]].append(record)
    if baseline not in by_method:
        raise ValueError(f"Baseline {baseline!r} is absent; methods are {sorted(by_method)}")

    aggregate: dict[str, Any] = {}
    for method, rows in sorted(by_method.items()):
        metric_names = sorted(set.intersection(*(
            {key for key, value in row.items() if key not in IDENTIFIERS and finite_number(value)}
            for row in rows
        ))) if rows else []
        aggregate[method] = {
            metric: interval([float(row[metric]) for row in rows], samples, rng)
            for metric in metric_names
        }

    baseline_rows = {row["episode_id"]: row for row in by_method[baseline]}
    paired: dict[str, Any] = {}
    for method, rows in sorted(by_method.items()):
        if method == baseline:
            continue
        method_rows = {row["episode_id"]: row for row in rows}
        common = sorted(set(baseline_rows) & set(method_rows))
        metric_names = sorted(set.intersection(*(
            {
                key for key, value in method_rows[episode].items()
                if key not in IDENTIFIERS and finite_number(value)
                and finite_number(baseline_rows[episode].get(key))
            }
            for episode in common
        ))) if common else []
        paired[method] = {}
        for metric in metric_names:
            deltas = [
                float(method_rows[episode][metric]) - float(baseline_rows[episode][metric])
                for episode in common
            ]
            result = interval(deltas, samples, rng)
            result["direction"] = "lower_is_better" if lower_is_better(metric) else "higher_is_better"
            paired[method][metric] = result

    return {
        "format_version": 1,
        "baseline": baseline,
        "bootstrap_samples": samples,
        "seed": seed,
        "aggregate": aggregate,
        "paired_delta_method_minus_baseline": paired,
    }


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics).expanduser()
    if not metrics_path.is_absolute():
        metrics_path = (EXPERIMENT_ROOT / metrics_path).resolve()
    else:
        metrics_path = metrics_path.resolve()
    output_path = writable_path(args.output) if args.output else writable_path(metrics_path.with_name("summary.json"))
    summary = summarize(load_records(metrics_path), args.baseline, args.bootstrap_samples, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
