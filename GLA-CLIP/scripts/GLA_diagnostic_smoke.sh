#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Run only after GLA_reference_smoke.sh confirms that the faithful forward gate
# is still empty/saturated. This is a separate algorithm variant: adaptive mode
# keeps >0 unless it is empty or >95% positive, then retains the exact top 10%.
export MANIFEST="${MANIFEST:-manifests/isaid_fold0_refdiag.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/A0_A7_refdiag_v2_adaptive_smoke}"
export EPISODE_LIMIT="${EPISODE_LIMIT:-5}"
export METHODS="${METHODS:-B1,A0,A1,A3,A4,A5,A7}"
export MIN_REFERENCE_TOKENS="${MIN_REFERENCE_TOKENS:-200}"
export MIN_REFERENCE_RATIO="${MIN_REFERENCE_RATIO:-0.05}"
export FORWARD_GATE_MODE="${FORWARD_GATE_MODE:-adaptive}"
export FORWARD_QUANTILE="${FORWARD_QUANTILE:-0.9}"
export FORWARD_MAX_POSITIVE_RATIO="${FORWARD_MAX_POSITIVE_RATIO:-0.95}"
export MATCHING_DIAGNOSTICS="${MATCHING_DIAGNOSTICS:-1}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"

exec bash "${SCRIPT_DIR}/GLA_ablation.sh"
