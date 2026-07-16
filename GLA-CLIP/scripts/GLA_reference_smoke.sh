#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Stage 1: change only reference eligibility while keeping INSID3's faithful
# sim>0 forward gate. This separates the reference-imbalance hypothesis from
# the later adaptive-gate algorithm variant.
export MANIFEST="${MANIFEST:-manifests/isaid_fold0_refdiag.jsonl}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/A0_A7_refdiag_v2_zero_smoke}"
export EPISODE_LIMIT="${EPISODE_LIMIT:-5}"
export METHODS="${METHODS:-B1,A0,A1,A3,A4,A5,A7}"
export MIN_REFERENCE_TOKENS="${MIN_REFERENCE_TOKENS:-200}"
export MIN_REFERENCE_RATIO="${MIN_REFERENCE_RATIO:-0.05}"
export FORWARD_GATE_MODE="zero"
export MATCHING_DIAGNOSTICS="${MATCHING_DIAGNOSTICS:-1}"
export SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-1}"

exec bash "${SCRIPT_DIR}/GLA_ablation.sh"
