#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DN_VALUES="${DN_VALUES:-1 3 10 30}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dn_refdiag_v2}"

# Run only after the corrected diagnostic smoke has non-empty predictions.
# A0/A1 are controls; A3/A5/A7 expose DN alone and both KVE+DN interactions.
for DN_VALUE in ${DN_VALUES}; do
  MANIFEST="${MANIFEST:-manifests/isaid_fold0_refdiag.jsonl}" \
  OUTPUT_DIR="${OUTPUT_ROOT}_lambda${DN_VALUE}" \
  EPISODE_LIMIT="${EPISODE_LIMIT:-5}" \
  METHODS="${METHODS:-A0,A1,A3,A5,A7}" \
  MIN_REFERENCE_TOKENS="${MIN_REFERENCE_TOKENS:-200}" \
  MIN_REFERENCE_RATIO="${MIN_REFERENCE_RATIO:-0.05}" \
  FORWARD_GATE_MODE="${FORWARD_GATE_MODE:-adaptive}" \
  MATCHING_DIAGNOSTICS="${MATCHING_DIAGNOSTICS:-0}" \
  SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}" \
  DN_LAMBDA2="${DN_VALUE}" \
  RESUME="${RESUME:-0}" \
  bash "${SCRIPT_DIR}/GLA_ablation.sh"
done
