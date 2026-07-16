#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# This manifest is deliberately separate from the preregistered ref20 manifest.
# It tests whether reference foreground imbalance is the immediate matching
# bottleneck; it must not silently replace the final small-object evaluation.
export MANIFEST="${MANIFEST:-manifests/isaid_fold0_refdiag.jsonl}"
export NUM_EPISODES="${NUM_EPISODES:-10}"
export MIN_REFERENCE_TOKENS="${MIN_REFERENCE_TOKENS:-200}"
export MIN_REFERENCE_RATIO="${MIN_REFERENCE_RATIO:-0.05}"

exec bash "${SCRIPT_DIR}/manifest.sh"
