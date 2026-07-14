#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/../gla_insid3_experiments"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${EXP_DIR}"

"${PYTHON_BIN}" run_experiment.py \
  --command run \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --output-dir outputs/isaid_fold0_factorial \
  --methods A0,A1,A2,A3,A4,A5,A6,A7 \
  --replays '' \
  --token-bank duplicate
