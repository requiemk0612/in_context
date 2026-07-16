#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/../gla_insid3_experiments"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-manifests/isaid_fold0_mvp.jsonl}"
NUM_EPISODES="${NUM_EPISODES:-50}"
MIN_REFERENCE_TOKENS="${MIN_REFERENCE_TOKENS:-20}"
MIN_REFERENCE_RATIO="${MIN_REFERENCE_RATIO:-0.0}"
SEED="${SEED:-45}"
cd "${EXP_DIR}"

"${PYTHON_BIN}" run_experiment.py \
  --command manifest \
  --data-root /data/lky/data/rs_seg \
  --manifest "${MANIFEST}" \
  --fold 0 --shots 1 --num-episodes "${NUM_EPISODES}" \
  --min-reference-tokens "${MIN_REFERENCE_TOKENS}" \
  --min-reference-ratio "${MIN_REFERENCE_RATIO}" \
  --window-crop 512 --window-stride 256 --seed "${SEED}"
