#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/../gla_insid3_experiments"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/SW_smoke}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-2}"
cd "${EXP_DIR}"

# One real iSAID episode through all three requested baselines.
"${PYTHON_BIN}" run_experiment.py \
  --command run \
  --insid3-root /data2/cld/in_context/INSID3-main \
  --data-root /data/lky/data/rs_seg \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --output-dir "${OUTPUT_DIR}" \
  --methods B0,B1,B3 \
  --replays '' \
  --image-size 1024 \
  --min-reference-tokens 20 \
  --min-reference-ratio 0.0 \
  --forward-gate-mode zero \
  --early-max-tokens 4096 \
  --episode-limit 1 \
  --device cuda --window-batch-size "${WINDOW_BATCH_SIZE}" \
  --save-checkpoints
