#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/../gla_insid3_experiments"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${EXP_DIR}"

"${PYTHON_BIN}" run_experiment.py \
  --command manifest \
  --data-root /data/lky/data/rs_seg \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --fold 0 --shots 1 --num-episodes 50 \
  --window-crop 512 --window-stride 256 --seed 45
