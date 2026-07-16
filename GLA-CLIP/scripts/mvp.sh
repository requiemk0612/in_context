#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/../gla_insid3_experiments"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "${EXP_DIR}"

"${PYTHON_BIN}" run_experiment.py \
 --command run \
 --insid3-root /data2/cld/in_context/INSID3-main \
 --data-root /data/lky/data/rs_seg \
 --manifest manifests/isaid_fold0_mvp.jsonl \
 --output-dir outputs/mvp_v2 \
 --methods B0,B1,B2,B3,A0,A7 \
 --min-reference-tokens 20 \
 --min-reference-ratio 0.0 \
 --forward-gate-mode zero \
 --device cuda \
 --window-batch-size 1
