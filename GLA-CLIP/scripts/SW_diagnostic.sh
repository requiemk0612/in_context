#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/../gla_insid3_experiments"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPISODE_LIMIT="${EPISODE_LIMIT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/SW_diagnostic}"
cd "${EXP_DIR}"

# B0: full image -> 1024 -> 64x64 reasoning
# B1: 512 crops -> each 1024 -> per-window 64x64 reasoning -> late score stitch
# B3: window features -> full-image canvas -> at most 4096 tokens -> early reasoning
"${PYTHON_BIN}" run_experiment.py \
  --command run \
  --insid3-root /data2/cld/in_context/INSID3-main \
  --data-root /data/lky/data/rs_seg \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --output-dir "${OUTPUT_DIR}" \
  --methods B0,B1,B3 \
  --replays '' \
  --image-size 1024 \
  --early-max-tokens 4096 \
  --episode-limit "${EPISODE_LIMIT}" \
  --device cuda --window-batch-size 1 \
  --save-checkpoints
