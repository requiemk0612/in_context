#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/../gla_insid3_experiments"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/isaid_fold0_A0_A7_ref20}"
EPISODE_LIMIT="${EPISODE_LIMIT:-0}"
TOKEN_BANK="${TOKEN_BANK:-duplicate}"
RESUME="${RESUME:-0}"
cd "${EXP_DIR}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

# A0-A7: KVE / Proxy Anchor / Dynamic Normalization 的 2^3 全因子消融。
# manifest 必须由 scripts/manifest.sh 生成，以保证每个 reference >= 20 tokens。
"${PYTHON_BIN}" run_experiment.py \
  --command run \
  --insid3-root /data2/cld/in_context/INSID3-main \
  --data-root /data/lky/data/rs_seg \
  --manifest manifests/isaid_fold0_mvp.jsonl \
  --output-dir "${OUTPUT_DIR}" \
  --methods A0,A1,A2,A3,A4,A5,A6,A7 \
  --replays '' \
  --image-size 1024 \
  --min-reference-tokens 20 \
  --token-bank "${TOKEN_BANK}" \
  --proxy-rho 0.6 \
  --proxy-iters 2 \
  --dn-lambda1 0.3 \
  --dn-lambda2 30 \
  --fixed-beta 1.2 \
  --fixed-gamma 3.0 \
  --query-chunk 128 \
  --episode-limit "${EPISODE_LIMIT}" \
  --device cuda \
  --window-batch-size 1 \
  "${EXTRA_ARGS[@]}"
