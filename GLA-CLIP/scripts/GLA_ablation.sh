#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="${SCRIPT_DIR}/../gla_insid3_experiments"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:-manifests/isaid_fold0_mvp.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/isaid_fold0_A0_A7_v2_ref20}"
EPISODE_LIMIT="${EPISODE_LIMIT:-0}"
TOKEN_BANK="${TOKEN_BANK:-duplicate}"
METHODS="${METHODS:-B1,A0,A1,A2,A3,A4,A5,A6,A7}"
MIN_REFERENCE_TOKENS="${MIN_REFERENCE_TOKENS:-20}"
MIN_REFERENCE_RATIO="${MIN_REFERENCE_RATIO:-0.0}"
FORWARD_GATE_MODE="${FORWARD_GATE_MODE:-zero}"
FORWARD_QUANTILE="${FORWARD_QUANTILE:-0.9}"
FORWARD_MAX_POSITIVE_RATIO="${FORWARD_MAX_POSITIVE_RATIO:-0.95}"
DN_LAMBDA2="${DN_LAMBDA2:-30}"
MATCHING_DIAGNOSTICS="${MATCHING_DIAGNOSTICS:-0}"
SAVE_CHECKPOINTS="${SAVE_CHECKPOINTS:-0}"
WINDOW_BATCH_SIZE="${WINDOW_BATCH_SIZE:-2}"
QUERY_CHUNK="${QUERY_CHUNK:-128}"
RESUME="${RESUME:-0}"
cd "${EXP_DIR}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ "${MATCHING_DIAGNOSTICS}" == "1" ]]; then
  EXTRA_ARGS+=(--matching-diagnostics)
fi
if [[ "${SAVE_CHECKPOINTS}" == "1" ]]; then
  EXTRA_ARGS+=(--save-checkpoints)
fi

# A0-A7: KVE / Proxy Anchor / Dynamic Normalization 的 2^3 全因子消融。
# B1 是原始 late-SW 对照；A0 是共同 local attention 下的三开关全关。
# v2 修复了所有 token-bank 下 DN cutoff 的实际 mask，不能续跑旧版输出目录。
"${PYTHON_BIN}" run_experiment.py \
  --command run \
  --insid3-root /data2/cld/in_context/INSID3-main \
  --data-root /data/lky/data/rs_seg \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --methods "${METHODS}" \
  --replays '' \
  --image-size 1024 \
  --min-reference-tokens "${MIN_REFERENCE_TOKENS}" \
  --min-reference-ratio "${MIN_REFERENCE_RATIO}" \
  --token-bank "${TOKEN_BANK}" \
  --proxy-rho 0.6 \
  --proxy-iters 2 \
  --dn-lambda1 0.3 \
  --dn-lambda2 "${DN_LAMBDA2}" \
  --forward-gate-mode "${FORWARD_GATE_MODE}" \
  --forward-quantile "${FORWARD_QUANTILE}" \
  --forward-max-positive-ratio "${FORWARD_MAX_POSITIVE_RATIO}" \
  --fixed-beta 1.2 \
  --fixed-gamma 3.0 \
  --query-chunk "${QUERY_CHUNK}" \
  --episode-limit "${EPISODE_LIMIT}" \
  --device cuda \
  --window-batch-size "${WINDOW_BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}"
