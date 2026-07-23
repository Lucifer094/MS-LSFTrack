#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export MS_LSF_DATA_ROOT="${MS_LSF_DATA_ROOT:-${PWD}/data}"
export MS_LSF_OUTPUT_ROOT="${MS_LSF_OUTPUT_ROOT:-${PWD}/outputs}"

SPLIT="${SPLIT:-test}"
TRACKER_NAME="${TRACKER_NAME:-asdt_tc_filter}"
MAX_BBOXES_PER_FRAME="${MAX_BBOXES_PER_FRAME:-0}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.0}"
MAX_MOVEMENT="${MAX_MOVEMENT:-0.09}"
MAX_MISSING="${MAX_MISSING:-3}"
INTERPOLATE_GAP="${INTERPOLATE_GAP:-3}"
MIN_TRACK_LENGTH="${MIN_TRACK_LENGTH:-1}"
OVERWRITE_FLAG=()

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

DATASETS=(
  "configs/datasets/irdmstrack_v3_det_label.yaml"
  "configs/datasets/gmot40_small_det_label.yaml"
  "configs/datasets/irsatvideo_leo_resunet_rfr.yaml"
)

for DATASET in "${DATASETS[@]}"; do
  echo "[RUN] ${DATASET} split=${SPLIT} tracker=${TRACKER_NAME}"
  python tools/run_asdt_tracker.py \
    --dataset "${DATASET}" \
    --split "${SPLIT}" \
    --tracker-name "${TRACKER_NAME}" \
    --score-threshold "${SCORE_THRESHOLD}" \
    --max-bboxes-per-frame "${MAX_BBOXES_PER_FRAME}" \
    --max-movement "${MAX_MOVEMENT}" \
    --max-missing "${MAX_MISSING}" \
    --interpolate-gap "${INTERPOLATE_GAP}" \
    --min-track-length "${MIN_TRACK_LENGTH}" \
    "${OVERWRITE_FLAG[@]}"
done
