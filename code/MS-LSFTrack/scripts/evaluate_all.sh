#!/usr/bin/env bash
set -euo pipefail

# Check, evaluate and summarize a tracker list for one clean-layout dataset.
#
# Example:
#   TRACKERS="bytetrack ocsort sfsort ms_lsf_ird" \
#   bash scripts/evaluate_all.sh configs/datasets/irdmstrack_v3_det_label.yaml test

DATASET_CONFIG="${1:-configs/datasets/irdmstrack_v3_det_label.yaml}"
SPLIT="${2:-test}"
export MS_LSF_DATA_ROOT="${MS_LSF_DATA_ROOT:-${PWD}/data}"
export MS_LSF_OUTPUT_ROOT="${MS_LSF_OUTPUT_ROOT:-${PWD}/outputs}"

TRACKERS="${TRACKERS:-bytetrack ocsort sfsort ms_lsf_ird}"

python tools/check_results.py \
  --dataset "${DATASET_CONFIG}" \
  --split "${SPLIT}" \
  --trackers ${TRACKERS}

python tools/evaluate_tracks.py \
  --dataset "${DATASET_CONFIG}" \
  --split "${SPLIT}" \
  --mode point \
  --trackers ${TRACKERS}

python tools/evaluate_tracks.py \
  --dataset "${DATASET_CONFIG}" \
  --split "${SPLIT}" \
  --mode iou \
  --trackers ${TRACKERS}

python tools/compare_trackers.py \
  --dataset "${DATASET_CONFIG}" \
  --split "${SPLIT}" \
  --trackers ${TRACKERS}
