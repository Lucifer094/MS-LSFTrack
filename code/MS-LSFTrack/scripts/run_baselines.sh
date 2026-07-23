#!/usr/bin/env bash
set -euo pipefail

# Run BoxMOT baseline trackers on a clean-layout dataset.
#
# Example:
#   TRACKERS="bytetrack ocsort sfsort" \
#   bash scripts/run_baselines.sh configs/datasets/irdmstrack_v3_det_label.yaml test
#
# ReID trackers use the bundled/default osnet_x0_25_msmt17.pt unless REID_WEIGHTS is set:
#   TRACKERS="strongsort botsort deepocsort hybridsort boosttrack" \
#   REID_WEIGHTS=osnet_x0_25_msmt17.pt \
#   bash scripts/run_baselines.sh configs/datasets/irdmstrack_v3_det_label.yaml test

DATASET_CONFIG="${1:-configs/datasets/irdmstrack_v3_det_label.yaml}"
SPLIT="${2:-test}"
export MS_LSF_DATA_ROOT="${MS_LSF_DATA_ROOT:-${PWD}/data}"
export MS_LSF_OUTPUT_ROOT="${MS_LSF_OUTPUT_ROOT:-${PWD}/outputs}"
export MS_LSF_WEIGHT_ROOT="${MS_LSF_WEIGHT_ROOT:-${PWD}/weights}"

TRACKERS="${TRACKERS:-bytetrack ocsort sfsort}"
REID_WEIGHTS="${REID_WEIGHTS:-osnet_x0_25_msmt17.pt}"
if [ -f "${MS_LSF_WEIGHT_ROOT}/reid/${REID_WEIGHTS}" ]; then
  REID_WEIGHTS="${MS_LSF_WEIGHT_ROOT}/reid/${REID_WEIGHTS}"
fi
DEVICE="${DEVICE:-0}"
MAX_SCENES="${MAX_SCENES:-}"
OVERWRITE="${OVERWRITE:-0}"
BOOSTTRACK_NO_REID="${BOOSTTRACK_NO_REID:-0}"
HALF="${HALF:-0}"
STRICT="${STRICT:-0}"

CMD=(python tools/run_baseline_trackers.py
  --dataset "${DATASET_CONFIG}"
  --split "${SPLIT}"
  --trackers ${TRACKERS}
  --device "${DEVICE}"
)

if [ -n "${REID_WEIGHTS}" ]; then
  CMD+=(--reid-weights "${REID_WEIGHTS}")
fi
if [ -n "${MAX_SCENES}" ]; then
  CMD+=(--max-scenes "${MAX_SCENES}")
fi
if [ "${OVERWRITE}" = "1" ]; then
  CMD+=(--overwrite)
fi
if [ "${BOOSTTRACK_NO_REID}" = "1" ]; then
  CMD+=(--boosttrack-no-reid)
fi
if [ "${HALF}" = "1" ]; then
  CMD+=(--half)
fi
if [ "${STRICT}" = "1" ]; then
  CMD+=(--strict)
fi

"${CMD[@]}"
