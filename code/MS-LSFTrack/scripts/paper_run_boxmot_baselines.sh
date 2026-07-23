#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/paper_run_boxmot_baselines.sh DATASET_CONFIG [SPLIT] [REID_WEIGHTS]"
  echo
  echo "REID_WEIGHTS is used for StrongSORT/BoT-SORT/DeepOCSORT/HybridSORT/BoostTrack."
  echo "If omitted, weights/reid/osnet_x0_25_msmt17.pt will be used automatically when present."
  exit 2
fi

DATASET_CONFIG="$1"
SPLIT="${2:-test}"
REID_WEIGHTS="${3:-}"
DEFAULT_REID_WEIGHTS="${MS_LSF_WEIGHT_ROOT:-${PWD}/weights}/reid/osnet_x0_25_msmt17.pt"
BUNDLED_REID_WEIGHTS="${PWD}/weights/reid/osnet_x0_25_msmt17.pt"
if [ -z "${REID_WEIGHTS}" ] && [ -f "${DEFAULT_REID_WEIGHTS}" ]; then
  REID_WEIGHTS="${DEFAULT_REID_WEIGHTS}"
elif [ -z "${REID_WEIGHTS}" ] && [ -f "${BUNDLED_REID_WEIGHTS}" ]; then
  REID_WEIGHTS="${BUNDLED_REID_WEIGHTS}"
fi

python tools/run_baseline_trackers.py \
  --dataset "$DATASET_CONFIG" \
  --split "$SPLIT" \
  --trackers bytetrack ocsort sfsort

if [ -n "$REID_WEIGHTS" ]; then
  python tools/run_baseline_trackers.py \
    --dataset "$DATASET_CONFIG" \
    --split "$SPLIT" \
    --trackers strongsort botsort deepocsort hybridsort boosttrack \
    --reid-weights "$REID_WEIGHTS"
else
  echo "[SKIP] ReID-based baselines were not run because REID_WEIGHTS was not provided."
fi
