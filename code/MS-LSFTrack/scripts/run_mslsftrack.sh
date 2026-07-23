#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash scripts/run_mslsftrack.sh DATASET_CONFIG CHECKPOINT TRACKER_NAME [SPLIT] [extra run_tracker args...]"
  echo
  echo "Example:"
  echo "  bash scripts/run_mslsftrack.sh \\"
  echo "    configs/datasets/irdmstrack_v3_det_label.yaml \\"
  echo "    /path/to/checkpoints/ms_lsf_ird_v3/best.pt \\"
  echo "    ms_lsf_ird_v3 test"
  exit 2
fi

DATASET_CONFIG="$1"
CHECKPOINT="$2"
TRACKER_NAME="$3"
SPLIT="${4:-test}"
if [ "$#" -ge 4 ]; then
  shift 4
else
  shift 3
fi

python tools/run_tracker.py \
  --dataset "$DATASET_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split "$SPLIT" \
  --tracker-name "$TRACKER_NAME" \
  --score-mode ambiguity_rerank \
  --scale-mode learned_density \
  --structure-radii-override 10,15,20 \
  --density-prior-direction normal \
  --density-prior-lambda 0.5 \
  --asr-density-gain 0.0 \
  "$@"
