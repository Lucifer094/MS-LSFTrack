#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/evaluate_tracker.sh DATASET_CONFIG TRACKER_NAME [SPLIT]"
  exit 2
fi

DATASET_CONFIG="$1"
TRACKER_NAME="$2"
SPLIT="${3:-test}"

python tools/evaluate_tracks.py \
  --dataset "$DATASET_CONFIG" \
  --trackers "$TRACKER_NAME" \
  --split "$SPLIT" \
  --mode point

python tools/evaluate_tracks.py \
  --dataset "$DATASET_CONFIG" \
  --trackers "$TRACKER_NAME" \
  --split "$SPLIT" \
  --mode iou
