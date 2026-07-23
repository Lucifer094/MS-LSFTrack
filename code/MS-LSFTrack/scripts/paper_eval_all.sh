#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/paper_eval_all.sh DATASET_CONFIG TRACKER_NAME [SPLIT]"
  exit 2
fi

DATASET_CONFIG="$1"
TRACKER_NAME="$2"
SPLIT="${3:-test}"

bash scripts/paper_evaluate_tracker.sh "$DATASET_CONFIG" "$TRACKER_NAME" "$SPLIT"
