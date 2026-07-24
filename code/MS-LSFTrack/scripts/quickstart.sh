#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/quickstart.sh configs/datasets/irdmstrack_v3_det_label.yaml MS-LSFTrack cuda
#
# Optional environment variables:
#   MS_LSF_DATA_ROOT=/path/to/MS-LSFTrack-Datasets
#   MS_LSF_OUTPUT_ROOT=/path/to/MS-LSFTrack-Outputs
#   MS_LSF_RUN_ROOT=/path/to/MS-LSFTrack-Runs
#   MS_LSF_CACHE_ROOT=/path/to/MS-LSFTrack-Cache

DATASET_CONFIG="${1:-configs/datasets/irdmstrack_v3_det_label.yaml}"
EXP_NAME="${2:-ms_lsf_assoc}"
DEVICE="${3:-cuda}"
export MS_LSF_DATA_ROOT="${MS_LSF_DATA_ROOT:-${PWD}/data}"
export MS_LSF_OUTPUT_ROOT="${MS_LSF_OUTPUT_ROOT:-${PWD}/outputs}"
export MS_LSF_RUN_ROOT="${MS_LSF_RUN_ROOT:-${PWD}/runs}"
export MS_LSF_CACHE_ROOT="${MS_LSF_CACHE_ROOT:-${PWD}/cache}"

python -m pip install -r requirements.txt

python tools/run_assoc_pipeline.py \
  --dataset "${DATASET_CONFIG}" \
  --exp-name "${EXP_NAME}" \
  --device "${DEVICE}" \
  --model multiscale_lsf \
  --neighbor-mode radius \
  --radii 10,15,20 \
  --epochs 8 \
  --batch-size 96 \
  --workers 4 \
  --score-mode ambiguity_rerank \
  --scale-mode learned_density \
  --density-prior-direction normal \
  --density-prior-lambda 0.5 \
  --asr-density-gain 0.0 \
  --eval-mode point
