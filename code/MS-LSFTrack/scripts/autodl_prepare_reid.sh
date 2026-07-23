#!/usr/bin/env bash
set -euo pipefail

# Prepare or refresh ReID weights under MS_LSF_WEIGHT_ROOT.
#
# Public/fair default:
#   bash scripts/autodl_prepare_reid.sh osnet_x0_25_msmt17.pt
#
# Local checkpoint:
#   COPY_LOCAL=1 bash scripts/autodl_prepare_reid.sh /path/to/my_reid/best.pt

MODEL="${1:-osnet_x0_25_msmt17.pt}"
DEVICE="${DEVICE:-0}"
export MS_LSF_WEIGHT_ROOT="${MS_LSF_WEIGHT_ROOT:-${PWD}/weights}"

CMD=(python tools/prepare_reid_weights.py
  --model "${MODEL}"
  --output-dir "${MS_LSF_WEIGHT_ROOT}/reid"
  --device "${DEVICE}"
)

if [ "${COPY_LOCAL:-0}" = "1" ]; then
  CMD+=(--copy-local)
fi

"${CMD[@]}"
