#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" \
conda run -n semantic_entropy python -m src.stage_c \
  --target_epochs "${1:-250}" \
  --output_csv "results/metrics/stageC_w4_w6_${1:-250}.csv"
