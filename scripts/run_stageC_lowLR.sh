#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" \
conda run -n semantic_entropy python -m src.stage_c_lowlr \
  --resume_mode low_lr_finetune \
  --fine_tune_epochs "${1:-100}" \
  --output_csv "results/metrics/stageC_lowLR_w4_w6.csv"
