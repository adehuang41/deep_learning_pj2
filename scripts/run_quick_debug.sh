#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
ENV_NAME="${ENV_NAME:-semantic_entropy}"

conda run -n "${ENV_NAME}" python -m src.train \
  --config configs/baseline.yaml \
  --epochs 1 \
  --subset_size 512 \
  --val_subset_size 512 \
  --num_workers 2
conda run -n "${ENV_NAME}" python -m src.train \
  --config configs/final_model.yaml \
  --epochs 1 \
  --subset_size 512 \
  --val_subset_size 512 \
  --num_workers 2
conda run -n "${ENV_NAME}" python -m src.evaluate \
  --config configs/final_model.yaml \
  --val_subset_size 512 \
  --num_workers 2
