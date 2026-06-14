#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
ENV_NAME="${ENV_NAME:-semantic_entropy}"
conda run -n "${ENV_NAME}" python -m src.train --config configs/final_model.yaml "$@"
conda run -n "${ENV_NAME}" python -m src.evaluate --config configs/final_model.yaml --split val
