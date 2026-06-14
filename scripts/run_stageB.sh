#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
ENV_NAME="${ENV_NAME:-semantic_entropy}"
conda run -n "${ENV_NAME}" python -m src.stage_b "$@"
