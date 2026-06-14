#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}" \
conda run --no-capture-output -n semantic_entropy python -m src.bn_p0 "$@"
