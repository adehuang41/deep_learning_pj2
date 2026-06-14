#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
ENV_NAME="${ENV_NAME:-semantic_entropy}"

conda run -n "${ENV_NAME}" python -m src.train --config configs/baseline.yaml
conda run -n "${ENV_NAME}" python -m src.train --config configs/final_model.yaml
conda run -n "${ENV_NAME}" python -m src.evaluate --config configs/final_model.yaml

conda run -n "${ENV_NAME}" python -m src.ablation --config configs/ablation.yaml

conda run -n "${ENV_NAME}" python -m src.bn_analysis \
  --config configs/vgga.yaml \
  --config_bn configs/vgga_bn.yaml \
  --analysis_config configs/bn_analysis.yaml
conda run -n "${ENV_NAME}" python -m src.loss_landscape --config configs/loss_landscape.yaml

conda run -n "${ENV_NAME}" python -m src.visualize --results_dir results
