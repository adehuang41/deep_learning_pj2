"""Smoke-test model factory forward passes."""

from __future__ import annotations

import torch

from src.models import build_model


def main() -> None:
    x = torch.randn(2, 3, 32, 32)
    for model_name in [
        "simple_cnn",
        "compact_resnet",
        "compact_resnet_v2",
        "preact_resnet_cifar",
        "vgg_a",
        "vgg_a_bn",
    ]:
        model = build_model(model_name)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        assert tuple(logits.shape) == (2, 10), model_name


if __name__ == "__main__":
    main()
