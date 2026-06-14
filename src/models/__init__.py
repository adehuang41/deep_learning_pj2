"""Model factory for the CIFAR-10 project."""

from __future__ import annotations

from typing import Any

from torch import nn

from .compact_resnet import CompactResNetCIFAR
from .compact_resnet_v2 import CompactResNetV2
from .preact_resnet_cifar import CIFARPreActResNet
from .simple_cnn import SimpleCNN
from .vgg_a import VGGA
from .vgg_a_bn import VGGABatchNorm


def build_model(model_name: str, **kwargs: Any) -> nn.Module:
    name = model_name.lower()
    if name == "simple_cnn":
        return SimpleCNN(
            num_classes=kwargs.get("num_classes", 10),
            dropout=kwargs.get("dropout", 0.3),
            activation=kwargs.get("activation", "relu"),
        )
    if name == "compact_resnet":
        return CompactResNetCIFAR(
            num_classes=kwargs.get("num_classes", 10),
            channels=kwargs.get("channels", (64, 128, 256)),
            dropout=kwargs.get("dropout", 0.2),
            activation=kwargs.get("activation", "silu"),
        )
    if name == "compact_resnet_v2":
        return CompactResNetV2(
            num_classes=kwargs.get("num_classes", 10),
            channels=kwargs.get("channels", (64, 128, 256, 384)),
            blocks_per_stage=kwargs.get("blocks_per_stage", (2, 2, 2, 2)),
            dropout=kwargs.get("dropout", 0.2),
            activation=kwargs.get("activation", "silu"),
            drop_path_rate=kwargs.get("drop_path_rate", 0.05),
            use_eca=kwargs.get("use_eca", True),
        )
    if name == "preact_resnet_cifar":
        return CIFARPreActResNet(
            num_classes=kwargs.get("num_classes", 10),
            depth=int(kwargs.get("depth", 28) or 28),
            widen_factor=int(kwargs.get("widen_factor", 4) or 4),
            dropout=float(kwargs.get("dropout", 0.3)),
            activation=str(kwargs.get("activation", "relu")),
        )
    if name == "vgg_a":
        return VGGA(
            num_classes=kwargs.get("num_classes", 10),
            dropout=kwargs.get("dropout", 0.5),
        )
    if name == "vgg_a_bn":
        return VGGABatchNorm(
            num_classes=kwargs.get("num_classes", 10),
            dropout=kwargs.get("dropout", 0.5),
        )
    raise ValueError(f"Unknown model: {model_name}")


__all__ = [
    "CompactResNetCIFAR",
    "CompactResNetV2",
    "CIFARPreActResNet",
    "SimpleCNN",
    "VGGA",
    "VGGABatchNorm",
    "build_model",
]
