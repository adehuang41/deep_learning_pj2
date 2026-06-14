"""Custom CIFAR pre-activation residual network used as champion candidate."""

from __future__ import annotations

import torch
from torch import nn

from .common import init_weights, make_activation


class PreActBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = make_activation(activation)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=int(stride),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = make_activation(activation)
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=int(stride), bias=False)
            if stride != 1 or in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(self.dropout(self.act2(self.bn2(out))))
        return out + shortcut


class CIFARPreActResNet(nn.Module):
    """Pre-activation residual network adapted to 32x32 CIFAR images."""

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 28,
        widen_factor: int = 4,
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if (int(depth) - 4) % 6 != 0:
            raise ValueError("depth must satisfy depth = 6n + 4")
        blocks_per_group = (int(depth) - 4) // 6
        widths = [16, 16 * int(widen_factor), 32 * int(widen_factor), 64 * int(widen_factor)]
        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.group1 = self._make_group(
            widths[0], widths[1], blocks_per_group, stride=1, dropout=dropout, activation=activation
        )
        self.group2 = self._make_group(
            widths[1], widths[2], blocks_per_group, stride=2, dropout=dropout, activation=activation
        )
        self.group3 = self._make_group(
            widths[2], widths[3], blocks_per_group, stride=2, dropout=dropout, activation=activation
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.act = make_activation(activation)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)
        init_weights(self)

    @staticmethod
    def _make_group(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        dropout: float,
        activation: str,
    ) -> nn.Sequential:
        layers = [
            PreActBasicBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dropout=dropout,
                activation=activation,
            )
        ]
        for _ in range(1, int(blocks)):
            layers.append(
                PreActBasicBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    dropout=dropout,
                    activation=activation,
                )
            )
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.act(self.bn(x))
        x = self.pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.forward_features(x))
