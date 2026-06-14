"""Compact residual CNN designed for CIFAR-10."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from .common import init_weights, make_activation


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = "silu",
        downsample: bool = False,
    ) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = make_activation(activation)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = make_activation(activation)
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.act2(out)


class CompactResNetCIFAR(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: Sequence[int] = (64, 128, 256),
        dropout: float = 0.2,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if len(channels) != 3:
            raise ValueError("channels must contain exactly three stage widths")
        c1, c2, c3 = [int(c) for c in channels]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            make_activation(activation),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(c1, c1, activation=activation),
            ResidualBlock(c1, c1, activation=activation),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(c1, c2, activation=activation, downsample=True),
            ResidualBlock(c2, c2, activation=activation),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(c2, c3, activation=activation, downsample=True),
            ResidualBlock(c3, c3, activation=activation),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, num_classes),
        )
        init_weights(self)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

