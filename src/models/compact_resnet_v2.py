"""CompactResNetV2: an efficient custom residual CNN for CIFAR-10."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .common import init_weights, make_activation


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class ECALite(nn.Module):
    """Lightweight channel attention with a 1D local channel interaction."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (int(kernel_size) - 1) // 2
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).squeeze(-1).transpose(1, 2)
        y = self.conv(y).transpose(1, 2).unsqueeze(-1)
        return x * self.gate(y)


class CompactV2Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "silu",
        drop_path: float = 0.0,
        use_eca: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=int(stride),
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
        self.eca = ECALite(out_channels) if use_eca else nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.act_out = make_activation(activation)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=int(stride), bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.eca(out)
        out = residual + self.drop_path(out)
        return self.act_out(out)


class CompactResNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        channels: Sequence[int] = (64, 128, 256, 384),
        blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.2,
        activation: str = "silu",
        drop_path_rate: float = 0.05,
        use_eca: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) != len(blocks_per_stage):
            raise ValueError("channels and blocks_per_stage must have the same length")
        widths = [int(c) for c in channels]
        depths = [int(n) for n in blocks_per_stage]
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            make_activation(activation),
        )
        total_blocks = sum(depths)
        drop_values = torch.linspace(0.0, float(drop_path_rate), total_blocks).tolist()
        drop_idx = 0
        stages: list[nn.Module] = []
        in_channels = widths[0]
        for stage_idx, (out_channels, num_blocks) in enumerate(zip(widths, depths)):
            blocks: list[nn.Module] = []
            for block_idx in range(num_blocks):
                stride = 2 if stage_idx > 0 and block_idx == 0 else 1
                blocks.append(
                    CompactV2Block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride,
                        activation=activation,
                        drop_path=float(drop_values[drop_idx]),
                        use_eca=use_eca,
                    )
                )
                in_channels = out_channels
                drop_idx += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(widths[-1], num_classes),
        )
        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)
