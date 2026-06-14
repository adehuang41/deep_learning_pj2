"""A small CNN baseline for CIFAR-10."""

from __future__ import annotations

from torch import nn

from .common import init_weights, make_activation


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            make_activation(activation),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            make_activation(activation),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            make_activation(activation),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        init_weights(self)

    def forward(self, x):
        return self.classifier(self.features(x))

