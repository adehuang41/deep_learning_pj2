"""Shared utilities for experiments."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)


def get_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_json(data: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).sum().item() / targets.numel())


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def epoch_time(start_time: float) -> float:
    return float(time.perf_counter() - start_time)


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def worker_seed_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def merge_config_with_overrides(
    config: dict[str, Any],
    overrides: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged


def infer_run_name(config: dict[str, Any]) -> str:
    if config.get("run_name"):
        return str(config["run_name"])
    model = str(config.get("model", "model"))
    suffix = str(config.get("activation", "")).strip()
    return f"{model}_{suffix}".strip("_")


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minute}m {sec}s"
    if minute:
        return f"{minute}m {sec}s"
    return f"{sec}s"


def set_cuda_visible_default(device_id: str = "7") -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", device_id)
