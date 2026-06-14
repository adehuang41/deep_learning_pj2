"""CIFAR-10 data loading utilities with a strict validation protocol."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from .utils import CIFAR10_CLASSES, ensure_dir, worker_seed_fn


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_SPLIT_FILE = "splits/cifar10_train45000_val5000_seed42.json"


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader | None = None
    test: DataLoader | None = None
    split_info: dict[str, Any] | None = None


class Cutout:
    """Mask a square image region after tensor conversion."""

    def __init__(self, length: int = 16, p: float = 1.0) -> None:
        self.length = int(length)
        self.p = float(p)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.length <= 0 or torch.rand(1).item() > self.p:
            return image
        _, height, width = image.shape
        cy = int(torch.randint(0, height, (1,)).item())
        cx = int(torch.randint(0, width, (1,)).item())
        half = self.length // 2
        y1 = max(0, cy - half)
        y2 = min(height, cy + half)
        x1 = max(0, cx - half)
        x2 = min(width, cx + half)
        image = image.clone()
        image[:, y1:y2, x1:x2] = 0.0
        return image


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _subset(dataset: Dataset, subset_size: Optional[int], seed: int) -> Dataset:
    if subset_size is None or subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[: int(subset_size)].tolist()
    return Subset(dataset, indices)


def build_transforms(train: bool, config: dict[str, Any] | None = None) -> transforms.Compose:
    cfg = config or {}
    ops: list[Any] = []
    if train:
        ops.extend(
            [
                transforms.RandomCrop(32, padding=int(cfg.get("crop_padding", 4))),
                transforms.RandomHorizontalFlip(p=float(cfg.get("horizontal_flip_p", 0.5))),
            ]
        )
        if _as_bool(cfg.get("randaugment", False)):
            ops.append(
                transforms.RandAugment(
                    num_ops=int(cfg.get("randaugment_num_ops", 2)),
                    magnitude=int(cfg.get("randaugment_magnitude", 9)),
                )
            )
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    if train and _as_bool(cfg.get("cutout", False)):
        ops.append(
            Cutout(
                length=int(cfg.get("cutout_length", 16)),
                p=float(cfg.get("cutout_p", 1.0)),
            )
        )
    return transforms.Compose(ops)


def _class_count_dict(targets: list[int], indices: list[int]) -> dict[str, int]:
    counts = Counter(int(targets[i]) for i in indices)
    return {name: int(counts.get(class_idx, 0)) for class_idx, name in enumerate(CIFAR10_CLASSES)}


def _validate_split(split: dict[str, Any], targets: list[int], val_size_per_class: int) -> None:
    train_indices = [int(i) for i in split["train_indices"]]
    val_indices = [int(i) for i in split["val_indices"]]
    if len(set(train_indices).intersection(val_indices)) != 0:
        raise ValueError("train_dev and val_dev split indices overlap")
    if len(train_indices) + len(val_indices) != len(targets):
        raise ValueError("train_dev and val_dev do not cover the official training set")
    expected_val = {name: val_size_per_class for name in CIFAR10_CLASSES}
    expected_train = {
        name: int(5000 - val_size_per_class) for name in CIFAR10_CLASSES
    }
    if _class_count_dict(targets, val_indices) != expected_val:
        raise ValueError("val_dev split is not class-balanced")
    if _class_count_dict(targets, train_indices) != expected_train:
        raise ValueError("train_dev split is not class-balanced")


def get_or_create_cifar10_split(
    data_dir: str = "./data",
    split_file: str | Path = DEFAULT_SPLIT_FILE,
    split_seed: int = 42,
    val_size_per_class: int = 500,
    download: bool = True,
) -> dict[str, Any]:
    split_path = Path(split_file)
    base_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=None,
    )
    targets = [int(t) for t in base_dataset.targets]
    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)
        _validate_split(split, targets, int(val_size_per_class))
        return split

    rng = np.random.default_rng(int(split_seed))
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_idx in range(len(CIFAR10_CLASSES)):
        class_indices = np.asarray(
            [i for i, target in enumerate(targets) if int(target) == class_idx],
            dtype=np.int64,
        )
        rng.shuffle(class_indices)
        val = class_indices[: int(val_size_per_class)].tolist()
        train = class_indices[int(val_size_per_class) :].tolist()
        val_indices.extend(int(i) for i in val)
        train_indices.extend(int(i) for i in train)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    split = {
        "dataset": "CIFAR-10",
        "split_seed": int(split_seed),
        "val_size_per_class": int(val_size_per_class),
        "train_indices": train_indices,
        "val_indices": val_indices,
        "class_counts_train": _class_count_dict(targets, train_indices),
        "class_counts_val": _class_count_dict(targets, val_indices),
    }
    _validate_split(split, targets, int(val_size_per_class))
    ensure_dir(split_path.parent)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, sort_keys=True)
    return split


def _loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "worker_init_fn": worker_seed_fn,
    }
    if shuffle:
        kwargs["generator"] = generator
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **kwargs)


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    subset_size: Optional[int] = None,
    val_subset_size: Optional[int] = None,
    test_subset_size: Optional[int] = None,
    seed: int = 42,
    split_seed: int = 42,
    split_file: str | Path = DEFAULT_SPLIT_FILE,
    val_size_per_class: int = 500,
    train_full: bool = False,
    include_val: bool = True,
    include_test: bool = False,
    train_transform_config: dict[str, Any] | None = None,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    download: bool = True,
) -> DataLoaders:
    train_full_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=build_transforms(train=True, config=train_transform_config),
    )
    eval_train_full_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=build_transforms(train=False),
    )

    split_info: dict[str, Any] | None = None
    val_set: Dataset | None = None
    if train_full:
        train_set: Dataset = train_full_set
        include_val = False
    else:
        split_info = get_or_create_cifar10_split(
            data_dir=data_dir,
            split_file=split_file,
            split_seed=split_seed,
            val_size_per_class=val_size_per_class,
            download=download,
        )
        train_set = Subset(train_full_set, [int(i) for i in split_info["train_indices"]])
        if include_val:
            val_set = Subset(eval_train_full_set, [int(i) for i in split_info["val_indices"]])

    train_set = _subset(train_set, subset_size, seed)
    if val_set is not None:
        val_set = _subset(val_set, val_subset_size, seed + 17)

    train_loader = _loader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        seed=seed,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = (
        _loader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed + 1,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        if val_set is not None
        else None
    )

    test_loader = None
    if include_test:
        test_set: Dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=download,
            transform=build_transforms(train=False),
        )
        test_set = _subset(test_set, test_subset_size, seed + 2)
        test_loader = _loader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed + 2,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    return DataLoaders(train=train_loader, val=val_loader, test=test_loader, split_info=split_info)
