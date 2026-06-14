"""Check the CIFAR-10 train_dev/val_dev split protocol."""

from __future__ import annotations

from src.data import DEFAULT_SPLIT_FILE, get_cifar10_loaders, get_or_create_cifar10_split
from src.utils import CIFAR10_CLASSES


def main() -> None:
    split = get_or_create_cifar10_split(
        data_dir="./data",
        split_file=DEFAULT_SPLIT_FILE,
        split_seed=42,
        val_size_per_class=500,
    )
    assert len(split["train_indices"]) == 45000
    assert len(split["val_indices"]) == 5000
    assert set(split["train_indices"]).isdisjoint(set(split["val_indices"]))
    assert split["class_counts_train"] == {name: 4500 for name in CIFAR10_CLASSES}
    assert split["class_counts_val"] == {name: 500 for name in CIFAR10_CLASSES}

    loaders = get_cifar10_loaders(
        data_dir="./data",
        batch_size=64,
        num_workers=0,
        pin_memory=False,
        include_val=True,
        include_test=False,
    )
    assert len(loaders.train.dataset) == 45000
    assert loaders.val is not None
    assert len(loaders.val.dataset) == 5000
    assert loaders.test is None
    train_transform = loaders.train.dataset.dataset.transform
    val_transform = loaders.val.dataset.dataset.transform
    assert train_transform is not val_transform


if __name__ == "__main__":
    main()
