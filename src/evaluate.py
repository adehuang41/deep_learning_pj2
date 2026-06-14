"""Evaluate validation-selected or locked CIFAR-10 checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn

from .data import DEFAULT_SPLIT_FILE, get_cifar10_loaders
from .models import build_model
from .train import DEFAULT_CONFIG
from .utils import (
    CIFAR10_CLASSES,
    count_parameters,
    ensure_dir,
    get_device,
    load_config,
    merge_config_with_overrides,
    save_json,
    set_seed,
)


def str_to_bool(value: str | bool | None) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def prefixed_path(save_prefix: str, suffix: str) -> Path:
    return Path(f"{save_prefix}{suffix}")


def validate_evaluation_request(config: dict[str, Any]) -> None:
    split = str(config.get("split", "val")).lower()
    official = str_to_bool(config.get("official_final_eval", False))
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split}")
    if split == "test":
        if not official:
            raise ValueError("official_test evaluation requires --official_final_eval true.")
        lock_path = Path(str(config.get("final_selection_lock", "results/protocol/final_selection_lock.json")))
        if not lock_path.exists():
            raise FileNotFoundError(f"official_test evaluation requires final selection lock: {lock_path}")
    if split == "val" and official:
        raise ValueError("--official_final_eval true is only valid with --split test.")


def _to_device(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(device, non_blocking=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    y = y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    channels_last: bool = False,
    horizontal_flip_tta: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    preds: list[int] = []
    targets: list[int] = []
    confidences: list[float] = []
    batch_times: list[float] = []
    for x, y in loader:
        x, y = _to_device(x, y, device, channels_last)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        logits = model(x)
        if horizontal_flip_tta:
            flipped = torch.flip(x, dims=[3])
            logits = 0.5 * (logits + model(flipped))
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_times.append(time.perf_counter() - start)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)
        total_loss += float(loss.item()) * y.size(0)
        total_correct += int((pred == y).sum().item())
        total_seen += int(y.size(0))
        preds.extend(pred.cpu().tolist())
        targets.extend(y.cpu().tolist())
        confidences.extend(confidence.cpu().tolist())
    return {
        "loss": total_loss / max(1, total_seen),
        "accuracy": total_correct / max(1, total_seen),
        "preds": np.asarray(preds),
        "targets": np.asarray(targets),
        "confidences": np.asarray(confidences),
        "inference_time_per_batch": float(np.mean(batch_times)) if batch_times else 0.0,
        "num_examples": int(total_seen),
    }


def plot_confusion(cm: np.ndarray, output_path: str | Path, title: str) -> None:
    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CIFAR10_CLASSES)))
    ax.set_yticks(range(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            color = "white" if value > cm.max() * 0.5 else "black"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_classwise(acc: np.ndarray, output_path: str | Path, title: str) -> None:
    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(CIFAR10_CLASSES, acc, color="#2f6f8f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    config = merge_config_with_overrides(config, {}, DEFAULT_CONFIG)
    validate_evaluation_request(config)
    split = str(config.get("split", "val")).lower()
    official = str_to_bool(config.get("official_final_eval", False))
    horizontal_flip_tta = str_to_bool(config.get("horizontal_flip_tta", False))
    set_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", True)))
    device = get_device(config.get("device"))
    channels_last = bool(config.get("channels_last", False)) and device.type == "cuda"
    checkpoint_path = Path(str(config["checkpoint"]))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_config = checkpoint.get("config", {})
    model_name = str(config.get("model") or ckpt_config.get("model"))
    model = build_model(
        model_name,
        channels=config.get("channels", ckpt_config.get("channels", [64, 128, 256])),
        activation=str(config.get("activation", ckpt_config.get("activation", "silu"))),
        dropout=float(config.get("dropout", ckpt_config.get("dropout", 0.2))),
        blocks_per_stage=config.get("blocks_per_stage", ckpt_config.get("blocks_per_stage")),
        drop_path_rate=config.get("drop_path_rate", ckpt_config.get("drop_path_rate")),
        use_eca=config.get("use_eca", ckpt_config.get("use_eca")),
        depth=config.get("depth", ckpt_config.get("depth")),
        widen_factor=config.get("widen_factor", ckpt_config.get("widen_factor")),
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()
    loaders = get_cifar10_loaders(
        data_dir=str(config.get("data_dir", "./data")),
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=bool(config.get("pin_memory", True)) and device.type == "cuda",
        persistent_workers=bool(config.get("persistent_workers", False)),
        prefetch_factor=config.get("prefetch_factor"),
        subset_size=None,
        val_subset_size=config.get("val_subset_size"),
        test_subset_size=config.get("test_subset_size"),
        seed=int(config.get("seed", 42)),
        split_seed=int(config.get("split_seed", 42)),
        split_file=str(config.get("split_file", DEFAULT_SPLIT_FILE)),
        val_size_per_class=int(config.get("val_size_per_class", 500)),
        train_full=False,
        include_val=split == "val",
        include_test=split == "test",
        train_transform_config=config,
    )
    loader = loaders.test if split == "test" else loaders.val
    if loader is None:
        raise RuntimeError(f"Requested split loader is unavailable: {split}")

    result = collect_predictions(
        model,
        loader,
        criterion,
        device,
        channels_last=channels_last,
        horizontal_flip_tta=horizontal_flip_tta,
    )
    cm = confusion_matrix(result["targets"], result["preds"], labels=list(range(10)))
    class_totals = cm.sum(axis=1).clip(min=1)
    class_acc = cm.diagonal() / class_totals

    save_prefix = str(config.get("save_prefix", f"results/metrics/{split}"))
    prefix_path = Path(save_prefix)
    ensure_dir(prefix_path.parent)
    figures_dir = ensure_dir(config.get("figures_dir", "results/figures"))
    suffix = "official_test" if split == "test" else "val_dev"
    if horizontal_flip_tta:
        suffix = f"{suffix}_flip_tta"
    cm_path = figures_dir / str(config.get("confusion_figure", f"confusion_matrix_{suffix}.png"))
    class_path = figures_dir / str(config.get("classwise_figure", f"classwise_accuracy_{suffix}.png"))
    plot_confusion(cm, cm_path, f"{suffix.replace('_', ' ').title()} Confusion Matrix")
    plot_classwise(class_acc, class_path, f"{suffix.replace('_', ' ').title()} Class-wise Accuracy")

    metrics: dict[str, Any] = {
        "split": split,
        "loss_standard_ce": float(result["loss"]),
        "accuracy": float(result["accuracy"]),
        "error": float(1.0 - result["accuracy"]),
        "parameter_count": count_parameters(model),
        "inference_time_per_batch": float(result["inference_time_per_batch"]),
        "num_examples": int(result["num_examples"]),
        "checkpoint": str(checkpoint_path),
        "selection_metric": checkpoint.get("selection_metric", "validation accuracy"),
        "selected_weights": checkpoint.get("selected_weights", "raw"),
        "official_final_eval": official,
        "evaluated_after_lock": official,
        "horizontal_flip_tta": horizontal_flip_tta,
        "confusion_matrix": cm.tolist(),
        "classwise_accuracy": {
            name: float(value) for name, value in zip(CIFAR10_CLASSES, class_acc)
        },
        "confusion_matrix_figure": str(cm_path),
        "classwise_accuracy_figure": str(class_path),
    }
    if split == "test":
        metrics.update(
            {
                "official_test_loss_standard_ce": metrics["loss_standard_ce"],
                "official_test_accuracy": metrics["accuracy"],
                "official_test_error": metrics["error"],
            }
        )
    else:
        metrics.update(
            {
                "val_loss_standard_ce": metrics["loss_standard_ce"],
                "val_accuracy": metrics["accuracy"],
                "val_error": metrics["error"],
            }
        )
    save_json(metrics, prefixed_path(save_prefix, "_eval.json"))
    pd.DataFrame(
        [
            {
                "split": split,
                "loss_standard_ce": metrics["loss_standard_ce"],
                "accuracy": metrics["accuracy"],
                "error": metrics["error"],
                "parameter_count": metrics["parameter_count"],
                "inference_time_per_batch": metrics["inference_time_per_batch"],
                "horizontal_flip_tta": horizontal_flip_tta,
                "official_final_eval": official,
            }
        ]
    ).to_csv(prefixed_path(save_prefix, "_eval.csv"), index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "compact_resnet", "compact_resnet_v2", "vgg_a", "vgg_a_bn", "preact_resnet_cifar"],
    )
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--split", choices=["val", "test"], default=None)
    parser.add_argument("--official_final_eval", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--horizontal_flip_tta", nargs="?", const=True, default=None, type=str_to_bool)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--save_prefix", type=str)
    parser.add_argument("--figures_dir", type=str)
    parser.add_argument("--val_subset_size", type=int)
    parser.add_argument("--test_subset_size", type=int)
    parser.add_argument("--device", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k != "config" and v is not None
    }
    config = merge_config_with_overrides(config, overrides, DEFAULT_CONFIG)
    metrics = run_evaluation(config)
    printable = dict(metrics)
    printable.pop("confusion_matrix", None)
    printable.pop("classwise_accuracy", None)
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
