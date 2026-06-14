"""Final official CIFAR-10 test evaluation locked by final_selection_lock.json."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets

from .data import build_transforms
from .evaluate import collect_predictions, plot_classwise, plot_confusion
from .models import build_model
from .utils import CIFAR10_CLASSES, count_parameters, ensure_dir, get_device, save_json, set_seed


LOCK_PATH = Path("results/protocol/final_selection_lock.json")
OUTPUT_CSV = Path("results/metrics/final_official_test_results.csv")
OUTPUT_JSON = Path("results/metrics/final_official_test_results.json")
PROTOCOL_JSON = Path("results/protocol/final_official_test_run_protocol.json")
FINAL_W4_CONFUSION = Path("results/figures/final_w4_official_confusion_matrix.png")
FINAL_W4_CLASSWISE = Path("results/figures/final_w4_official_classwise_accuracy.png")
FINAL_W4_MISCLASSIFIED = Path("results/figures/final_w4_official_misclassified_examples.png")

EXPECTED_ROWS = [
    ("SimpleCNN baseline", "results/checkpoints/baseline_best_val.pt"),
    ("VGG-A no BN", "results/checkpoints/bn_p0_vgg_a_best_val.pt"),
    ("VGG-A BN", "results/checkpoints/bn_p0_vgg_a_bn_best_val.pt"),
    ("Final W4 full-train champion", "results/checkpoints/final_w4_fulltrain_raw.pt"),
]


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_lock(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        lock = json.load(f)
    rows = lock.get("official_test_rows", [])
    if len(rows) != 4:
        raise ValueError(f"Expected exactly 4 locked official-test rows, got {len(rows)}.")
    actual = [(str(row["name"]), str(row["checkpoint"])) for row in rows]
    if actual != EXPECTED_ROWS:
        raise ValueError(f"Locked rows differ from expected rows: {actual}")
    forbidden = {"w6", "compactresnetv2", "compact_resnet_v2"}
    row_text = json.dumps(rows).lower()
    if any(token in row_text for token in forbidden):
        raise ValueError("Forbidden validation-stage model appears in official-test rows.")
    if str(lock.get("tta_policy")) != "no-TTA":
        raise ValueError("Final official test requires no-TTA policy.")
    if "raw" not in str(lock.get("ema_policy", "")).lower():
        raise ValueError("Final official test requires raw weights / no EMA policy.")
    if not bool(lock.get("will_not_tune_after_official_test", False)):
        raise ValueError("Lock must commit to no tuning after official test.")
    return lock


def _build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = checkpoint.get("config", {})
    model = build_model(
        str(cfg.get("model")),
        num_classes=10,
        channels=cfg.get("channels", [64, 128, 256]),
        activation=str(cfg.get("activation", "silu")),
        dropout=float(cfg.get("dropout", 0.2)),
        blocks_per_stage=cfg.get("blocks_per_stage"),
        drop_path_rate=cfg.get("drop_path_rate"),
        use_eca=cfg.get("use_eca"),
        depth=cfg.get("depth"),
        widen_factor=cfg.get("widen_factor"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_final_official_test(lock_path: Path = LOCK_PATH) -> dict[str, Any]:
    lock = _load_lock(lock_path)
    set_seed(42, deterministic=True)
    device = get_device(None)
    ensure_dir(OUTPUT_CSV.parent)
    ensure_dir(OUTPUT_JSON.parent)
    ensure_dir(FINAL_W4_CONFUSION.parent)
    ensure_dir(PROTOCOL_JSON.parent)

    rows = lock["official_test_rows"]
    lock_hash = _sha256(lock_path)
    checkpoint_hashes = {row["checkpoint"]: _sha256(row["checkpoint"]) for row in rows}
    pre_run = {
        "lock_path": str(lock_path),
        "lock_sha256": lock_hash,
        "checkpoint_hashes": checkpoint_hashes,
        "official_test_rows": rows,
        "tta_policy": "no-TTA",
        "ema_policy": "no EMA / raw weights",
        "post_test_tuning_allowed": False,
    }
    save_json(
        {
            "status": "running",
            "pre_run": pre_run,
            "official_test_clean_eval_transform": True,
            "only_locked_rows": True,
            "evaluates_w6": False,
            "evaluates_compact_resnet_v2": False,
        },
        PROTOCOL_JSON,
    )

    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=build_transforms(train=False),
    )
    loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    criterion = nn.CrossEntropyLoss()

    result_rows: list[dict[str, Any]] = []
    json_rows: list[dict[str, Any]] = []
    for row in rows:
        checkpoint_path = Path(row["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if str(checkpoint.get("selected_weights", "raw")).lower() != "raw":
            raise ValueError(f"Checkpoint is not raw selected weights: {checkpoint_path}")
        model = _build_model_from_checkpoint(checkpoint, device)
        result = collect_predictions(
            model,
            loader,
            criterion,
            device,
            channels_last=False,
            horizontal_flip_tta=False,
        )
        cm = confusion_matrix(result["targets"], result["preds"], labels=list(range(10)))
        class_totals = cm.sum(axis=1).clip(min=1)
        class_acc = cm.diagonal() / class_totals
        is_final_w4 = row["name"] == "Final W4 full-train champion"
        confusion_path = None
        classwise_path = None
        if is_final_w4:
            plot_confusion(cm, FINAL_W4_CONFUSION, "Final W4 Official Test Confusion Matrix")
            plot_classwise(class_acc, FINAL_W4_CLASSWISE, "Final W4 Official Test Class-wise Accuracy")
            confusion_path = str(FINAL_W4_CONFUSION)
            classwise_path = str(FINAL_W4_CLASSWISE)

        metrics = {
            "row": int(row["row"]),
            "name": row["name"],
            "role": row["role"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hashes[str(checkpoint_path)],
            "source": row["source"],
            "official_test_loss": float(result["loss"]),
            "official_test_accuracy": float(result["accuracy"]),
            "official_test_error": float(1.0 - result["accuracy"]),
            "num_examples": int(result["num_examples"]),
            "parameter_count": count_parameters(model),
            "tta": "no-TTA",
            "weights": "raw",
            "ema_used": False,
            "official_test_clean_eval_transform": True,
            "locked_before_test": True,
            "post_test_tuning_allowed": False,
            "confusion_matrix_figure": confusion_path,
            "classwise_accuracy_figure": classwise_path,
            "misclassified_examples_figure": None,
        }
        result_rows.append(metrics)
        json_metrics = dict(metrics)
        json_metrics["confusion_matrix"] = cm.tolist()
        json_metrics["classwise_accuracy"] = {
            name: float(value) for name, value in zip(CIFAR10_CLASSES, class_acc)
        }
        json_rows.append(json_metrics)

    pd.DataFrame(result_rows).to_csv(OUTPUT_CSV, index=False)
    payload = {
        "pre_run": pre_run,
        "results": json_rows,
        "outputs": {
            "csv": str(OUTPUT_CSV),
            "json": str(OUTPUT_JSON),
            "final_w4_confusion_matrix": str(FINAL_W4_CONFUSION),
            "final_w4_classwise_accuracy": str(FINAL_W4_CLASSWISE),
            "final_w4_misclassified_examples": None,
        },
        "official_test_rows_count": len(json_rows),
        "only_locked_rows": True,
        "evaluated_w6": False,
        "evaluated_compact_resnet_v2": False,
        "tta_used": False,
        "ema_used": False,
        "post_test_tuning_performed": False,
        "will_not_tune_after_official_test": True,
    }
    save_json(payload, OUTPUT_JSON)
    save_json(
        {
            "status": "complete",
            "pre_run": pre_run,
            "outputs": payload["outputs"],
            "official_test_rows_count": len(json_rows),
            "only_locked_rows": True,
            "evaluated_w6": False,
            "evaluated_compact_resnet_v2": False,
            "tta_used": False,
            "ema_used": False,
            "post_test_tuning_performed": False,
        },
        PROTOCOL_JSON,
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked final official CIFAR-10 test evaluation.")
    parser.add_argument("--lock", type=Path, default=LOCK_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_final_official_test(args.lock)
    printable = [
        {
            "row": row["row"],
            "name": row["name"],
            "official_test_accuracy": row["official_test_accuracy"],
            "official_test_error": row["official_test_error"],
            "official_test_loss": row["official_test_loss"],
        }
        for row in payload["results"]
    ]
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
