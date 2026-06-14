"""Batch Normalization training and gradient-geometry analysis."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .data import get_cifar10_loaders
from .models import build_model
from .train import DEFAULT_CONFIG, build_criterion, run_training
from .utils import count_parameters, ensure_dir, get_device, infer_run_name, load_config, merge_config_with_overrides, set_seed


DEFAULT_ANALYSIS: dict[str, Any] = {
    "alpha_list": [1e-3, 5e-3, 1e-2],
    "max_records": 40,
    "train_if_missing": True,
    "force_train": False,
    "output_bn_csv": "results/metrics/bn_comparison.csv",
    "output_predictiveness_csv": "results/metrics/gradient_predictiveness.csv",
    "output_gradient_difference_csv": "results/metrics/gradient_difference.csv",
    "figure_training": "results/figures/bn_training_curves.png",
    "figure_predictiveness": "results/figures/gradient_predictiveness.png",
    "figure_gradient_difference": "results/figures/gradient_difference.png",
}


def checkpoint_path_for(config: dict[str, Any]) -> Path:
    if config.get("checkpoint"):
        return Path(str(config["checkpoint"]))
    save_dir = Path(str(config.get("save_dir", "results/checkpoints")))
    if config.get("best_checkpoint_name"):
        return save_dir / str(config["best_checkpoint_name"])
    return save_dir / f"{infer_run_name(config)}_best_val.pt"


def summary_csv_for(config: dict[str, Any]) -> Path:
    return Path(str(config.get("metrics_dir", "results/metrics"))) / f"{infer_run_name(config)}_results.csv"


def history_csv_for(config: dict[str, Any]) -> Path:
    return Path(str(config.get("log_dir", "results/logs"))) / f"{infer_run_name(config)}_history.csv"


def checkpoint_matches_config(path: Path, config: dict[str, Any]) -> bool:
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception:
        return False
    saved = checkpoint.get("config", {})
    keys = [
        "run_name",
        "model",
        "epochs",
        "subset_size",
        "val_subset_size",
        "batch_size",
        "lr",
        "optimizer",
        "weight_decay",
        "scheduler",
    ]
    for key in keys:
        if saved.get(key) != config.get(key):
            return False
    return True


def ensure_training(config: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    ckpt = checkpoint_path_for(config)
    summary_csv = summary_csv_for(config)
    if (
        ckpt.exists()
        and summary_csv.exists()
        and not bool(analysis.get("force_train", False))
        and checkpoint_matches_config(ckpt, config)
    ):
        return pd.read_csv(summary_csv).iloc[0].to_dict()
    if not bool(analysis.get("train_if_missing", True)):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    return run_training(config)


def plot_bn_training(no_bn_cfg: dict[str, Any], bn_cfg: dict[str, Any], output_path: str) -> None:
    histories = []
    for label, cfg in [("VGG-A", no_bn_cfg), ("VGG-A-BN", bn_cfg)]:
        path = history_csv_for(cfg)
        if path.exists():
            df = pd.read_csv(path)
            df["label"] = label
            histories.append(df)
    if not histories:
        return
    ensure_dir(Path(output_path).parent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for df in histories:
        label = str(df["label"].iloc[0])
        axes[0].plot(df["epoch"], df["train_loss"], label=label)
        axes[1].plot(df["epoch"], df["val_acc"], label=label)
        axes[2].plot(df["epoch"], df["grad_norm"], label=label)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[2].set_title("Gradient Norm")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Norm")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def clone_grads(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    grads: list[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            grads.append(torch.zeros_like(p))
        else:
            grads.append(p.grad.detach().clone())
    return grads


def tensor_list_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    total = tensors[0].new_tensor(0.0)
    for t in tensors:
        total = total + torch.sum(t.detach() * t.detach())
    return torch.sqrt(total)


def dot_lists(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
    total = left[0].new_tensor(0.0)
    for a, b in zip(left, right):
        total = total + torch.sum(a.detach() * b.detach())
    return total


def add_steps(params: list[torch.nn.Parameter], steps: list[torch.Tensor], scale: float = 1.0) -> None:
    with torch.no_grad():
        for p, s in zip(params, steps):
            p.add_(s, alpha=scale)


def measure_gradient_geometry(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_label: str,
    alpha_list: list[float],
    max_records: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    predictiveness_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    iterator = tqdm(loader, desc=f"geometry:{model_label}", total=min(len(loader), max_records))
    for sample_idx, (x, y) in enumerate(iterator):
        if sample_idx >= max_records:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        base_loss = criterion(model(x), y)
        base_loss.backward()
        base_grads = clone_grads(params)
        grad_norm = tensor_list_norm(base_grads)
        if not torch.isfinite(grad_norm) or float(grad_norm.item()) <= 0.0:
            continue
        for alpha in alpha_list:
            steps = [(-float(alpha) / grad_norm) * g for g in base_grads]
            linear_pred = float(dot_lists(base_grads, steps).item())
            add_steps(params, steps, scale=1.0)
            model.zero_grad(set_to_none=True)
            perturbed_loss = criterion(model(x), y)
            actual_delta = float(perturbed_loss.item() - base_loss.item())
            perturbed_loss.backward()
            perturbed_grads = clone_grads(params)
            grad_diff_tensors = [b - a for a, b in zip(base_grads, perturbed_grads)]
            grad_diff = float(tensor_list_norm(grad_diff_tensors).item())
            distance = float(math.sqrt(sum(float(torch.sum(s * s).item()) for s in steps)))
            add_steps(params, steps, scale=-1.0)
            model.zero_grad(set_to_none=True)
            error = abs(actual_delta - linear_pred)
            relative_error = error / (abs(actual_delta) + 1e-12)
            predictiveness_rows.append(
                {
                    "model": model_label,
                    "sample_idx": sample_idx,
                    "alpha": float(alpha),
                    "base_loss": float(base_loss.item()),
                    "actual_delta": actual_delta,
                    "linear_pred_delta": linear_pred,
                    "predictiveness_error": error,
                    "relative_error": relative_error,
                    "grad_norm": float(grad_norm.item()),
                }
            )
            difference_rows.append(
                {
                    "model": model_label,
                    "sample_idx": sample_idx,
                    "alpha": float(alpha),
                    "grad_diff": grad_diff,
                    "distance": distance,
                    "grad_lipschitz_estimate": grad_diff / max(distance, 1e-12),
                }
            )
    return pd.DataFrame(predictiveness_rows), pd.DataFrame(difference_rows)


def load_checkpoint_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    ckpt_path = checkpoint_path_for(config)
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_config = checkpoint.get("config", config)
    model = build_model(
        str(ckpt_config.get("model", config.get("model"))),
        channels=ckpt_config.get("channels", config.get("channels", [64, 128, 256])),
        activation=str(ckpt_config.get("activation", config.get("activation", "relu"))),
        dropout=float(ckpt_config.get("dropout", config.get("dropout", 0.5))),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def plot_geometry(predictiveness: pd.DataFrame, difference: pd.DataFrame, analysis: dict[str, Any]) -> None:
    ensure_dir("results/figures")
    if not predictiveness.empty:
        grouped = predictiveness.groupby(["model", "alpha"], as_index=False)["relative_error"].mean()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, df in grouped.groupby("model"):
            ax.plot(df["alpha"], df["relative_error"], marker="o", label=model)
        ax.set_xscale("log")
        ax.set_xlabel("Step size alpha")
        ax.set_ylabel("Mean relative prediction error")
        ax.set_title("Gradient Predictiveness")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(analysis["figure_predictiveness"]), dpi=220)
        plt.close(fig)
    if not difference.empty:
        grouped = difference.groupby(["model", "alpha"], as_index=False)["grad_lipschitz_estimate"].mean()
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model, df in grouped.groupby("model"):
            ax.plot(df["alpha"], df["grad_lipschitz_estimate"], marker="o", label=model)
        ax.set_xscale("log")
        ax.set_xlabel("Step size alpha")
        ax.set_ylabel("Mean ||g' - g|| / ||w' - w||")
        ax.set_title("Maximum Gradient Difference over Distance")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(str(analysis["figure_gradient_difference"]), dpi=220)
        plt.close(fig)


def run_bn_analysis(
    no_bn_config: dict[str, Any],
    bn_config: dict[str, Any],
    analysis_config: dict[str, Any],
) -> pd.DataFrame:
    analysis = dict(DEFAULT_ANALYSIS)
    analysis.update(analysis_config)
    no_bn_config = merge_config_with_overrides(no_bn_config, {}, DEFAULT_CONFIG)
    bn_config = merge_config_with_overrides(bn_config, {}, DEFAULT_CONFIG)
    set_seed(int(no_bn_config.get("seed", 42)))
    no_bn_summary = ensure_training(no_bn_config, analysis)
    bn_summary = ensure_training(bn_config, analysis)
    plot_bn_training(no_bn_config, bn_config, str(analysis["figure_training"]))

    device = get_device(no_bn_config.get("device"))
    loaders = get_cifar10_loaders(
        data_dir=str(no_bn_config.get("data_dir", "./data")),
        batch_size=int(no_bn_config.get("batch_size", 128)),
        num_workers=int(no_bn_config.get("num_workers", 4)),
        pin_memory=bool(no_bn_config.get("pin_memory", True)) and device.type == "cuda",
        subset_size=None,
        val_subset_size=no_bn_config.get("val_subset_size"),
        include_val=True,
        include_test=False,
        seed=int(no_bn_config.get("seed", 42)),
    )
    criterion = build_criterion(no_bn_config)
    no_bn_model = load_checkpoint_model(no_bn_config, device)
    bn_model = load_checkpoint_model(bn_config, device)
    alpha_list = [float(a) for a in analysis.get("alpha_list", [1e-3, 5e-3, 1e-2])]
    max_records = int(analysis.get("max_records", 40))
    pred_no_bn, diff_no_bn = measure_gradient_geometry(
        no_bn_model, loaders.val, criterion, device, "VGG-A", alpha_list, max_records
    )
    pred_bn, diff_bn = measure_gradient_geometry(
        bn_model, loaders.val, criterion, device, "VGG-A-BN", alpha_list, max_records
    )
    predictiveness = pd.concat([pred_no_bn, pred_bn], ignore_index=True)
    difference = pd.concat([diff_no_bn, diff_bn], ignore_index=True)
    ensure_dir(Path(str(analysis["output_predictiveness_csv"])).parent)
    predictiveness.to_csv(str(analysis["output_predictiveness_csv"]), index=False)
    difference.to_csv(str(analysis["output_gradient_difference_csv"]), index=False)
    plot_geometry(predictiveness, difference, analysis)

    pred_summary = (
        predictiveness.groupby("model")["relative_error"].mean().to_dict()
        if not predictiveness.empty
        else {}
    )
    diff_summary = (
        difference.groupby("model")["grad_lipschitz_estimate"].mean().to_dict()
        if not difference.empty
        else {}
    )
    rows = [
        {
            "model": "VGG-A",
            "params": int(no_bn_summary.get("params", 0)) or count_parameters(no_bn_model),
            "best_val_acc": float(no_bn_summary.get("best_val_acc", np.nan)),
            "final_val_acc": float(no_bn_summary.get("final_val_acc", np.nan)),
            "final_val_loss": float(no_bn_summary.get("final_val_loss", np.nan)),
            "train_time_seconds": float(no_bn_summary.get("train_time_seconds", np.nan)),
            "mean_gradient_predictiveness_relative_error": float(pred_summary.get("VGG-A", np.nan)),
            "mean_grad_diff_over_distance": float(diff_summary.get("VGG-A", np.nan)),
            "checkpoint": str(checkpoint_path_for(no_bn_config)),
        },
        {
            "model": "VGG-A-BN",
            "params": int(bn_summary.get("params", 0)) or count_parameters(bn_model),
            "best_val_acc": float(bn_summary.get("best_val_acc", np.nan)),
            "final_val_acc": float(bn_summary.get("final_val_acc", np.nan)),
            "final_val_loss": float(bn_summary.get("final_val_loss", np.nan)),
            "train_time_seconds": float(bn_summary.get("train_time_seconds", np.nan)),
            "mean_gradient_predictiveness_relative_error": float(pred_summary.get("VGG-A-BN", np.nan)),
            "mean_grad_diff_over_distance": float(diff_summary.get("VGG-A-BN", np.nan)),
            "checkpoint": str(checkpoint_path_for(bn_config)),
        },
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(str(analysis["output_bn_csv"]), index=False)
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGG-A BN comparison and gradient analysis.")
    parser.add_argument("--config", type=str, required=True, help="VGG-A without BN config")
    parser.add_argument("--config_bn", type=str, required=True, help="VGG-A with BN config")
    parser.add_argument("--analysis_config", type=str, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--val_subset_size", type=int)
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--force_train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    no_bn_config = load_config(args.config)
    bn_config = load_config(args.config_bn)
    analysis = load_config(args.analysis_config)
    for cfg in (no_bn_config, bn_config):
        if args.epochs is not None:
            cfg["epochs"] = args.epochs
        if args.subset_size is not None:
            cfg["subset_size"] = args.subset_size
        if args.val_subset_size is not None:
            cfg["val_subset_size"] = args.val_subset_size
    if args.max_records is not None:
        analysis["max_records"] = args.max_records
    if args.force_train:
        analysis["force_train"] = True
    comparison = run_bn_analysis(no_bn_config, bn_config, analysis)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
