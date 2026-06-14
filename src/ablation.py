"""Run a compact ablation matrix for the final CIFAR-10 model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .train import DEFAULT_CONFIG, run_training
from .utils import ensure_dir, load_config, merge_config_with_overrides


DEFAULT_ABLATION: dict[str, Any] = {
    "base": {
        "model": "compact_resnet",
        "epochs": 30,
        "batch_size": 128,
        "optimizer": "sgd",
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "scheduler": "cosine",
        "activation": "silu",
        "dropout": 0.2,
        "channels": [64, 128, 256],
        "seed": 42,
        "save_dir": "results/checkpoints",
        "log_dir": "results/logs",
        "metrics_dir": "results/metrics",
    },
    "experiments": [
        {
            "name": "width_narrow",
            "factor": "Width",
            "setting": "[32,64,128]",
            "channels": [32, 64, 128],
        },
        {
            "name": "width_default",
            "factor": "Width",
            "setting": "[64,128,256]",
            "channels": [64, 128, 256],
        },
        {
            "name": "activation_relu",
            "factor": "Activation",
            "setting": "ReLU",
            "activation": "relu",
        },
        {
            "name": "activation_leaky_relu",
            "factor": "Activation",
            "setting": "LeakyReLU",
            "activation": "leaky_relu",
        },
        {
            "name": "activation_silu",
            "factor": "Activation",
            "setting": "SiLU",
            "activation": "silu",
        },
        {
            "name": "regularization_label_smoothing",
            "factor": "Regularization",
            "setting": "Label smoothing 0.1",
            "loss": "ce_label_smoothing",
            "label_smoothing": 0.1,
        },
        {
            "name": "regularization_dropout",
            "factor": "Regularization",
            "setting": "Dropout 0.2",
            "dropout": 0.2,
        },
        {
            "name": "optimizer_adamw",
            "factor": "Optimizer",
            "setting": "AdamW cosine",
            "optimizer": "adamw",
            "lr": 0.001,
        },
    ],
    "output_csv": "results/metrics/ablation_results.csv",
    "figure": "results/figures/ablation_accuracy_bar.png",
}


def plot_ablation(results: pd.DataFrame, figure_path: str) -> None:
    ensure_dir("results/figures")
    labels = [f"{row.factor}\n{row.setting}" for row in results.itertuples()]
    fig, ax = plt.subplots(figsize=(max(9, len(results) * 1.2), 5.2))
    ax.bar(labels, results["best_val_acc"], color="#3a7d6b")
    ax.set_ylim(max(0.0, results["best_val_acc"].min() - 0.05), 1.0)
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("CompactResNet Validation Ablation Study")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)


def run_ablation(config: dict[str, Any]) -> pd.DataFrame:
    full = dict(DEFAULT_ABLATION)
    full.update(config)
    base = merge_config_with_overrides(full.get("base", {}), {}, DEFAULT_CONFIG)
    rows: list[dict[str, Any]] = []
    for exp in full.get("experiments", []):
        exp_config = dict(base)
        exp_config.update({k: v for k, v in exp.items() if k not in {"name", "factor", "setting"}})
        exp_config["run_name"] = f"ablation_{exp['name']}"
        summary = run_training(exp_config)
        rows.append(
            {
                "experiment": exp["name"],
                "factor": exp.get("factor", ""),
                "setting": exp.get("setting", ""),
                "best_val_acc": summary["best_val_acc"],
                "best_val_error": summary["best_val_error"],
                "best_val_epoch": summary["best_val_epoch"],
                "best_val_loss": summary["best_val_loss"],
                "final_val_acc": summary["final_val_acc"],
                "final_val_loss": summary["final_val_loss"],
                "params": summary["params"],
                "time_seconds": summary["train_time_seconds"],
                "mean_epoch_time": summary["mean_epoch_time"],
                "mean_images_per_second": summary["mean_images_per_second"],
                "checkpoint": summary["best_checkpoint"],
            }
        )
    df = pd.DataFrame(rows)
    output_csv = str(full.get("output_csv", "results/metrics/ablation_results.csv"))
    ensure_dir(Path(output_csv).parent)
    df.to_csv(output_csv, index=False)
    plot_ablation(df, str(full.get("figure", "results/figures/ablation_accuracy_bar.png")))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ablation matrix.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--val_subset_size", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--data_dir", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if "base" not in config:
        config["base"] = {}
    for key in ["epochs", "subset_size", "val_subset_size", "batch_size", "data_dir"]:
        value = getattr(args, key)
        if value is not None:
            config["base"][key] = value
    df = run_ablation(config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
