"""Generate figures from completed experiment logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from .models import build_model
from .utils import ensure_dir, load_config


def maybe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"skip missing {path}")
        return None
    return pd.read_csv(path)


def plot_baseline_vs_final(results_dir: Path) -> None:
    logs = results_dir / "logs"
    figures = ensure_dir(results_dir / "figures")
    baseline = maybe_read(logs / "baseline_history.csv")
    final = maybe_read(logs / "final_model_history.csv")
    if baseline is None or final is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for df, label in [(baseline, "SimpleCNN"), (final, "CompactResNet")]:
        axes[0].plot(df["epoch"], df["train_loss"], label=label)
        axes[1].plot(df["epoch"], df["val_acc"], label=label)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "training_curves_baseline_vs_final.png", dpi=220)
    plt.close(fig)


def plot_ablation(results_dir: Path) -> None:
    metrics = results_dir / "metrics" / "ablation_results.csv"
    df = maybe_read(metrics)
    if df is None or df.empty:
        return
    labels = [f"{row.factor}\n{row.setting}" for row in df.itertuples()]
    fig, ax = plt.subplots(figsize=(max(9, len(df) * 1.2), 5.2))
    values = df["best_val_acc"] if "best_val_acc" in df else df["final_val_acc"]
    ax.bar(labels, values, color="#3a7d6b")
    ax.set_ylim(max(0.0, float(values.min()) - 0.05), 1.0)
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Validation Ablation Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / "figures" / "ablation_accuracy_bar.png", dpi=220)
    plt.close(fig)


def find_first_conv(model: nn.Module) -> nn.Conv2d | None:
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    return None


def plot_first_layer_filters(checkpoint_path: Path, output_path: Path) -> None:
    if not checkpoint_path.exists():
        print(f"skip missing {checkpoint_path}")
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint.get("config", {})
    model = build_model(
        str(cfg.get("model", "compact_resnet")),
        channels=cfg.get("channels", [64, 128, 256]),
        activation=str(cfg.get("activation", "silu")),
        dropout=float(cfg.get("dropout", 0.2)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    conv = find_first_conv(model)
    if conv is None:
        return
    weights = conv.weight.detach().cpu()
    n = min(32, weights.shape[0])
    cols = 8
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        filt = weights[i]
        if filt.shape[0] == 3:
            img = filt.permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img)
        else:
            img = filt.mean(dim=0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax.imshow(img, cmap="viridis")
    fig.suptitle("First-layer Convolution Filters", y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_visualize(results_dir: str, checkpoint: str | None = None) -> None:
    root = Path(results_dir)
    ensure_dir(root / "figures")
    plot_baseline_vs_final(root)
    plot_ablation(root)
    checkpoint_path = Path(checkpoint or root / "checkpoints" / "final_model_best_val.pt")
    plot_first_layer_filters(checkpoint_path, root / "figures" / "first_layer_filters.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate figures from logs.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        args.results_dir = cfg.get("results_dir", args.results_dir)
        args.checkpoint = cfg.get("checkpoint", args.checkpoint)
    run_visualize(args.results_dir, args.checkpoint)


if __name__ == "__main__":
    main()
