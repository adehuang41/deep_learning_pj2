"""Loss-envelope analysis for VGG-A with and without BatchNorm."""

from __future__ import annotations

import argparse
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
from .train import DEFAULT_CONFIG, build_criterion
from .utils import ensure_dir, get_device, load_config, merge_config_with_overrides, set_seed


DEFAULT_LANDSCAPE: dict[str, Any] = {
    "data_dir": "./data",
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "subset_size": None,
    "test_subset_size": None,
    "epochs": 20,
    "seed": 42,
    "models": ["vgg_a", "vgg_a_bn"],
    "learning_rates": [1e-4, 5e-4, 1e-3, 2e-3],
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "max_steps": None,
    "metrics_dir": "results/metrics",
    "figure": "results/figures/bn_loss_landscape_envelope.png",
    "summary_csv": "results/metrics/loss_landscape_summary.csv",
}


def moving_average(values: np.ndarray, window: int = 20) -> np.ndarray:
    if values.size < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def train_step_loss_curve(config: dict[str, Any], model_name: str, lr: float) -> list[float]:
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device"))
    loaders = get_cifar10_loaders(
        data_dir=str(config.get("data_dir", "./data")),
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=bool(config.get("pin_memory", True)) and device.type == "cuda",
        subset_size=config.get("subset_size"),
        test_subset_size=config.get("test_subset_size"),
        seed=int(config.get("seed", 42)),
    )
    model = build_model(model_name, dropout=float(config.get("dropout", 0.5))).to(device)
    criterion = build_criterion(config)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=float(config.get("momentum", 0.9)),
        weight_decay=float(config.get("weight_decay", 5e-4)),
    )
    losses: list[float] = []
    max_steps = config.get("max_steps")
    pbar = tqdm(range(1, int(config.get("epochs", 20)) + 1), desc=f"landscape:{model_name}:{lr:g}")
    for _epoch in pbar:
        model.train()
        for x, y in loaders.train:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            losses.append(float(loss.item()))
            loss.backward()
            optimizer.step()
            if max_steps is not None and len(losses) >= int(max_steps):
                return losses
    return losses


def envelope(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min(len(c) for c in curves)
    stacked = np.stack([c[:min_len] for c in curves], axis=0)
    return stacked.min(axis=0), stacked.max(axis=0), stacked.mean(axis=0)


def run_loss_landscape(config: dict[str, Any]) -> pd.DataFrame:
    cfg = dict(DEFAULT_LANDSCAPE)
    cfg.update(config)
    cfg = merge_config_with_overrides(cfg, {}, DEFAULT_CONFIG)
    metrics_dir = ensure_dir(cfg.get("metrics_dir", "results/metrics"))
    all_envelopes: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for model_name in cfg.get("models", ["vgg_a", "vgg_a_bn"]):
        curves = []
        for lr in cfg.get("learning_rates", [1e-4, 5e-4, 1e-3, 2e-3]):
            curve = np.asarray(train_step_loss_curve(cfg, str(model_name), float(lr)), dtype=np.float64)
            curves.append(curve)
            safe_lr = str(lr).replace(".", "p").replace("-", "m")
            pd.DataFrame({"step": np.arange(len(curve)), "loss": curve}).to_csv(
                metrics_dir / f"loss_landscape_{model_name}_lr{safe_lr}.csv",
                index=False,
            )
        min_curve, max_curve, mean_curve = envelope(curves)
        all_envelopes[str(model_name)] = (min_curve, max_curve, mean_curve)
        width = max_curve - min_curve
        rows.append(
            {
                "model": str(model_name),
                "mean_envelope_width": float(width.mean()),
                "median_envelope_width": float(np.median(width)),
                "final_envelope_width": float(width[-1]),
                "area_under_envelope": float(np.trapz(width)),
                "steps": int(len(width)),
            }
        )

    ensure_dir(Path(str(cfg["figure"])).parent)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    styles = {
        "vgg_a": ("#b3564b", "VGG-A"),
        "vgg_a_bn": ("#2f6f8f", "VGG-A-BN"),
    }
    for model_name, (min_curve, max_curve, mean_curve) in all_envelopes.items():
        color, label = styles.get(model_name, ("#444444", model_name))
        steps = np.arange(len(mean_curve))
        min_s = moving_average(min_curve)
        max_s = moving_average(max_curve)
        mean_s = moving_average(mean_curve)
        ax.fill_between(steps, min_s, max_s, color=color, alpha=0.18)
        ax.plot(steps, mean_s, color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Loss Landscape Envelope across Learning Rates")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(cfg["figure"]), dpi=220)
    plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(str(cfg.get("summary_csv", "results/metrics/loss_landscape_summary.csv")), index=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGG-A loss-envelope analysis.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--max_steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for key in ["epochs", "subset_size", "max_steps"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    summary = run_loss_landscape(config)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

