"""Bounded BatchNorm P0 mini-study for VGG-A on CIFAR-10.

This runner is intentionally separate from the older BN scripts so the Stage D
protocol is auditable: train_dev/val_dev only, shared initialization, fixed
snapshots, and no official test access.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from .data import (
    CIFAR10_MEAN,
    CIFAR10_STD,
    DEFAULT_SPLIT_FILE,
    build_transforms,
    get_cifar10_loaders,
    get_or_create_cifar10_split,
)
from .models import build_model
from .train import (
    DEFAULT_CONFIG,
    build_criterion,
    build_optimizer,
    build_scheduler,
    checkpoint_payload,
    evaluate_model,
    train_one_epoch,
)
from .utils import (
    accuracy_from_logits,
    count_parameters,
    current_lr,
    ensure_dir,
    epoch_time,
    save_json,
    set_seed,
    worker_seed_fn,
)


MODEL_SPECS = [
    ("vgg_a", "VGG-A"),
    ("vgg_a_bn", "VGG-A-BN"),
]
SNAPSHOT_EPOCHS = [0, 1, 5, 10, 20, 30, 50]
REQUIRED_LRS = [1e-4, 5e-4, 1e-3, 2e-3]
ROBUSTNESS_LRS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
SCALE_FACTORS = [0.25, 0.5, 1.0, 2.0, 4.0]
DIRECTION_ALPHA_GRID = [-1e-2, -5e-3, -2e-3, -1e-3, 0.0, 1e-3, 2e-3, 5e-3, 1e-2]
PREDICT_ALPHA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
EPS = 1e-12


def bn_p0_config(model_name: str, run_name: str, epochs: int, lr: float, args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(
        {
            "model": model_name,
            "run_name": run_name,
            "data_dir": "./data",
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "pin_memory": True,
            "persistent_workers": int(args.num_workers) > 0,
            "prefetch_factor": 4 if int(args.num_workers) > 0 else None,
            "epochs": int(epochs),
            "optimizer": "sgd",
            "lr": float(lr),
            "momentum": 0.9,
            "nesterov": False,
            "weight_decay": 5e-4,
            "scheduler": "cosine",
            "loss": "ce",
            "dropout": 0.5,
            "seed": int(args.seed),
            "deterministic": True,
            "cudnn_benchmark": False,
            "train_split": "dev",
            "checkpoint_selection_split": "val",
            "use_test_during_training": False,
            "split_seed": 42,
            "split_file": DEFAULT_SPLIT_FILE,
            "save_dir": "results/checkpoints",
            "log_dir": "results/logs",
            "metrics_dir": "results/metrics",
            "protocol_dir": "results/protocol",
            "device": args.device,
            "amp": False,
            "channels_last": False,
            "subset_size": args.subset_size,
            "val_subset_size": args.val_subset_size,
        }
    )
    return cfg


def get_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv_layers(model: nn.Module) -> list[nn.Conv2d]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def linear_layers(model: nn.Module) -> list[nn.Linear]:
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def bn_layers(model: nn.Module) -> list[nn.BatchNorm2d]:
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]


def create_shared_initial_states(seed: int, dropout: float = 0.5) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    no_bn = build_model("vgg_a", dropout=dropout)
    bn = build_model("vgg_a_bn", dropout=dropout)

    copied_conv = 0
    for left, right in zip(conv_layers(no_bn), conv_layers(bn)):
        if left.weight.shape == right.weight.shape:
            right.weight.data.copy_(left.weight.data)
            copied_conv += 1

    copied_linear = 0
    copied_linear_bias = 0
    for left, right in zip(linear_layers(no_bn), linear_layers(bn)):
        if left.weight.shape == right.weight.shape:
            right.weight.data.copy_(left.weight.data)
            copied_linear += 1
        if left.bias is not None and right.bias is not None and left.bias.shape == right.bias.shape:
            right.bias.data.copy_(left.bias.data)
            copied_linear_bias += 1

    initialized_bn = 0
    for layer in bn_layers(bn):
        layer.weight.data.fill_(1.0)
        layer.bias.data.zero_()
        layer.running_mean.zero_()
        layer.running_var.fill_(1.0)
        initialized_bn += 1

    metadata = {
        "seed": int(seed),
        "conv_weights_copied": copied_conv,
        "linear_weights_copied": copied_linear,
        "linear_biases_copied": copied_linear_bias,
        "bn_layers_initialized": initialized_bn,
        "bn_gamma": 1.0,
        "bn_beta": 0.0,
        "bn_running_mean": 0.0,
        "bn_running_var": 1.0,
        "official_test_used": False,
    }
    return no_bn.state_dict(), bn.state_dict(), metadata


def load_initial_model(model_name: str, initial_states: dict[str, dict[str, Any]], device: torch.device) -> nn.Module:
    model = build_model(model_name, dropout=0.5).to(device)
    model.load_state_dict(initial_states[model_name])
    return model


def make_train_val_loaders(config: dict[str, Any], device: torch.device):
    loaders = get_cifar10_loaders(
        data_dir=str(config.get("data_dir", "./data")),
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=bool(config.get("pin_memory", True)) and device.type == "cuda",
        persistent_workers=bool(config.get("persistent_workers", False)),
        prefetch_factor=config.get("prefetch_factor"),
        subset_size=config.get("subset_size"),
        val_subset_size=config.get("val_subset_size"),
        seed=int(config.get("seed", 42)),
        split_seed=int(config.get("split_seed", 42)),
        split_file=str(config.get("split_file", DEFAULT_SPLIT_FILE)),
        train_full=False,
        include_val=True,
        include_test=False,
        train_transform_config=config,
    )
    if loaders.test is not None:
        raise RuntimeError("BN P0 training unexpectedly received official test loader.")
    if loaders.val is None:
        raise RuntimeError("BN P0 training requires val_dev loader.")
    return loaders


def save_snapshot(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_val_acc: float | None,
    best_val_loss: float | None,
    config: dict[str, Any],
    global_step: int,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        checkpoint_payload(
            model,
            optimizer,
            int(epoch),
            best_val_acc,
            best_val_loss,
            config,
            selected_weights="raw",
            scheduler=scheduler,
            scaler=None,
            ema=None,
            global_step=global_step,
        ),
        path,
    )


def run_controlled_training(
    model_name: str,
    label: str,
    initial_states: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_name = f"bn_p0_{model_name}"
    config = bn_p0_config(model_name, run_name, args.controlled_epochs, args.controlled_lr, args)
    set_seed(int(args.seed))
    device = get_device(args.device)
    loaders = make_train_val_loaders(config, device)
    model = load_initial_model(model_name, initial_states, device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    save_dir = ensure_dir("results/checkpoints")
    log_dir = ensure_dir("results/logs")
    metrics_dir = ensure_dir("results/metrics")

    snapshot_paths: dict[int, str] = {}
    snapshot_path = save_dir / f"{run_name}_snapshot_epoch0.pt"
    save_snapshot(snapshot_path, model, optimizer, scheduler, 0, None, None, config, 0)
    snapshot_paths[0] = str(snapshot_path)

    rows: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_val_epoch = 0
    best_path = save_dir / f"{run_name}_best_val.pt"
    last_path = save_dir / f"{run_name}_last.pt"
    global_step = 0
    start = time.perf_counter()
    device_for_mem = device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_for_mem)

    snapshot_epoch_set = {e for e in SNAPSHOT_EPOCHS if e <= int(args.controlled_epochs)}
    pbar = tqdm(range(1, int(args.controlled_epochs) + 1), desc=f"bn-control:{label}", unit="epoch")
    for epoch in pbar:
        epoch_start = time.perf_counter()
        lr_start = current_lr(optimizer)
        train_loss, train_acc, grad_norm = train_one_epoch(model, loaders.train, criterion, optimizer, device)
        global_step += len(loaders.train)
        val_loss, val_acc = evaluate_model(model, loaders.val, criterion, device)
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)
            best_val_epoch = int(epoch)
            save_snapshot(best_path, model, optimizer, scheduler, epoch, best_val_acc, best_val_loss, config, global_step)
        elapsed = epoch_time(epoch_start)
        if scheduler is not None:
            scheduler.step()
        lr_end = current_lr(optimizer)
        peak_memory = float(torch.cuda.max_memory_allocated(device_for_mem) / (1024**3)) if device.type == "cuda" else 0.0
        rows.append(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "model": label,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr_start": float(lr_start),
                "lr_end": float(lr_end),
                "grad_norm": float(grad_norm),
                "epoch_time": float(elapsed),
                "images_per_second": float(len(loaders.train.dataset) / max(elapsed, EPS)),
                "peak_gpu_memory": peak_memory,
            }
        )
        if int(epoch) in snapshot_epoch_set:
            snapshot_path = save_dir / f"{run_name}_snapshot_epoch{epoch}.pt"
            save_snapshot(snapshot_path, model, optimizer, scheduler, epoch, best_val_acc, best_val_loss, config, global_step)
            snapshot_paths[int(epoch)] = str(snapshot_path)
        save_snapshot(last_path, model, optimizer, scheduler, epoch, best_val_acc, best_val_loss, config, global_step)
        pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")

    history = pd.DataFrame(rows)
    history_path = log_dir / f"{run_name}_history.csv"
    history.to_csv(history_path, index=False)
    summary = {
        "model_name": model_name,
        "model": label,
        "run_name": run_name,
        "params": count_parameters(model),
        "epochs": int(args.controlled_epochs),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_val_epoch": int(best_val_epoch),
        "final_val_acc": float(history["val_acc"].iloc[-1]),
        "final_val_loss": float(history["val_loss"].iloc[-1]),
        "mean_epoch_time": float(history["epoch_time"].mean()),
        "images_per_second": float(history["images_per_second"].mean()),
        "peak_gpu_memory": float(history["peak_gpu_memory"].max()),
        "total_train_time_seconds": float(time.perf_counter() - start),
        "checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "history_csv": str(history_path),
        "snapshot_paths": snapshot_paths,
        "official_test_used": False,
    }
    pd.DataFrame([summary]).drop(columns=["snapshot_paths"]).to_csv(
        metrics_dir / f"{run_name}_results.csv",
        index=False,
    )
    return summary


def plot_training_dashboard(summaries: list[dict[str, Any]]) -> None:
    histories = []
    for summary in summaries:
        df = pd.read_csv(summary["history_csv"])
        df["model"] = summary["model"]
        histories.append(df)
    all_history = pd.concat(histories, ignore_index=True)
    ensure_dir("results/figures")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("train_loss", "Train Loss"),
        ("val_acc", "Validation Accuracy"),
        ("grad_norm", "Gradient Norm"),
        ("lr_end", "Learning Rate"),
    ]
    for ax, (column, title) in zip(axes.ravel(), panels):
        for model, df in all_history.groupby("model"):
            ax.plot(df["epoch"], df[column], label=model, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/bn_training_dynamics_dashboard.png", dpi=220)
    plt.close(fig)


def run_controlled_comparison(initial_states: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    summaries = []
    for model_name, label in MODEL_SPECS:
        summaries.append(run_controlled_training(model_name, label, initial_states, args))
    comparison = pd.DataFrame([{k: v for k, v in row.items() if k != "snapshot_paths"} for row in summaries])
    comparison.to_csv("results/metrics/bn_controlled_vgga_comparison.csv", index=False)
    plot_training_dashboard(summaries)
    save_json(
        {
            "stage": "BN-P0-controlled-comparison",
            "train_split": "train_dev",
            "selection_split": "val_dev",
            "checkpoint_selection": "best validation accuracy, tie-break lower validation loss",
            "official_test_used": False,
            "summaries": [{k: v for k, v in row.items() if k != "snapshot_paths"} for row in summaries],
        },
        "results/protocol/bn_controlled_vgga_comparison_protocol.json",
    )
    return summaries


def train_lr_curve(
    model_name: str,
    label: str,
    initial_states: dict[str, dict[str, Any]],
    base_lr: float,
    epochs: int,
    args: argparse.Namespace,
    stress_test: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_name = f"bn_p0_{model_name}_lr_{base_lr:g}".replace(".", "p").replace("-", "m")
    config = bn_p0_config(model_name, run_name, epochs, base_lr, args)
    set_seed(int(args.seed))
    device = get_device(args.device)
    loaders = make_train_val_loaders(config, device)
    model = load_initial_model(model_name, initial_states, device)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    rows: list[dict[str, Any]] = []
    step = 0
    diverged = False
    divergence_reason = "none"
    max_loss = -float("inf")
    final_loss = float("nan")
    final_grad_norm = float("nan")
    pbar = tqdm(range(1, int(epochs) + 1), desc=f"lr:{label}:{base_lr:g}", unit="epoch")
    for epoch in pbar:
        model.train()
        for batch_idx, (x, y) in enumerate(loaders.train):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                diverged = True
                divergence_reason = "non_finite_loss"
                break
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
            if not torch.isfinite(grad_norm):
                diverged = True
                divergence_reason = "non_finite_grad_norm"
                break
            optimizer.step()
            step += 1
            final_loss = float(loss.item())
            final_grad_norm = float(grad_norm)
            max_loss = max(max_loss, final_loss)
            if stress_test and step > 50 and final_loss > 50.0:
                diverged = True
                divergence_reason = "loss_exceeded_50"
            if stress_test and final_grad_norm > 1e5:
                diverged = True
                divergence_reason = "grad_norm_exceeded_1e5"
            rows.append(
                {
                    "model": label,
                    "model_name": model_name,
                    "base_lr": float(base_lr),
                    "epoch": int(epoch),
                    "step": int(step),
                    "batch_idx": int(batch_idx),
                    "lr": current_lr(optimizer),
                    "train_loss": final_loss,
                    "grad_norm": final_grad_norm,
                    "stress_test": bool(stress_test),
                    "diverged_so_far": bool(diverged),
                    "divergence_reason": divergence_reason,
                }
            )
            if diverged:
                break
        if scheduler is not None:
            scheduler.step()
        if diverged:
            break
    final_val_loss, final_val_acc = evaluate_model(model, loaders.val, criterion, device)
    stable = not diverged and math.isfinite(final_loss) and max_loss < 50.0 and math.isfinite(final_val_loss)
    summary = {
        "model": label,
        "model_name": model_name,
        "base_lr": float(base_lr),
        "epochs_budget": int(epochs),
        "steps_completed": int(step),
        "final_train_loss": float(final_loss),
        "max_train_loss": float(max_loss),
        "final_grad_norm": float(final_grad_norm),
        "final_val_loss": float(final_val_loss),
        "final_val_acc": float(final_val_acc),
        "stable": bool(stable),
        "diverged": bool(diverged),
        "divergence_reason": divergence_reason,
    }
    return pd.DataFrame(rows), summary


def run_required_loss_envelope(initial_states: dict[str, dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    all_steps = []
    for model_name, label in MODEL_SPECS:
        for lr in REQUIRED_LRS:
            steps, _summary = train_lr_curve(
                model_name,
                label,
                initial_states,
                lr,
                int(args.required_epochs),
                args,
                stress_test=False,
            )
            all_steps.append(steps)
    raw = pd.concat(all_steps, ignore_index=True)
    raw.to_csv("results/metrics/bn_required_loss_envelope_steps.csv", index=False)
    rows = []
    for model, df_model in raw.groupby("model"):
        grouped = df_model.groupby("step")["train_loss"]
        env = grouped.agg(["min", "max", "mean"]).reset_index()
        env["model"] = model
        env["envelope_width"] = env["max"] - env["min"]
        env["base_lr_count"] = df_model["base_lr"].nunique()
        rows.append(env)
    envelope = pd.concat(rows, ignore_index=True)
    envelope = envelope.rename(columns={"min": "min_loss", "max": "max_loss", "mean": "mean_loss"})
    envelope.to_csv("results/metrics/bn_required_loss_envelope.csv", index=False)
    summary = (
        envelope.groupby("model")
        .agg(
            mean_envelope_width=("envelope_width", "mean"),
            median_envelope_width=("envelope_width", "median"),
            final_envelope_width=("envelope_width", "last"),
            max_envelope_width=("envelope_width", "max"),
            steps=("step", "max"),
        )
        .reset_index()
    )
    summary.to_csv("results/metrics/bn_required_loss_envelope_summary.csv", index=False)
    plot_required_envelope(envelope)
    return envelope


def plot_required_envelope(envelope: pd.DataFrame) -> None:
    ensure_dir("results/figures")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"VGG-A": "#9b4d48", "VGG-A-BN": "#2c6f87"}
    for model, df in envelope.groupby("model"):
        color = colors.get(model, "#444444")
        df = df.sort_values("step")
        ax.fill_between(df["step"].to_numpy(), df["min_loss"].to_numpy(), df["max_loss"].to_numpy(), alpha=0.18, color=color)
        ax.plot(df["step"], df["mean_loss"], label=model, linewidth=1.8, color=color)
    ax.set_title("Required LR Loss Envelope")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/bn_required_loss_envelope.png", dpi=220)
    plt.close(fig)


def run_lr_robustness(initial_states: dict[str, dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    summaries = []
    for model_name, label in MODEL_SPECS:
        for lr in ROBUSTNESS_LRS:
            _steps, summary = train_lr_curve(
                model_name,
                label,
                initial_states,
                lr,
                int(args.stress_epochs),
                args,
                stress_test=True,
            )
            summaries.append(summary)
    df = pd.DataFrame(summaries)
    largest = {}
    for model, model_df in df.groupby("model"):
        stable_lrs = model_df.loc[model_df["stable"], "base_lr"]
        largest[model] = float(stable_lrs.max()) if not stable_lrs.empty else float("nan")
    df["largest_stable_lr"] = df["model"].map(largest)
    df["stress_test_type"] = "short_lr_stress_test"
    df.to_csv("results/metrics/bn_lr_robustness.csv", index=False)
    plot_lr_heatmap(df)
    return df


def plot_lr_heatmap(df: pd.DataFrame) -> None:
    ensure_dir("results/figures")
    models = [label for _name, label in MODEL_SPECS]
    lrs = ROBUSTNESS_LRS
    matrix = np.zeros((len(models), len(lrs)), dtype=float)
    for i, model in enumerate(models):
        for j, lr in enumerate(lrs):
            row = df[(df["model"] == model) & (np.isclose(df["base_lr"], lr))]
            matrix[i, j] = float(row["stable"].iloc[0]) if not row.empty else np.nan
    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(models)), labels=models)
    ax.set_xticks(np.arange(len(lrs)), labels=[f"{lr:g}" for lr in lrs], rotation=35, ha="right")
    ax.set_xlabel("base_lr")
    ax.set_title("Short LR Stress-Test Stability")
    for i in range(len(models)):
        for j in range(len(lrs)):
            ax.text(j, i, "stable" if matrix[i, j] == 1 else "fail", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig("results/figures/bn_lr_stability_heatmap.png", dpi=220)
    plt.close(fig)


def checkpoint_to_model(path: str | Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model_name = str(config.get("model"))
    model = build_model(model_name, dropout=float(config.get("dropout", 0.5))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def make_probe_loader(args: argparse.Namespace, split: str = "train") -> DataLoader:
    split_info = get_or_create_cifar10_split(
        data_dir="./data",
        split_file=DEFAULT_SPLIT_FILE,
        split_seed=42,
        val_size_per_class=500,
        download=True,
    )
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=build_transforms(train=False))
    key = "train_indices" if split == "train" else "val_indices"
    indices = [int(i) for i in split_info[key]]
    max_items = int(args.probe_batches) * int(args.batch_size)
    indices = indices[:max_items]
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_seed_fn,
        persistent_workers=int(args.num_workers) > 0,
        prefetch_factor=4 if int(args.num_workers) > 0 else None,
    )


def batches_to_device(loader: DataLoader, device: torch.device, max_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for idx, (x, y) in enumerate(loader):
        if idx >= max_batches:
            break
        batches.append((x.to(device, non_blocking=True), y.to(device, non_blocking=True)))
    return batches


def set_probe_mode(model: nn.Module, bn_train: bool = False) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train(mode=bn_train)
            if bn_train:
                module.momentum = 0.0
        if isinstance(module, nn.Dropout):
            module.eval()


def params_requiring_grad(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def tensor_list_norm(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    total = None
    for tensor in tensors:
        value = torch.sum(tensor.detach() * tensor.detach())
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return torch.sqrt(total)


def dot_lists(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
    total = left[0].new_tensor(0.0)
    for a, b in zip(left, right):
        total = total + torch.sum(a.detach() * b.detach())
    return total


def clone_grads(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]


def average_loss(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]], criterion: nn.Module, bn_train: bool) -> torch.Tensor:
    set_probe_mode(model, bn_train=bn_train)
    total = None
    count = 0
    for x, y in batches:
        loss = criterion(model(x), y)
        weighted = loss * y.size(0)
        total = weighted if total is None else total + weighted
        count += y.size(0)
    if total is None:
        raise RuntimeError("No probe batches available.")
    return total / max(count, 1)


def loss_and_grads(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    bn_train: bool,
) -> tuple[float, list[torch.Tensor], float]:
    model.zero_grad(set_to_none=True)
    loss = average_loss(model, batches, criterion, bn_train=bn_train)
    loss.backward()
    params = params_requiring_grad(model)
    grads = clone_grads(params)
    grad_norm = float(tensor_list_norm(grads).item())
    return float(loss.item()), grads, grad_norm


def apply_steps(params: list[nn.Parameter], steps: list[torch.Tensor], scale: float = 1.0) -> None:
    with torch.no_grad():
        for p, step in zip(params, steps):
            p.add_(step, alpha=scale)


def global_direction_from_grads(grads: list[torch.Tensor], negative: bool = True) -> list[torch.Tensor]:
    norm = tensor_list_norm(grads).clamp_min(EPS)
    sign = -1.0 if negative else 1.0
    return [sign * g / norm for g in grads]


def random_global_direction(params: list[nn.Parameter], seed: int) -> list[torch.Tensor]:
    generator = torch.Generator(device=params[0].device).manual_seed(seed)
    raw = [torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator) for p in params]
    norm = tensor_list_norm(raw).clamp_min(EPS)
    return [r / norm for r in raw]


def steps_from_global_direction(direction: list[torch.Tensor], alpha: float, weight_norm: float) -> list[torch.Tensor]:
    return [float(alpha) * float(weight_norm) * d for d in direction]


def layer_normalized_steps(params: list[nn.Parameter], grads: list[torch.Tensor], alpha: float) -> list[torch.Tensor]:
    steps = []
    for p, g in zip(params, grads):
        p_norm = torch.norm(p.detach()).clamp_min(EPS)
        g_norm = torch.norm(g.detach()).clamp_min(EPS)
        if not torch.isfinite(g_norm) or float(g_norm.item()) <= EPS:
            steps.append(torch.zeros_like(p))
        else:
            steps.append(-float(alpha) * p_norm * g / g_norm)
    return steps


def logits_and_loss(
    model: nn.Module,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    bn_train: bool,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    set_probe_mode(model, bn_train=bn_train)
    logits_all = []
    targets_all = []
    total = None
    count = 0
    with torch.no_grad():
        for x, y in batches:
            logits = model(x)
            loss = criterion(logits, y)
            logits_all.append(logits.detach())
            targets_all.append(y.detach())
            weighted = loss * y.size(0)
            total = weighted if total is None else total + weighted
            count += y.size(0)
    if total is None:
        raise RuntimeError("No probe batches available.")
    return float((total / max(count, 1)).item()), torch.cat(logits_all), torch.cat(targets_all)


def run_scale_invariance(summaries: list[dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    device = get_device(args.device)
    criterion = build_criterion({"loss": "ce"})
    batches = batches_to_device(make_probe_loader(args, split="train"), device, int(args.probe_batches))
    rows = []
    for summary in summaries:
        model = checkpoint_to_model(summary["checkpoint"], device)
        label = summary["model"]
        modes = [("train_batch_stats", True)] if label == "VGG-A-BN" else [("eval_no_bn", False)]
        if label == "VGG-A-BN":
            modes.append(("eval_running_stats", False))
        convs = conv_layers(model)
        for mode_name, bn_train in modes:
            base_loss = None
            base_logits = None
            base_preds = None
            for factor in SCALE_FACTORS:
                with torch.no_grad():
                    for conv in convs:
                        conv.weight.mul_(float(factor))
                loss, logits, targets = logits_and_loss(model, batches, criterion, bn_train=bn_train)
                preds = logits.argmax(dim=1)
                if factor == 1.0:
                    base_loss = loss
                    base_logits = logits.detach().clone()
                    base_preds = preds.detach().clone()
                if base_loss is None or base_logits is None or base_preds is None:
                    # Scale 1.0 is not first in the list, so compute an explicit baseline.
                    with torch.no_grad():
                        for conv in convs:
                            conv.weight.div_(float(factor))
                    with torch.no_grad():
                        for conv in convs:
                            conv.weight.mul_(1.0)
                    base_loss, base_logits, targets = logits_and_loss(model, batches, criterion, bn_train=bn_train)
                    base_preds = base_logits.argmax(dim=1)
                    with torch.no_grad():
                        for conv in convs:
                            conv.weight.mul_(float(factor))
                    loss, logits, targets = logits_and_loss(model, batches, criterion, bn_train=bn_train)
                    preds = logits.argmax(dim=1)
                logit_change = float(torch.norm(logits - base_logits).item() / max(float(torch.norm(base_logits).item()), EPS))
                flip_rate = float((preds != base_preds).float().mean().item())
                rows.append(
                    {
                        "model": label,
                        "mode": mode_name,
                        "probe_split": "train_dev",
                        "scale_factor": float(factor),
                        "loss": float(loss),
                        "loss_change": float(loss - base_loss),
                        "relative_logit_change": logit_change,
                        "prediction_flip_rate": flip_rate,
                        "accuracy": accuracy_from_logits(logits, targets),
                    }
                )
                with torch.no_grad():
                    for conv in convs:
                        conv.weight.div_(float(factor))
    df = pd.DataFrame(rows)
    df.to_csv("results/metrics/bn_scale_invariance.csv", index=False)
    plot_scale_invariance(df)
    return df


def plot_scale_invariance(df: pd.DataFrame) -> None:
    ensure_dir("results/figures")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (model, mode), group in df.groupby(["model", "mode"]):
        group = group.sort_values("scale_factor")
        ax.plot(group["scale_factor"], group["loss_change"], marker="o", label=f"{model} {mode}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Conv weight scale factor")
    ax.set_ylabel("Loss change vs scale=1")
    ax.set_title("Scale-Invariance Probe")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/figures/bn_scale_invariance_loss.png", dpi=220)
    plt.close(fig)


def snapshot_entries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    best_epoch = int(summary["best_val_epoch"])
    entries: dict[str, dict[str, Any]] = {}
    for epoch, path in summary["snapshot_paths"].items():
        entries[f"epoch{int(epoch)}"] = {
            "snapshot_label": f"epoch{int(epoch)}",
            "snapshot_epoch": int(epoch),
            "checkpoint": path,
        }
    entries["best-val"] = {
        "snapshot_label": "best-val",
        "snapshot_epoch": best_epoch,
        "checkpoint": summary["checkpoint"],
    }
    dedup: dict[int, dict[str, Any]] = {}
    for entry in entries.values():
        key = int(entry["snapshot_epoch"])
        if key not in dedup or entry["snapshot_label"] == "best-val":
            dedup[key] = entry
    return sorted(dedup.values(), key=lambda item: (int(item["snapshot_epoch"]), item["snapshot_label"] != "best-val"))


def run_directional_loss_profile(summaries: list[dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    device = get_device(args.device)
    criterion = build_criterion({"loss": "ce"})
    batches = batches_to_device(make_probe_loader(args, split="train"), device, int(args.probe_batches))
    rows = []
    for summary in summaries:
        for entry in snapshot_entries(summary):
            model = checkpoint_to_model(entry["checkpoint"], device)
            label = summary["model"]
            bn_train = label == "VGG-A-BN"
            params = params_requiring_grad(model)
            base_loss, grads, grad_norm = loss_and_grads(model, batches, criterion, bn_train=bn_train)
            weight_norm = float(tensor_list_norm([p.detach() for p in params]).item())
            directions = {
                "d_grad": global_direction_from_grads(grads, negative=True),
                "d_random": random_global_direction(params, seed=int(args.seed) + int(entry["snapshot_epoch"]) + (17 if label == "VGG-A-BN" else 0)),
            }
            for direction_name, direction in directions.items():
                direction_norm = float(tensor_list_norm(direction).item())
                for alpha in DIRECTION_ALPHA_GRID:
                    steps = steps_from_global_direction(direction, alpha, weight_norm)
                    delta_norm = float(tensor_list_norm(steps).item())
                    apply_steps(params, steps, scale=1.0)
                    with torch.no_grad():
                        loss = float(average_loss(model, batches, criterion, bn_train=bn_train).item())
                    apply_steps(params, steps, scale=-1.0)
                    rows.append(
                        {
                            "model": label,
                            "probe_split": "train_dev",
                            "snapshot_label": entry["snapshot_label"],
                            "snapshot_epoch": int(entry["snapshot_epoch"]),
                            "direction": direction_name,
                            "alpha": float(alpha),
                            "loss": loss,
                            "loss_delta": float(loss - base_loss),
                            "base_loss": float(base_loss),
                            "grad_norm": float(grad_norm),
                            "direction_l2_norm": direction_norm,
                            "delta_norm": delta_norm,
                            "weight_norm": weight_norm,
                            "relative_perturbation_norm": float(delta_norm / max(weight_norm, EPS)),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv("results/metrics/bn_directional_loss_profile.csv", index=False)
    plot_directional_profile(df)
    return df


def plot_directional_profile(df: pd.DataFrame) -> None:
    ensure_dir("results/figures")
    plot_df = df[df["snapshot_label"] == "best-val"].copy()
    if plot_df.empty:
        max_epoch = df["snapshot_epoch"].max()
        plot_df = df[df["snapshot_epoch"] == max_epoch].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, direction in zip(axes, ["d_grad", "d_random"]):
        subset = plot_df[plot_df["direction"] == direction]
        for model, group in subset.groupby("model"):
            group = group.sort_values("alpha")
            ax.plot(group["relative_perturbation_norm"] * np.sign(group["alpha"]), group["loss_delta"], marker="o", label=model)
        ax.set_title(direction)
        ax.set_xlabel("signed relative perturbation norm")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel("Loss delta")
    fig.suptitle("Directional Local Loss Profile on train_dev Probe Batches")
    fig.tight_layout()
    fig.savefig("results/figures/bn_directional_loss_profile.png", dpi=220)
    plt.close(fig)


def run_gradient_predictiveness_v2(summaries: list[dict[str, Any]], args: argparse.Namespace) -> pd.DataFrame:
    device = get_device(args.device)
    criterion = build_criterion({"loss": "ce"})
    batches = batches_to_device(make_probe_loader(args, split="train"), device, int(args.probe_batches))
    rows = []
    for summary in summaries:
        for entry in snapshot_entries(summary):
            model = checkpoint_to_model(entry["checkpoint"], device)
            label = summary["model"]
            bn_train = label == "VGG-A-BN"
            params = params_requiring_grad(model)
            base_loss, grads, grad_norm = loss_and_grads(model, batches, criterion, bn_train=bn_train)
            weight_norm = float(tensor_list_norm([p.detach() for p in params]).item())
            for alpha in PREDICT_ALPHA_GRID:
                steps = layer_normalized_steps(params, grads, alpha)
                delta_norm = float(tensor_list_norm(steps).item())
                linear_pred_delta = float(dot_lists(grads, steps).item())
                apply_steps(params, steps, scale=1.0)
                perturbed_loss, perturbed_grads, perturbed_grad_norm = loss_and_grads(model, batches, criterion, bn_train=bn_train)
                grad_diffs = [b - a for a, b in zip(grads, perturbed_grads)]
                grad_diff_norm = float(tensor_list_norm(grad_diffs).item())
                grad_cosine = float(dot_lists(grads, perturbed_grads).item() / max(float(grad_norm) * float(perturbed_grad_norm), EPS))
                actual_delta = float(perturbed_loss - base_loss)
                apply_steps(params, steps, scale=-1.0)
                rows.append(
                    {
                        "model": label,
                        "probe_split": "train_dev",
                        "snapshot_label": entry["snapshot_label"],
                        "snapshot_epoch": int(entry["snapshot_epoch"]),
                        "alpha": float(alpha),
                        "perturbation_mode": "layer_normalized",
                        "base_loss": float(base_loss),
                        "perturbed_loss": float(perturbed_loss),
                        "actual_delta": actual_delta,
                        "linear_pred_delta": linear_pred_delta,
                        "relative_prediction_error": float(abs(actual_delta - linear_pred_delta) / max(abs(actual_delta), EPS)),
                        "gradient_cosine_similarity": grad_cosine,
                        "grad_norm": float(grad_norm),
                        "perturbed_grad_norm": float(perturbed_grad_norm),
                        "grad_diff_norm": grad_diff_norm,
                        "relative_gradient_change": float(grad_diff_norm / max(float(grad_norm), EPS)),
                        "delta_norm": delta_norm,
                        "weight_norm": weight_norm,
                        "relative_perturbation_norm": float(delta_norm / max(weight_norm, EPS)),
                        "grad_lipschitz_estimate": float(grad_diff_norm / max(delta_norm, EPS)),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv("results/metrics/bn_gradient_predictiveness_v2.csv", index=False)
    plot_gradient_predictiveness(df)
    return df


def plot_gradient_predictiveness(df: pd.DataFrame) -> None:
    ensure_dir("results/figures")
    best = df[df["snapshot_label"] == "best-val"].copy()
    if best.empty:
        best = df[df["snapshot_epoch"] == df["snapshot_epoch"].max()].copy()

    fig, ax = plt.subplots(figsize=(7.5, 5))
    grouped = best.groupby(["model", "alpha"], as_index=False)["relative_prediction_error"].mean()
    for model, group in grouped.groupby("model"):
        ax.plot(group["alpha"], group["relative_prediction_error"], marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Relative prediction error")
    ax.set_title("Gradient Predictiveness 2.0")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/bn_gradient_predictiveness_vs_alpha.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    grouped = best.groupby(["model", "alpha"], as_index=False)["gradient_cosine_similarity"].mean()
    for model, group in grouped.groupby("model"):
        ax.plot(group["alpha"], group["gradient_cosine_similarity"], marker="o", label=model)
    ax.set_xscale("log")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Gradient cosine similarity")
    ax.set_title("Gradient Direction Stability")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig("results/figures/bn_gradient_cosine_similarity.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [group["grad_lipschitz_estimate"].dropna().to_numpy() for _model, group in best.groupby("model")]
    labels = [model for model, _group in best.groupby("model")]
    if data:
        ax.violinplot(data, showmeans=True, showmedians=True)
        ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_ylabel("||g' - g|| / ||w' - w||")
    ax.set_title("Gradient Lipschitz Estimate Distribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("results/figures/bn_gradient_lipschitz_violin.png", dpi=220)
    plt.close(fig)


def run_bn_p0(args: argparse.Namespace) -> None:
    ensure_dir("results/metrics")
    ensure_dir("results/figures")
    ensure_dir("results/checkpoints")
    ensure_dir("results/logs")
    ensure_dir("results/protocol")

    no_bn_state, bn_state, init_metadata = create_shared_initial_states(int(args.seed), dropout=0.5)
    initial_states = {"vgg_a": no_bn_state, "vgg_a_bn": bn_state}
    save_json(init_metadata, "results/protocol/bn_p0_shared_initialization.json")

    started = time.perf_counter()
    summaries = run_controlled_comparison(initial_states, args)
    run_required_loss_envelope(initial_states, args)
    run_lr_robustness(initial_states, args)
    run_scale_invariance(summaries, args)
    run_directional_loss_profile(summaries, args)
    run_gradient_predictiveness_v2(summaries, args)

    save_json(
        {
            "stage": "BN-P0",
            "status": "complete",
            "official_test_used": False,
            "final_lock_created": False,
            "p1_activation_statistics_run": False,
            "p2_bn_placement_ablation_run": False,
            "probe_main_split": "train_dev",
            "snapshot_epochs": SNAPSHOT_EPOCHS + ["best-val"],
            "required_lrs_as_base_lr": REQUIRED_LRS,
            "robustness_lrs_as_base_lr": ROBUSTNESS_LRS,
            "robustness_type": "short_lr_stress_test",
            "total_elapsed_seconds": float(time.perf_counter() - started),
        },
        "results/protocol/bn_p0_run_protocol.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded BN P0 mini-study.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--controlled_epochs", type=int, default=50)
    parser.add_argument("--controlled_lr", type=float, default=1e-2)
    parser.add_argument("--required_epochs", type=int, default=10)
    parser.add_argument("--stress_epochs", type=int, default=5)
    parser.add_argument("--probe_batches", type=int, default=4)
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--val_subset_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_bn_p0(args)


if __name__ == "__main__":
    main()
