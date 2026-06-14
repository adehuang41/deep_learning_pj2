"""Clean Stage C: 150-epoch base training followed by explicit low-LR fine-tuning."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from .data import DEFAULT_SPLIT_FILE, get_cifar10_loaders
from .models import build_model
from .train import (
    DEFAULT_CONFIG,
    ModelEMA,
    _cuda_memory_gb,
    _is_better_val,
    _protocol_guard,
    build_criterion,
    build_optimizer,
    checkpoint_payload,
    evaluate_model,
    train_one_epoch,
)
from .utils import (
    count_parameters,
    current_lr,
    ensure_dir,
    epoch_time,
    get_device,
    load_config,
    merge_config_with_overrides,
    save_json,
    set_seed,
)


COMMON_OVERRIDES: dict[str, Any] = {
    "batch_size": 256,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "optimizer": "sgd",
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,
    "loss": "ce_label_smoothing",
    "label_smoothing": 0.05,
    "randaugment": True,
    "randaugment_num_ops": 2,
    "randaugment_magnitude": 9,
    "cutout": True,
    "cutout_length": 16,
    "cutout_p": 1.0,
    "ema": True,
    "ema_decay": 0.999,
    "train_split": "dev",
    "checkpoint_selection_split": "val",
    "use_test_during_training": False,
    "split_seed": 42,
    "split_file": "splits/cifar10_train45000_val5000_seed42.json",
    "amp": False,
    "channels_last": False,
    "cudnn_benchmark": True,
    "save_dir": "results/checkpoints",
    "log_dir": "results/logs",
    "metrics_dir": "results/metrics",
    "protocol_dir": "results/protocol",
}


DEFAULT_STAGE: dict[str, Any] = {
    "fine_tune_epochs": 100,
    "fine_tune_max_lr": 3e-4,
    "fine_tune_min_lr": 1e-6,
    "warmup_epochs": 3,
    "fallback_start_lr": 1e-5,
    "resume_mode": "standard",
    "output_csv": "results/metrics/stageC_lowLR_w4_w6.csv",
    "lr_history_csv": "results/metrics/stageC_lowLR_lr_history.csv",
    "decision_json": "results/protocol/stageC_lowLR_decision_inputs.json",
    "curve_figure": "results/figures/stageC_lowLR_train_val_curves.png",
    "lr_figure": "results/figures/stageC_lowLR_lr_curve.png",
    "stage_b_results_csv": "results/metrics/stageB_top_models.csv",
    "experiments": [
        {
            "role": "champion_candidate",
            "name": "champion_w4_stageC_lowLR",
            "stage_b_name": "champion_w4_stageB",
            "config": "configs/champion_w4.yaml",
            "resume_raw_checkpoint": "results/checkpoints/champion_w4_stageB_last.pt",
            "base_history_csv": "results/logs/champion_w4_stageB_history.csv",
        },
        {
            "role": "champion_candidate",
            "name": "champion_w6_stageC_lowLR",
            "stage_b_name": "champion_w6_stageB",
            "config": "configs/champion_w6.yaml",
            "resume_raw_checkpoint": "results/checkpoints/champion_w6_stageB_last.pt",
            "base_history_csv": "results/logs/champion_w6_stageB_history.csv",
        },
    ],
}


class LowLRFineTuneScheduler:
    """Epoch-level warmup plus cosine decay fine-tuning schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        start_lr: float,
        max_lr: float,
        min_lr: float,
        warmup_epochs: int,
        total_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.start_lr = float(start_lr)
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)
        self.last_epoch = 0

    def lr_for_epoch(self, fine_tune_epoch: int) -> float:
        epoch = int(fine_tune_epoch)
        if epoch <= self.warmup_epochs:
            if self.warmup_epochs <= 1:
                return self.max_lr
            alpha = (epoch - 1) / max(1, self.warmup_epochs - 1)
            return float(self.start_lr + alpha * (self.max_lr - self.start_lr))
        decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = min(max(epoch - self.warmup_epochs, 0), decay_epochs) / decay_epochs
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.min_lr + (self.max_lr - self.min_lr) * cosine)

    def set_epoch(self, fine_tune_epoch: int) -> float:
        lr = self.lr_for_epoch(fine_tune_epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_epoch = int(fine_tune_epoch)
        return lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "name": "low_lr_warmup_cosine",
            "start_lr": self.start_lr,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "warmup_epochs": self.warmup_epochs,
            "total_epochs": self.total_epochs,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.start_lr = float(state["start_lr"])
        self.max_lr = float(state["max_lr"])
        self.min_lr = float(state["min_lr"])
        self.warmup_epochs = int(state["warmup_epochs"])
        self.total_epochs = int(state["total_epochs"])
        self.last_epoch = int(state["last_epoch"])


def _validate_resume_checkpoint(checkpoint: dict[str, Any], resume_mode: str) -> None:
    if checkpoint.get("scheduler_state_dict") is None and resume_mode != "low_lr_finetune":
        raise ValueError(
            "Resume checkpoint lacks scheduler_state_dict. "
            "Use --resume_mode low_lr_finetune only for explicit low-LR fine-tuning."
        )


def _checkpoint_lr(checkpoint: dict[str, Any], fallback: float) -> float:
    groups = checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])
    if groups and "lr" in groups[0]:
        return float(groups[0]["lr"])
    return float(fallback)


def _build_model_from_config(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = build_model(
        str(config["model"]),
        num_classes=10,
        channels=config.get("channels", [64, 128, 256]),
        activation=str(config.get("activation", "silu")),
        dropout=float(config.get("dropout", 0.2)),
        blocks_per_stage=config.get("blocks_per_stage"),
        drop_path_rate=config.get("drop_path_rate"),
        use_eca=config.get("use_eca"),
        depth=config.get("depth"),
        widen_factor=config.get("widen_factor"),
    ).to(device)
    if bool(config.get("channels_last", False)) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def _raw_ema_best(history: pd.DataFrame) -> dict[str, float | int]:
    raw_idx = int(history["val_acc"].idxmax())
    ema_idx = int(history["val_acc_ema"].idxmax())
    return {
        "best_val_acc_raw": float(history.loc[raw_idx, "val_acc"]),
        "best_val_epoch_raw": int(history.loc[raw_idx, "fine_tune_epoch"]),
        "best_val_acc_ema": float(history.loc[ema_idx, "val_acc_ema"]),
        "best_val_epoch_ema": int(history.loc[ema_idx, "fine_tune_epoch"]),
    }


def _overfitting_summary(history: pd.DataFrame) -> dict[str, Any]:
    tail = history.tail(min(10, len(history)))
    raw_idx = int(history["val_acc"].idxmax())
    best_row = history.iloc[raw_idx]
    final_row = history.iloc[-1]
    return {
        "best_train_acc_at_raw_best_val": float(best_row["train_acc"]),
        "best_raw_val_acc": float(best_row["val_acc"]),
        "final_train_acc": float(final_row["train_acc"]),
        "final_train_val_gap": float(final_row["train_acc"] - final_row["val_acc"]),
        "last10_val_acc_mean": float(tail["val_acc"].mean()),
        "last10_val_acc_std": float(tail["val_acc"].std(ddof=0)),
        "last10_val_loss_mean": float(tail["val_loss"].mean()),
    }


def _plot_curves(base_and_ft: pd.DataFrame, curve_figure: str | Path) -> None:
    ensure_dir(Path(curve_figure).parent)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    for name, group in base_and_ft.groupby("experiment"):
        label = "W4" if "w4" in name else "W6"
        axes[0, 0].plot(group["absolute_epoch"], group["train_acc"], label=f"{label} train", alpha=0.65)
        axes[0, 0].plot(group["absolute_epoch"], group["val_acc"], label=f"{label} val")
        axes[0, 1].plot(group["absolute_epoch"], group["val_acc"], label=f"{label} raw")
        if "val_acc_ema" in group:
            axes[0, 1].plot(group["absolute_epoch"], group["val_acc_ema"], "--", label=f"{label} EMA", alpha=0.85)
        axes[1, 0].plot(group["absolute_epoch"], group["train_loss"], label=f"{label} train", alpha=0.65)
        axes[1, 0].plot(group["absolute_epoch"], group["val_loss"], label=f"{label} val")
        axes[1, 1].plot(group["absolute_epoch"], group["selected_val_acc"], label=label)
    for ax in axes.flat:
        ax.axvline(150, color="black", linewidth=1, alpha=0.35)
        ax.set_xlabel("absolute epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 0].set_title("Stage C-lowLR train/val accuracy")
    axes[0, 1].set_title("Stage C-lowLR raw vs EMA validation accuracy")
    axes[1, 0].set_title("Stage C-lowLR train/val loss")
    axes[1, 1].set_title("Stage C-lowLR selected validation accuracy")
    axes[0, 0].set_ylabel("accuracy")
    axes[0, 1].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("loss")
    axes[1, 1].set_ylabel("best-so-far accuracy")
    fig.tight_layout()
    fig.savefig(curve_figure)
    plt.close(fig)


def _plot_lr(lr_history: pd.DataFrame, lr_figure: str | Path) -> None:
    ensure_dir(Path(lr_figure).parent)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    for name, group in lr_history.groupby("experiment"):
        label = "W4" if "w4" in name else "W6"
        ax.plot(group["fine_tune_epoch"], group["lr_start"], label=f"{label} lr_start")
        ax.plot(group["fine_tune_epoch"], group["lr_end"], "--", label=f"{label} lr_end", alpha=0.8)
    ax.set_title("Stage C-lowLR fine-tuning LR schedule")
    ax.set_xlabel("fine-tuning epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(lr_figure)
    plt.close(fig)


def run_stage_c_lowlr(config: dict[str, Any]) -> pd.DataFrame:
    stage = dict(DEFAULT_STAGE)
    stage.update(config)
    resume_mode = str(stage.get("resume_mode", "standard"))
    fine_tune_epochs = int(stage["fine_tune_epochs"])
    summaries: list[dict[str, Any]] = []
    lr_rows: list[dict[str, Any]] = []
    plot_frames: list[pd.DataFrame] = []

    for exp in stage["experiments"]:
        exp_config = load_config(exp["config"])
        exp_config.update(COMMON_OVERRIDES)
        exp_config.update(stage.get("common_overrides", {}))
        exp_config["run_name"] = exp["name"]
        exp_config["scheduler"] = "low_lr_warmup_cosine"
        exp_config["resume_mode"] = resume_mode
        exp_config["fine_tune_epochs"] = fine_tune_epochs
        exp_config["fine_tune_max_lr"] = float(stage["fine_tune_max_lr"])
        exp_config["fine_tune_min_lr"] = float(stage["fine_tune_min_lr"])
        exp_config["warmup_epochs"] = int(stage["warmup_epochs"])
        config_merged = merge_config_with_overrides(exp_config, {}, DEFAULT_CONFIG)
        _protocol_guard(config_merged)

        run_name = str(config_merged["run_name"])
        set_seed(int(config_merged.get("seed", 42)), deterministic=bool(config_merged.get("deterministic", True)))
        torch.backends.cudnn.benchmark = bool(config_merged.get("cudnn_benchmark", False))
        device = get_device(config_merged.get("device"))
        channels_last = bool(config_merged.get("channels_last", False)) and device.type == "cuda"

        checkpoint_path = Path(str(exp["resume_raw_checkpoint"]))
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        _validate_resume_checkpoint(checkpoint, resume_mode)
        base_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint.get("global_step") or 0)
        start_lr = _checkpoint_lr(checkpoint, float(stage["fallback_start_lr"]))

        loaders = get_cifar10_loaders(
            data_dir=str(config_merged.get("data_dir", "./data")),
            batch_size=int(config_merged.get("batch_size", 256)),
            num_workers=int(config_merged.get("num_workers", 4)),
            pin_memory=bool(config_merged.get("pin_memory", True)) and device.type == "cuda",
            persistent_workers=bool(config_merged.get("persistent_workers", True)),
            prefetch_factor=config_merged.get("prefetch_factor"),
            subset_size=None,
            val_subset_size=None,
            seed=int(config_merged.get("seed", 42)),
            split_seed=int(config_merged.get("split_seed", 42)),
            split_file=str(config_merged.get("split_file", DEFAULT_SPLIT_FILE)),
            val_size_per_class=int(config_merged.get("val_size_per_class", 500)),
            train_full=False,
            include_val=True,
            include_test=False,
            train_transform_config=config_merged,
        )
        if loaders.test is not None or loaders.val is None:
            raise RuntimeError("Stage C-lowLR must use train_dev/val_dev only and no official_test loader.")

        model = _build_model_from_config(config_merged, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        criterion = build_criterion(config_merged)
        optimizer_config = dict(config_merged)
        optimizer_config["lr"] = start_lr
        optimizer = build_optimizer(model, optimizer_config)
        scheduler = LowLRFineTuneScheduler(
            optimizer,
            start_lr=start_lr,
            max_lr=float(stage["fine_tune_max_lr"]),
            min_lr=float(stage["fine_tune_min_lr"]),
            warmup_epochs=int(stage["warmup_epochs"]),
            total_epochs=fine_tune_epochs,
        )
        ema = ModelEMA(model, decay=float(config_merged.get("ema_decay", 0.999)))

        save_dir = ensure_dir(config_merged.get("save_dir", "results/checkpoints"))
        log_dir = ensure_dir(config_merged.get("log_dir", "results/logs"))
        metrics_dir = ensure_dir(config_merged.get("metrics_dir", "results/metrics"))
        protocol_dir = ensure_dir(config_merged.get("protocol_dir", "results/protocol"))
        raw_best_path = save_dir / f"{run_name}_best_val_raw.pt"
        ema_best_path = save_dir / f"{run_name}_best_val_ema.pt"
        selected_best_path = save_dir / f"{run_name}_best_val.pt"
        last_path = save_dir / f"{run_name}_last.pt"
        last_ema_path = save_dir / f"{run_name}_last_ema.pt"
        history_path = log_dir / f"{run_name}_history.csv"
        summary_path = metrics_dir / f"{run_name}_results.csv"

        best_selected_acc = -1.0
        best_selected_loss = float("inf")
        best_selected_epoch = 0
        best_selected_weights = "raw"
        best_raw_acc = -1.0
        best_raw_loss = float("inf")
        best_raw_epoch = 0
        best_ema_acc = -1.0
        best_ema_loss = float("inf")
        best_ema_epoch = 0
        rows: list[dict[str, Any]] = []
        start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        pbar = tqdm(range(1, fine_tune_epochs + 1), desc=f"stageC-lowLR:{run_name}", unit="epoch")
        for fine_epoch in pbar:
            absolute_epoch = base_epoch + fine_epoch
            lr_start = scheduler.set_epoch(fine_epoch)
            epoch_start = time.perf_counter()
            train_loss, train_acc, grad_norm = train_one_epoch(
                model,
                loaders.train,
                criterion,
                optimizer,
                device,
                scaler=None,
                channels_last=channels_last,
                ema=ema,
            )
            global_step += len(loaders.train)
            val_loss, val_acc = evaluate_model(model, loaders.val, criterion, device, channels_last=channels_last)
            val_loss_ema, val_acc_ema = evaluate_model(
                ema.module,
                loaders.val,
                criterion,
                device,
                channels_last=channels_last,
            )
            lr_end = current_lr(optimizer)

            if _is_better_val(val_acc, val_loss, best_raw_acc, best_raw_loss):
                best_raw_acc = val_acc
                best_raw_loss = val_loss
                best_raw_epoch = fine_epoch
                torch.save(
                    checkpoint_payload(
                        model,
                        optimizer,
                        fine_epoch,
                        best_raw_acc,
                        best_raw_loss,
                        config_merged,
                        selected_weights="raw",
                        scheduler=scheduler,
                        scaler=None,
                        ema=ema,
                        global_step=global_step,
                    ),
                    raw_best_path,
                )
            if _is_better_val(val_acc_ema, val_loss_ema, best_ema_acc, best_ema_loss):
                best_ema_acc = val_acc_ema
                best_ema_loss = val_loss_ema
                best_ema_epoch = fine_epoch
                torch.save(
                    checkpoint_payload(
                        ema.module,
                        optimizer,
                        fine_epoch,
                        best_ema_acc,
                        best_ema_loss,
                        config_merged,
                        selected_weights="ema",
                        scheduler=scheduler,
                        scaler=None,
                        ema=ema,
                        global_step=global_step,
                    ),
                    ema_best_path,
                )

            candidate_acc = val_acc
            candidate_loss = val_loss
            candidate_weights = "raw"
            candidate_path = raw_best_path
            if _is_better_val(val_acc_ema, val_loss_ema, candidate_acc, candidate_loss):
                candidate_acc = val_acc_ema
                candidate_loss = val_loss_ema
                candidate_weights = "ema"
                candidate_path = ema_best_path
            if _is_better_val(candidate_acc, candidate_loss, best_selected_acc, best_selected_loss):
                best_selected_acc = candidate_acc
                best_selected_loss = candidate_loss
                best_selected_epoch = fine_epoch
                best_selected_weights = candidate_weights
                selected_best_path.write_bytes(candidate_path.read_bytes())

            elapsed_epoch = epoch_time(epoch_start)
            train_images = len(loaders.train.dataset)
            peak_memory = _cuda_memory_gb(device)
            row = {
                "experiment": run_name,
                "fine_tune_epoch": fine_epoch,
                "epoch": fine_epoch,
                "absolute_epoch": absolute_epoch,
                "phase": "low_lr_finetune",
                "global_step": global_step,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_loss_ema": val_loss_ema,
                "val_acc_ema": val_acc_ema,
                "selected_val_acc": best_selected_acc,
                "selected_weights": best_selected_weights,
                "lr_start": lr_start,
                "lr_end": lr_end,
                "epoch_time": elapsed_epoch,
                "images_per_second": float(train_images / max(elapsed_epoch, 1e-12)),
                "peak_gpu_memory": peak_memory,
                "gpu_peak_memory_gb": peak_memory,
                "grad_norm": grad_norm,
                "total_elapsed_time": time.perf_counter() - start_time,
            }
            rows.append(row)
            lr_rows.append(
                {
                    "experiment": run_name,
                    "fine_tune_epoch": fine_epoch,
                    "absolute_epoch": absolute_epoch,
                    "global_step": global_step,
                    "lr_start": lr_start,
                    "lr_end": lr_end,
                    "scheduler": "warmup_cosine_low_lr",
                }
            )
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    fine_epoch,
                    best_selected_acc,
                    best_selected_loss,
                    config_merged,
                    selected_weights="raw",
                    scheduler=scheduler,
                    scaler=None,
                    ema=ema,
                    global_step=global_step,
                ),
                last_path,
            )
            torch.save(
                checkpoint_payload(
                    ema.module,
                    optimizer,
                    fine_epoch,
                    best_selected_acc,
                    best_selected_loss,
                    config_merged,
                    selected_weights="ema",
                    scheduler=scheduler,
                    scaler=None,
                    ema=ema,
                    global_step=global_step,
                ),
                last_ema_path,
            )
            pbar.set_postfix(train_acc=f"{train_acc:.4f}", val_acc=f"{val_acc:.4f}", best_val=f"{best_selected_acc:.4f}")

        history = pd.DataFrame(rows)
        history.to_csv(history_path, index=False)
        raw_ema = _raw_ema_best(history)
        overfit = _overfitting_summary(history)
        final_row = history.iloc[-1]
        summary = {
            "stage": "C-lowLR",
            "role": exp.get("role", ""),
            "experiment": run_name,
            "model": config_merged.get("model"),
            "params": count_parameters(model),
            "base_epoch": base_epoch,
            "fine_tune_epochs": fine_tune_epochs,
            "best_val_acc": best_selected_acc,
            "best_val_error": 1.0 - best_selected_acc,
            "best_val_loss": best_selected_loss,
            "best_val_epoch": best_selected_epoch,
            "final_val_acc": float(final_row["val_acc"]),
            "final_val_loss": float(final_row["val_loss"]),
            "mean_epoch_time": float(history["epoch_time"].mean()),
            "mean_images_per_second": float(history["images_per_second"].mean()),
            "gpu_peak_memory_gb": float(history["peak_gpu_memory"].max()),
            "fine_tune_time_seconds": float(history["total_elapsed_time"].iloc[-1]),
            "selected_weights": best_selected_weights,
            "raw_best_checkpoint": str(raw_best_path),
            "ema_best_checkpoint": str(ema_best_path),
            "checkpoint": str(selected_best_path),
            "last_checkpoint": str(last_path),
            "history_csv": str(history_path),
            "resume_raw_checkpoint": str(checkpoint_path),
            "resume_mode": resume_mode,
            "fine_tune_start_lr": start_lr,
            "fine_tune_max_lr": float(stage["fine_tune_max_lr"]),
            "fine_tune_min_lr": float(stage["fine_tune_min_lr"]),
            "warmup_epochs": int(stage["warmup_epochs"]),
            "official_test_used": False,
            **raw_ema,
            **overfit,
        }
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(stage["output_csv"], index=False)
        save_json(
            {
                "run_name": run_name,
                "stage": "C-lowLR",
                "description": "150-epoch base training followed by explicit low-learning-rate fine-tuning.",
                "train_split": "dev",
                "base_epoch": base_epoch,
                "fine_tune_epochs": fine_tune_epochs,
                "resume_raw_checkpoint": str(checkpoint_path),
                "resume_mode": resume_mode,
                "old_scheduler_restored": False,
                "fine_tune_optimizer_reinitialized": True,
                "ema_reinitialized_from_raw": True,
                "uses_validation_for_selection": True,
                "uses_test_during_training": False,
                "official_test_evaluated": False,
                "checkpoint_selection_split": "val",
                "summary_csv": str(summary_path),
            },
            protocol_dir / f"{run_name}_protocol.json",
        )

        base = pd.read_csv(exp["base_history_csv"]).copy()
        base["experiment"] = run_name
        base["phase"] = "stageB_base"
        base["absolute_epoch"] = base["epoch"]
        if "lr_start" not in base:
            base["lr_start"] = base.get("lr", float("nan"))
        if "lr_end" not in base:
            base["lr_end"] = base.get("lr", float("nan"))
        base["fine_tune_epoch"] = 0
        plot_frames.append(pd.concat([base, history], ignore_index=True, sort=False))

    df = pd.DataFrame(summaries)
    df.to_csv(stage["output_csv"], index=False)
    lr_history = pd.DataFrame(lr_rows)
    lr_history.to_csv(stage["lr_history_csv"], index=False)
    if plot_frames:
        _plot_curves(pd.concat(plot_frames, ignore_index=True, sort=False), stage["curve_figure"])
    _plot_lr(lr_history, stage["lr_figure"])
    save_json(
        {
            "stage": "C-lowLR",
            "description": "150-epoch base training followed by explicit low-learning-rate fine-tuning.",
            "official_test_used": False,
            "compact_resnet_v2_continued": False,
            "resume_mode": resume_mode,
            "old_scheduler_restored": False,
            "fine_tune_optimizer_reinitialized": True,
            "ema_reinitialized_from_raw": True,
            "fine_tune_epochs": fine_tune_epochs,
            "fine_tune_max_lr": float(stage["fine_tune_max_lr"]),
            "fine_tune_min_lr": float(stage["fine_tune_min_lr"]),
            "warmup_epochs": int(stage["warmup_epochs"]),
            "common_overrides": COMMON_OVERRIDES,
            "results_csv": str(stage["output_csv"]),
            "lr_history_csv": str(stage["lr_history_csv"]),
            "curve_figure": str(stage["curve_figure"]),
            "lr_figure": str(stage["lr_figure"]),
            "decision_rules": {
                "w6_over_w4_select_w6_threshold": 0.005,
                "w6_over_w4_select_w4_threshold": 0.003,
                "middle_band": "Use Pareto tradeoff, validation curve, stability, and report narrative; prefer W4 without clear W6 evidence.",
            },
        },
        stage["decision_json"],
    )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean Stage C low-LR fine-tuning.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume_mode", type=str, default="standard", choices=["standard", "low_lr_finetune"])
    parser.add_argument("--fine_tune_epochs", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.resume_mode is not None:
        cfg["resume_mode"] = args.resume_mode
    if args.fine_tune_epochs is not None:
        cfg["fine_tune_epochs"] = args.fine_tune_epochs
    if args.output_csv is not None:
        cfg["output_csv"] = args.output_csv
    df = run_stage_c_lowlr(cfg)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
