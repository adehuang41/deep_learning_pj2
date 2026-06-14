"""Stage C long validation-only continuation for W4/W6 champion candidates."""

from __future__ import annotations

import argparse
import copy
import math
import shutil
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


COMMON_STAGE_C_OVERRIDES: dict[str, Any] = {
    "epochs": 250,
    "batch_size": 256,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "optimizer": "sgd",
    "lr": 0.2,
    "momentum": 0.9,
    "nesterov": True,
    "scheduler": "cosine",
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


DEFAULT_STAGE_C: dict[str, Any] = {
    "target_epochs": 250,
    "output_csv": "results/metrics/stageC_w4_w6_250.csv",
    "decision_json": "results/protocol/stageC_decision_inputs.json",
    "curve_figure": "results/figures/stageC_w4_w6_train_val_curves.png",
    "stage_b_results_csv": "results/metrics/stageB_top_models.csv",
    "scheduler_resume_policy": "explicit_total_budget_cosine_from_stage_b_checkpoint",
    "experiments": [
        {
            "role": "champion_candidate",
            "name": "champion_w4_stageC250",
            "stage_b_name": "champion_w4_stageB",
            "config": "configs/champion_w4.yaml",
            "resume_raw_checkpoint": "results/checkpoints/champion_w4_stageB_last.pt",
            "resume_ema_checkpoint": "results/checkpoints/champion_w4_stageB_last_ema.pt",
            "previous_history_csv": "results/logs/champion_w4_stageB_history.csv",
        },
        {
            "role": "champion_candidate",
            "name": "champion_w6_stageC250",
            "stage_b_name": "champion_w6_stageB",
            "config": "configs/champion_w6.yaml",
            "resume_raw_checkpoint": "results/checkpoints/champion_w6_stageB_last.pt",
            "resume_ema_checkpoint": "results/checkpoints/champion_w6_stageB_last_ema.pt",
            "previous_history_csv": "results/logs/champion_w6_stageB_history.csv",
        },
    ],
}


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _total_budget_cosine_lr(epoch: int, total_epochs: int, base_lr: float) -> float:
    progress = min(max(epoch - 1, 0), total_epochs)
    return float(0.5 * base_lr * (1.0 + math.cos(math.pi * progress / total_epochs)))


def _best_raw_ema_from_history(history: pd.DataFrame) -> dict[str, float | int | None]:
    raw_idx = int(history["val_acc"].idxmax())
    result: dict[str, float | int | None] = {
        "best_val_acc_raw": float(history.loc[raw_idx, "val_acc"]),
        "best_val_epoch_raw": int(history.loc[raw_idx, "epoch"]),
        "best_val_acc_ema": None,
        "best_val_epoch_ema": None,
    }
    if "val_acc_ema" in history and history["val_acc_ema"].notna().any():
        ema_idx = int(history["val_acc_ema"].idxmax())
        result["best_val_acc_ema"] = float(history.loc[ema_idx, "val_acc_ema"])
        result["best_val_epoch_ema"] = int(history.loc[ema_idx, "epoch"])
    return result


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


def _load_stage_b_row(stage_b_results_csv: str, stage_b_name: str) -> pd.Series:
    df = pd.read_csv(stage_b_results_csv)
    rows = df[df["experiment"] == stage_b_name]
    if rows.empty:
        raise ValueError(f"Missing Stage B row for {stage_b_name}: {stage_b_results_csv}")
    return rows.iloc[0]


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


def _copy_initial_best(stage_b_row: pd.Series, best_path: Path, best_ema_path: Path) -> tuple[Path, str]:
    selected_weights = str(stage_b_row["selected_weights"])
    src = Path(str(stage_b_row["checkpoint"]))
    if not src.exists():
        raise FileNotFoundError(f"Missing Stage B selected checkpoint: {src}")
    dest = best_ema_path if selected_weights == "ema" else best_path
    shutil.copy2(src, dest)
    return dest, selected_weights


def _plot_stage_c_curves(rows: pd.DataFrame, curve_figure: str | Path) -> None:
    ensure_dir(Path(curve_figure).parent)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    for name, group in rows.groupby("experiment"):
        label = "W4" if "w4" in name else "W6"
        axes[0, 0].plot(group["epoch"], group["train_acc"], label=f"{label} train", alpha=0.65)
        axes[0, 0].plot(group["epoch"], group["val_acc"], label=f"{label} val")
        axes[0, 1].plot(group["epoch"], group["val_acc"], label=f"{label} raw")
        axes[0, 1].plot(group["epoch"], group["val_acc_ema"], "--", label=f"{label} EMA", alpha=0.85)
        axes[1, 0].plot(group["epoch"], group["train_loss"], label=f"{label} train", alpha=0.65)
        axes[1, 0].plot(group["epoch"], group["val_loss"], label=f"{label} val")
        axes[1, 1].plot(group["epoch"], group["selected_val_acc"], label=label)
    axes[0, 0].set_title("Stage C train/val accuracy")
    axes[0, 1].set_title("Stage C raw vs EMA validation accuracy")
    axes[1, 0].set_title("Stage C train/val loss")
    axes[1, 1].set_title("Stage C selected validation accuracy")
    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("accuracy")
    axes[0, 1].set_ylabel("accuracy")
    axes[1, 0].set_ylabel("loss")
    axes[1, 1].set_ylabel("best-so-far accuracy")
    fig.tight_layout()
    fig.savefig(curve_figure)
    plt.close(fig)


def run_stage_c(config: dict[str, Any]) -> pd.DataFrame:
    stage = dict(DEFAULT_STAGE_C)
    stage.update(config)
    target_epochs = int(stage.get("target_epochs", 250))
    common = dict(COMMON_STAGE_C_OVERRIDES)
    common.update(stage.get("common_overrides", {}))
    common["epochs"] = target_epochs

    summaries: list[dict[str, Any]] = []
    combined_for_plot: list[pd.DataFrame] = []
    for exp in stage["experiments"]:
        exp_config = load_config(exp["config"])
        exp_config.update(common)
        exp_config["run_name"] = exp["name"]
        exp_config["metrics_filename"] = f"{exp['name']}_results.csv"
        exp_config["stage_c_resume_from"] = exp["resume_raw_checkpoint"]
        exp_config["stage_c_target_epochs"] = target_epochs
        exp_config["stage_c_scheduler_resume_policy"] = stage["scheduler_resume_policy"]
        config_merged = merge_config_with_overrides(exp_config, {}, DEFAULT_CONFIG)
        _protocol_guard(config_merged)

        run_name = str(config_merged["run_name"])
        set_seed(int(config_merged.get("seed", 42)), deterministic=bool(config_merged.get("deterministic", True)))
        torch.backends.cudnn.benchmark = bool(config_merged.get("cudnn_benchmark", False))
        device = get_device(config_merged.get("device"))
        channels_last = bool(config_merged.get("channels_last", False)) and device.type == "cuda"

        loaders = get_cifar10_loaders(
            data_dir=str(config_merged.get("data_dir", "./data")),
            batch_size=int(config_merged.get("batch_size", 256)),
            num_workers=int(config_merged.get("num_workers", 4)),
            pin_memory=bool(config_merged.get("pin_memory", True)) and device.type == "cuda",
            persistent_workers=bool(config_merged.get("persistent_workers", True)),
            prefetch_factor=config_merged.get("prefetch_factor"),
            subset_size=config_merged.get("subset_size"),
            val_subset_size=config_merged.get("val_subset_size"),
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
            raise RuntimeError("Stage C must use train_dev/val_dev only and no official_test loader.")

        raw_checkpoint_path = Path(str(exp["resume_raw_checkpoint"]))
        ema_checkpoint_path = Path(str(exp["resume_ema_checkpoint"]))
        raw_checkpoint = torch.load(raw_checkpoint_path, map_location=device)
        ema_checkpoint = torch.load(ema_checkpoint_path, map_location=device)
        resume_epoch = int(raw_checkpoint["epoch"])
        if resume_epoch >= target_epochs:
            raise ValueError(f"{run_name} resume epoch {resume_epoch} already reaches target {target_epochs}")

        model = _build_model_from_config(config_merged, device)
        model.load_state_dict(raw_checkpoint["model_state_dict"])
        criterion = build_criterion(config_merged)
        optimizer = build_optimizer(model, config_merged)
        optimizer.load_state_dict(raw_checkpoint["optimizer_state_dict"])
        ema = ModelEMA(model, decay=float(config_merged.get("ema_decay", 0.999)))
        ema.module.load_state_dict(ema_checkpoint["model_state_dict"])

        save_dir = ensure_dir(config_merged.get("save_dir", "results/checkpoints"))
        log_dir = ensure_dir(config_merged.get("log_dir", "results/logs"))
        metrics_dir = ensure_dir(config_merged.get("metrics_dir", "results/metrics"))
        protocol_dir = ensure_dir(config_merged.get("protocol_dir", "results/protocol"))
        best_path = save_dir / f"{run_name}_best_val.pt"
        best_ema_path = save_dir / f"{run_name}_best_val_ema.pt"
        last_path = save_dir / f"{run_name}_last.pt"
        ema_last_path = save_dir / f"{run_name}_last_ema.pt"
        log_path = log_dir / f"{run_name}_history.csv"
        summary_path = metrics_dir / str(config_merged.get("metrics_filename") or f"{run_name}_results.csv")

        previous_history = pd.read_csv(exp["previous_history_csv"])
        stage_b_row = _load_stage_b_row(str(stage["stage_b_results_csv"]), str(exp["stage_b_name"]))
        selected_best_path, best_weights = _copy_initial_best(stage_b_row, best_path, best_ema_path)
        best_val_acc = float(stage_b_row["best_val_acc"])
        best_val_loss = float(stage_b_row["best_val_loss"])
        best_val_epoch = int(stage_b_row["best_val_epoch"])
        previous_elapsed = float(previous_history["total_elapsed_time"].iloc[-1])
        rows: list[dict[str, Any]] = []
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        pbar = tqdm(range(resume_epoch + 1, target_epochs + 1), desc=f"stageC:{run_name}", unit="epoch")
        for epoch in pbar:
            lr = _total_budget_cosine_lr(epoch, target_epochs, float(config_merged.get("lr", 0.2)))
            _set_optimizer_lr(optimizer, lr)
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
            val_loss, val_acc = evaluate_model(model, loaders.val, criterion, device, channels_last=channels_last)
            val_loss_ema, val_acc_ema = evaluate_model(
                ema.module,
                loaders.val,
                criterion,
                device,
                channels_last=channels_last,
            )
            candidate_acc = val_acc
            candidate_loss = val_loss
            candidate_model = model
            candidate_weights = "raw"
            if _is_better_val(val_acc_ema, val_loss_ema, candidate_acc, candidate_loss):
                candidate_acc = val_acc_ema
                candidate_loss = val_loss_ema
                candidate_model = ema.module
                candidate_weights = "ema"
            if _is_better_val(candidate_acc, candidate_loss, best_val_acc, best_val_loss):
                best_val_acc = candidate_acc
                best_val_loss = candidate_loss
                best_val_epoch = epoch
                best_weights = candidate_weights
                selected_best_path = best_ema_path if candidate_weights == "ema" else best_path
                torch.save(
                    checkpoint_payload(
                        candidate_model,
                        optimizer,
                        epoch,
                        best_val_acc,
                        best_val_loss,
                        config_merged,
                        selected_weights=candidate_weights,
                    ),
                    selected_best_path,
                )

            elapsed_epoch = epoch_time(epoch_start)
            train_images = len(loaders.train.dataset)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_loss_ema": val_loss_ema,
                "val_acc_ema": val_acc_ema,
                "selected_val_acc": best_val_acc,
                "selected_weights": best_weights,
                "lr": current_lr(optimizer),
                "grad_norm": grad_norm,
                "epoch_time": elapsed_epoch,
                "images_per_second": float(train_images / max(elapsed_epoch, 1e-12)),
                "gpu_peak_memory_gb": _cuda_memory_gb(device),
                "total_elapsed_time": previous_elapsed + time.perf_counter() - start,
            }
            rows.append(row)
            torch.save(
                checkpoint_payload(
                    model,
                    optimizer,
                    epoch,
                    best_val_acc,
                    best_val_loss,
                    config_merged,
                    selected_weights="raw",
                ),
                last_path,
            )
            torch.save(
                checkpoint_payload(
                    ema.module,
                    optimizer,
                    epoch,
                    best_val_acc,
                    best_val_loss,
                    config_merged,
                    selected_weights="ema",
                ),
                ema_last_path,
            )
            pbar.set_postfix(train_acc=f"{train_acc:.4f}", val_acc=f"{val_acc:.4f}", best_val=f"{best_val_acc:.4f}")

        extension_history = pd.DataFrame(rows)
        combined_history = pd.concat([previous_history, extension_history], ignore_index=True)
        combined_history.to_csv(log_path, index=False)
        combined_for_plot.append(combined_history.assign(experiment=run_name))
        raw_ema = _best_raw_ema_from_history(combined_history)
        overfit = _overfitting_summary(combined_history)
        final_row = combined_history.iloc[-1].to_dict()
        summary = {
            "stage": "C",
            "role": exp.get("role", ""),
            "experiment": run_name,
            "model": config_merged.get("model"),
            "params": count_parameters(model),
            "resume_epoch": resume_epoch,
            "epochs": target_epochs,
            "best_val_acc": best_val_acc,
            "best_val_error": 1.0 - best_val_acc,
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
            "final_val_acc": float(final_row["val_acc"]),
            "final_val_loss": float(final_row["val_loss"]),
            "mean_epoch_time": float(combined_history["epoch_time"].mean()),
            "mean_images_per_second": float(combined_history["images_per_second"].mean()),
            "gpu_peak_memory_gb": float(combined_history["gpu_peak_memory_gb"].max()),
            "train_time_seconds": float(combined_history["total_elapsed_time"].iloc[-1]),
            "stage_c_extension_time_seconds": float(time.perf_counter() - start),
            "selected_weights": best_weights,
            "best_val_acc_raw": raw_ema["best_val_acc_raw"],
            "best_val_epoch_raw": raw_ema["best_val_epoch_raw"],
            "best_val_acc_ema": raw_ema["best_val_acc_ema"],
            "best_val_epoch_ema": raw_ema["best_val_epoch_ema"],
            **overfit,
            "checkpoint": str(selected_best_path),
            "last_checkpoint": str(last_path),
            "history_csv": str(log_path),
            "resume_raw_checkpoint": str(raw_checkpoint_path),
            "resume_ema_checkpoint": str(ema_checkpoint_path),
            "scheduler_resume_policy": stage["scheduler_resume_policy"],
            "official_test_used": False,
        }
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(stage["output_csv"], index=False)
        save_json(
            {
                "run_name": run_name,
                "stage": "C",
                "train_split": "dev",
                "resume_epoch": resume_epoch,
                "target_epochs": target_epochs,
                "resume_raw_checkpoint": str(raw_checkpoint_path),
                "resume_ema_checkpoint": str(ema_checkpoint_path),
                "scheduler_resume_policy": stage["scheduler_resume_policy"],
                "uses_validation_for_selection": True,
                "uses_test_during_training": False,
                "official_test_evaluated": False,
                "checkpoint_selection_split": "val",
                "summary_csv": str(summary_path),
            },
            protocol_dir / f"{run_name}_protocol.json",
        )

    df = pd.DataFrame(summaries)
    df.to_csv(stage["output_csv"], index=False)
    if combined_for_plot:
        _plot_stage_c_curves(pd.concat(combined_for_plot, ignore_index=True), stage["curve_figure"])
    save_json(
        {
            "stage": "C",
            "official_test_used": False,
            "compact_resnet_v2_continued": False,
            "target_epochs": target_epochs,
            "common_overrides": common,
            "scheduler_resume_policy": stage["scheduler_resume_policy"],
            "results_csv": str(stage["output_csv"]),
            "curve_figure": str(stage["curve_figure"]),
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
    parser = argparse.ArgumentParser(description="Run Stage C W4/W6 continuation.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--target_epochs", type=int, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.target_epochs is not None:
        cfg["target_epochs"] = args.target_epochs
        cfg.setdefault("common_overrides", {})["epochs"] = args.target_epochs
    if args.output_csv is not None:
        cfg["output_csv"] = args.output_csv
    df = run_stage_c(cfg)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
