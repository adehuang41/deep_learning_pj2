"""Stage E full-train run for the locked W4 recipe.

This run uses the full CIFAR-10 official training set only. It deliberately
does not build a validation loader or an official-test loader.
"""

from __future__ import annotations

import argparse
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
from .stage_c_lowlr import LowLRFineTuneScheduler, _build_model_from_config
from .train import (
    DEFAULT_CONFIG,
    _cuda_memory_gb,
    _protocol_guard,
    build_criterion,
    build_optimizer,
    checkpoint_payload,
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


STAGE_E_DEFAULTS: dict[str, Any] = {
    "config": "configs/champion_w4.yaml",
    "run_name": "final_w4_fulltrain",
    "phase1_epochs": 150,
    "phase1_base_lr": 0.2,
    "phase2_fine_tune_epochs": 55,
    "phase2_max_lr": 3e-4,
    "phase2_min_lr": 1e-6,
    "phase2_warmup_epochs": 3,
    "phase2_fallback_start_lr": 1e-5,
    "checkpoint_path": "results/checkpoints/final_w4_fulltrain_raw.pt",
    "last_checkpoint_path": "results/checkpoints/final_w4_fulltrain_last.pt",
    "history_csv": "results/metrics/final_w4_fulltrain_history.csv",
    "curve_figure": "results/figures/final_w4_fulltrain_curves.png",
    "lr_figure": "results/figures/final_w4_fulltrain_lr_curve.png",
    "lock_json": "results/protocol/stageE_training_lock.json",
    "final_selection_lock": "results/protocol/final_selection_lock.json",
}


def _stage_e_config(base_config: dict[str, Any], stage: dict[str, Any]) -> dict[str, Any]:
    total_epochs = int(stage["phase1_epochs"]) + int(stage["phase2_fine_tune_epochs"])
    overrides = {
        "run_name": stage["run_name"],
        "epochs": total_epochs,
        "lr": float(stage["phase1_base_lr"]),
        "scheduler": "two_phase_base_cosine_then_low_lr_warmup_cosine",
        "train_split": "full",
        "checkpoint_selection_split": "none",
        "use_test_during_training": False,
        "full_train_requires_lock": False,
        "ema": False,
        "ema_decay": 0.0,
        "amp": False,
        "channels_last": False,
        "cudnn_benchmark": True,
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
        "split_seed": 42,
        "split_file": DEFAULT_SPLIT_FILE,
        "save_dir": "results/checkpoints",
        "log_dir": "results/logs",
        "metrics_dir": "results/metrics",
        "protocol_dir": "results/protocol",
        "stageE_phase1_epochs": int(stage["phase1_epochs"]),
        "stageE_phase2_fine_tune_epochs": int(stage["phase2_fine_tune_epochs"]),
        "stageE_phase2_max_lr": float(stage["phase2_max_lr"]),
        "stageE_phase2_min_lr": float(stage["phase2_min_lr"]),
        "stageE_phase2_warmup_epochs": int(stage["phase2_warmup_epochs"]),
        "stageE_epoch_budget_reason": (
            "W4 best validation in clean Stage C-lowLR occurred at fine-tune epoch 55; "
            "the full-train epoch count was locked before official test."
        ),
    }
    config = merge_config_with_overrides(base_config, {}, DEFAULT_CONFIG)
    config.update(overrides)
    return config


def _assert_preconditions(stage: dict[str, Any]) -> None:
    required = [
        "results/checkpoints/baseline_best_val.pt",
        "results/checkpoints/bn_p0_vgg_a_best_val.pt",
        "results/checkpoints/bn_p0_vgg_a_bn_best_val.pt",
        "results/protocol/final_lock_preparation.md",
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required pre-Stage-E artifact(s): {missing}")
    final_lock = Path(str(stage["final_selection_lock"]))
    if final_lock.exists():
        raise FileExistsError(
            f"{final_lock} already exists. Stage E was requested before final official-test lock creation."
        )


def _plot_training_curves(history: pd.DataFrame, output_path: str | Path, phase1_epochs: int) -> None:
    ensure_dir(Path(output_path).parent)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=160)
    axes[0].plot(history["epoch"], history["train_loss"], color="#1f77b4")
    axes[0].axvline(phase1_epochs, color="black", linewidth=1, alpha=0.35)
    axes[0].set_title("Final W4 full-train loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(history["epoch"], history["train_acc"], color="#2ca02c")
    axes[1].axvline(phase1_epochs, color="black", linewidth=1, alpha=0.35)
    axes[1].set_title("Final W4 full-train accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("train accuracy")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_lr_curve(history: pd.DataFrame, output_path: str | Path, phase1_epochs: int) -> None:
    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.plot(history["epoch"], history["lr_start"], label="lr_start")
    ax.plot(history["epoch"], history["lr_end"], "--", label="lr_end", alpha=0.85)
    ax.axvline(phase1_epochs, color="black", linewidth=1, alpha=0.35)
    ax.set_title("Final W4 full-train LR schedule")
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _stage_lock_payload(
    *,
    status: str,
    config: dict[str, Any],
    stage: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": "E",
        "status": status,
        "clean_simplecnn_baseline_completed": True,
        "bn_p0_completed": True,
        "p1_activation_statistics": "skipped_by_default",
        "p2_bn_placement_ablation": "skipped_by_default",
        "official_test_used": False,
        "final_selection_lock_created": False,
        "final_selection_lock_path": str(stage["final_selection_lock"]),
        "provisional_final_champion": "CIFAR-PreActResNet-W4",
        "champion_reason": (
            "W4 is the Pareto winner; W6 improves validation accuracy by only 0.34pp "
            "in clean Stage C-lowLR while using about 2.25x parameters and more memory."
        ),
        "w6_role": "validation-stage larger candidate, not selected for Stage E",
        "compact_resnet_v2_role": "validation-stage development evidence only; dominated by W4",
        "final_policy": {
            "model": "CIFAR-PreActResNet-W4",
            "weights": "raw",
            "ema": False,
            "tta": "no-TTA",
        },
        "official_test_row_plan": [
            {
                "row": 1,
                "name": "SimpleCNN baseline",
                "checkpoint": "results/checkpoints/baseline_best_val.pt",
                "source": "best validation checkpoint",
            },
            {
                "row": 2,
                "name": "VGG-A no BN",
                "checkpoint": "results/checkpoints/bn_p0_vgg_a_best_val.pt",
                "source": "best validation checkpoint",
            },
            {
                "row": 3,
                "name": "VGG-A BN",
                "checkpoint": "results/checkpoints/bn_p0_vgg_a_bn_best_val.pt",
                "source": "best validation checkpoint",
            },
            {
                "row": 4,
                "name": "Final W4 full-train champion",
                "checkpoint": str(stage["checkpoint_path"]),
                "source": "locked full-train recipe",
            },
        ],
        "stageE_recipe": {
            "model": config.get("model"),
            "depth": int(config.get("depth", 28)),
            "widen_factor": int(config.get("widen_factor", 4)),
            "dropout": float(config.get("dropout", 0.3)),
            "activation": config.get("activation", "relu"),
            "train_full_official_train_set": True,
            "full_train_images": 50000,
            "validation_loader_used": False,
            "official_test_loader_used": False,
            "from_scratch": True,
            "phase1_base_epochs": int(stage["phase1_epochs"]),
            "phase1_base_lr": float(stage["phase1_base_lr"]),
            "phase1_scheduler": "cosine",
            "phase2_optimizer_reinitialized": True,
            "phase2_fine_tune_epochs": int(stage["phase2_fine_tune_epochs"]),
            "phase2_max_lr": float(stage["phase2_max_lr"]),
            "phase2_min_lr": float(stage["phase2_min_lr"]),
            "phase2_warmup_epochs": int(stage["phase2_warmup_epochs"]),
            "total_epochs": int(stage["phase1_epochs"]) + int(stage["phase2_fine_tune_epochs"]),
            "epoch_budget_reason": (
                "W4 best validation in clean Stage C-lowLR occurred at fine-tune epoch 55."
            ),
            "batch_size": int(config.get("batch_size", 256)),
            "augmentation": {
                "random_crop": True,
                "horizontal_flip": True,
                "randaugment": True,
                "randaugment_num_ops": int(config.get("randaugment_num_ops", 2)),
                "randaugment_magnitude": int(config.get("randaugment_magnitude", 9)),
                "cutout": True,
                "cutout_length": int(config.get("cutout_length", 16)),
            },
            "optimizer": {
                "name": "sgd",
                "momentum": float(config.get("momentum", 0.9)),
                "nesterov": bool(config.get("nesterov", True)),
                "weight_decay": float(config.get("weight_decay", 5e-4)),
            },
            "loss": {
                "name": config.get("loss", "ce_label_smoothing"),
                "label_smoothing": float(config.get("label_smoothing", 0.05)),
            },
        },
        "outputs": {
            "checkpoint": str(stage["checkpoint_path"]),
            "last_checkpoint": str(stage["last_checkpoint_path"]),
            "history_csv": str(stage["history_csv"]),
            "train_curve": str(stage["curve_figure"]),
            "lr_curve": str(stage["lr_figure"]),
        },
    }
    if metrics is not None:
        payload["metrics"] = metrics
    return payload


def run_stage_e(config_path: str | Path | None = None) -> dict[str, Any]:
    stage = dict(STAGE_E_DEFAULTS)
    if config_path is not None:
        stage["config"] = str(config_path)
    _assert_preconditions(stage)

    base_config = load_config(stage["config"])
    config = _stage_e_config(base_config, stage)
    _protocol_guard(config)
    if bool(config.get("ema", False)):
        raise RuntimeError("Stage E final W4 policy requires EMA disabled.")

    set_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", True)))
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
    device = get_device(config.get("device"))
    channels_last = bool(config.get("channels_last", False)) and device.type == "cuda"

    save_json(
        _stage_lock_payload(status="running", config=config, stage=stage),
        stage["lock_json"],
    )

    loaders = get_cifar10_loaders(
        data_dir=str(config.get("data_dir", "./data")),
        batch_size=int(config.get("batch_size", 256)),
        num_workers=int(config.get("num_workers", 4)),
        pin_memory=bool(config.get("pin_memory", True)) and device.type == "cuda",
        persistent_workers=bool(config.get("persistent_workers", True)),
        prefetch_factor=config.get("prefetch_factor"),
        subset_size=None,
        val_subset_size=None,
        seed=int(config.get("seed", 42)),
        split_seed=int(config.get("split_seed", 42)),
        split_file=str(config.get("split_file", DEFAULT_SPLIT_FILE)),
        val_size_per_class=int(config.get("val_size_per_class", 500)),
        train_full=True,
        include_val=False,
        include_test=False,
        train_transform_config=config,
    )
    if loaders.val is not None or loaders.test is not None:
        raise RuntimeError("Stage E must not build validation or official-test loaders.")
    train_images = len(loaders.train.dataset)
    if train_images != 50000:
        raise RuntimeError(f"Stage E expected 50,000 full-train images, got {train_images}.")

    model = _build_model_from_config(config, device)
    criterion = build_criterion(config)
    optimizer_config = dict(config)
    optimizer_config["lr"] = float(stage["phase1_base_lr"])
    optimizer = build_optimizer(model, optimizer_config)
    phase1_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(stage["phase1_epochs"]),
    )
    phase2_scheduler: LowLRFineTuneScheduler | None = None
    phase1_optimizer_state_dict: dict[str, Any] | None = None
    phase1_scheduler_state_dict: dict[str, Any] | None = None
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    checkpoint_path = Path(str(stage["checkpoint_path"]))
    last_checkpoint_path = Path(str(stage["last_checkpoint_path"]))
    ensure_dir(checkpoint_path.parent)
    ensure_dir(Path(str(stage["history_csv"])).parent)

    phase1_epochs = int(stage["phase1_epochs"])
    phase2_epochs = int(stage["phase2_fine_tune_epochs"])
    total_epochs = phase1_epochs + phase2_epochs
    rows: list[dict[str, Any]] = []
    global_step = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    pbar = tqdm(range(1, total_epochs + 1), desc="stageE:final_w4_fulltrain", unit="epoch")
    for epoch in pbar:
        if epoch <= phase1_epochs:
            phase = "base"
            phase_epoch = epoch
            active_scheduler: Any = phase1_scheduler
            lr_start = current_lr(optimizer)
        else:
            if phase2_scheduler is None:
                phase1_optimizer_state_dict = optimizer.state_dict()
                phase1_scheduler_state_dict = phase1_scheduler.state_dict()
                start_lr = max(current_lr(optimizer), float(stage["phase2_fallback_start_lr"]))
                optimizer_config = dict(config)
                optimizer_config["lr"] = start_lr
                optimizer = build_optimizer(model, optimizer_config)
                phase2_scheduler = LowLRFineTuneScheduler(
                    optimizer,
                    start_lr=start_lr,
                    max_lr=float(stage["phase2_max_lr"]),
                    min_lr=float(stage["phase2_min_lr"]),
                    warmup_epochs=int(stage["phase2_warmup_epochs"]),
                    total_epochs=phase2_epochs,
                )
            phase = "low_lr_finetune"
            phase_epoch = epoch - phase1_epochs
            active_scheduler = phase2_scheduler
            lr_start = phase2_scheduler.set_epoch(phase_epoch)

        epoch_start = time.perf_counter()
        train_loss, train_acc, grad_norm = train_one_epoch(
            model,
            loaders.train,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            channels_last=channels_last,
            ema=None,
        )
        global_step += len(loaders.train)
        if phase == "base":
            phase1_scheduler.step()
        lr_end = current_lr(optimizer)
        elapsed_epoch = epoch_time(epoch_start)
        peak_memory = _cuda_memory_gb(device)

        row = {
            "epoch": epoch,
            "phase": phase,
            "phase_epoch": phase_epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "epoch_time": elapsed_epoch,
            "images_per_second": float(train_images / max(elapsed_epoch, 1e-12)),
            "peak_gpu_memory": peak_memory,
            "gpu_peak_memory_gb": peak_memory,
            "grad_norm": grad_norm,
            "total_elapsed_time": time.perf_counter() - started,
        }
        rows.append(row)

        payload = checkpoint_payload(
            model,
            optimizer,
            epoch,
            best_val_acc=None,
            best_val_loss=None,
            config=config,
            selected_weights="raw",
            scheduler=active_scheduler,
            scaler=scaler,
            ema=None,
            global_step=global_step,
        )
        payload.update(
            {
                "stage": "E",
                "phase": phase,
                "phase_epoch": phase_epoch,
                "phase1_optimizer_state_dict": phase1_optimizer_state_dict,
                "phase1_scheduler_state_dict": phase1_scheduler_state_dict,
                "validation_loader_used": False,
                "official_test_loader_used": False,
                "full_train_images": train_images,
            }
        )
        torch.save(payload, last_checkpoint_path)
        pbar.set_postfix(train_acc=f"{train_acc:.4f}", lr=f"{lr_end:.2e}", phase=phase)

    final_payload = checkpoint_payload(
        model,
        optimizer,
        total_epochs,
        best_val_acc=None,
        best_val_loss=None,
        config=config,
        selected_weights="raw",
        scheduler=phase2_scheduler if phase2_scheduler is not None else phase1_scheduler,
        scaler=scaler,
        ema=None,
        global_step=global_step,
    )
    final_payload.update(
        {
            "stage": "E",
            "phase": "low_lr_finetune",
            "phase_epoch": phase2_epochs,
            "phase1_optimizer_state_dict": phase1_optimizer_state_dict,
            "phase1_scheduler_state_dict": phase1_scheduler_state_dict,
            "validation_loader_used": False,
            "official_test_loader_used": False,
            "full_train_images": train_images,
        }
    )
    torch.save(final_payload, checkpoint_path)

    history = pd.DataFrame(rows)
    history.to_csv(stage["history_csv"], index=False)
    _plot_training_curves(history, stage["curve_figure"], phase1_epochs)
    _plot_lr_curve(history, stage["lr_figure"], phase1_epochs)

    final_row = history.iloc[-1]
    metrics = {
        "params": count_parameters(model),
        "full_train_images": train_images,
        "total_epochs": total_epochs,
        "final_train_loss": float(final_row["train_loss"]),
        "final_train_acc": float(final_row["train_acc"]),
        "mean_epoch_time": float(history["epoch_time"].mean()),
        "mean_images_per_second": float(history["images_per_second"].mean()),
        "peak_gpu_memory_gb": float(history["peak_gpu_memory"].max()),
        "total_train_time_seconds": float(history["total_elapsed_time"].iloc[-1]),
        "phase1_final_lr": float(history.loc[history["phase"] == "base", "lr_end"].iloc[-1]),
        "phase2_start_lr": float(history.loc[history["phase"] == "low_lr_finetune", "lr_start"].iloc[0]),
        "phase2_final_lr": float(final_row["lr_end"]),
        "history_csv": str(stage["history_csv"]),
        "checkpoint": str(stage["checkpoint_path"]),
        "last_checkpoint": str(stage["last_checkpoint_path"]),
        "train_curve": str(stage["curve_figure"]),
        "lr_curve": str(stage["lr_figure"]),
        "official_test_used": False,
        "final_selection_lock_created": Path(str(stage["final_selection_lock"])).exists(),
    }
    save_json(
        _stage_lock_payload(status="complete", config=config, stage=stage, metrics=metrics),
        stage["lock_json"],
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage E W4 full-train.")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_stage_e(args.config)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
