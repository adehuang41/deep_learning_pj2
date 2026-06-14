"""Training entry point for validation-controlled CIFAR-10 experiments."""

from __future__ import annotations

import argparse
import copy
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .data import DEFAULT_SPLIT_FILE, get_cifar10_loaders
from .models import build_model
from .utils import (
    AverageMeter,
    accuracy_from_logits,
    count_parameters,
    current_lr,
    ensure_dir,
    epoch_time,
    get_device,
    infer_run_name,
    load_config,
    merge_config_with_overrides,
    save_json,
    set_seed,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "model": "compact_resnet",
    "run_name": None,
    "data_dir": "./data",
    "batch_size": 128,
    "num_workers": 4,
    "pin_memory": True,
    "persistent_workers": False,
    "prefetch_factor": None,
    "subset_size": None,
    "val_subset_size": None,
    "split_seed": 42,
    "split_file": DEFAULT_SPLIT_FILE,
    "val_size_per_class": 500,
    "train_split": "dev",
    "checkpoint_selection_split": "val",
    "use_test_during_training": False,
    "epochs": 100,
    "optimizer": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "nesterov": False,
    "weight_decay": 5e-4,
    "scheduler": "cosine",
    "scheduler_t_max": None,
    "stop_epoch": None,
    "step_size": 30,
    "gamma": 0.1,
    "loss": "ce",
    "label_smoothing": 0.0,
    "focal_gamma": 2.0,
    "activation": "silu",
    "dropout": 0.2,
    "channels": [64, 128, 256],
    "seed": 42,
    "device": None,
    "amp": False,
    "channels_last": False,
    "cudnn_benchmark": False,
    "deterministic": True,
    "ema": False,
    "ema_decay": 0.999,
    "save_dir": "results/checkpoints",
    "log_dir": "results/logs",
    "metrics_dir": "results/metrics",
    "protocol_dir": "results/protocol",
    "metrics_filename": None,
    "best_checkpoint_name": None,
    "last_checkpoint_name": None,
    "resume_from": None,
    "resume_history_csv": None,
    "full_train_requires_lock": True,
    "final_selection_lock": "results/protocol/final_selection_lock.json",
}


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_state = model.state_dict()
        ema_state = self.module.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(model_value)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        loss = ((1.0 - pt).clamp_min(0.0) ** self.gamma) * ce_loss
        return loss.mean()


def build_criterion(config: dict[str, Any]) -> nn.Module:
    loss_name = str(config.get("loss", "ce")).lower()
    if loss_name == "ce_label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.1)))
    if loss_name == "focal":
        return FocalLoss(
            gamma=float(config.get("focal_gamma", 2.0)),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
        )
    return nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    opt = str(config.get("optimizer", "sgd")).lower()
    lr = float(config.get("lr", 0.1))
    weight_decay = float(config.get("weight_decay", 0.0))
    if opt == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(config.get("nesterov", False)),
        )
    if opt == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if opt == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.LRScheduler | None:
    scheduler = str(config.get("scheduler", "none")).lower()
    if scheduler == "none":
        return None
    if scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 30)),
            gamma=float(config.get("gamma", 0.1)),
        )
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("scheduler_t_max") or config.get("epochs", 100)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def _cuda_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**3))


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


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    channels_last: bool = False,
    ema: ModelEMA | None = None,
) -> tuple[float, float, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    grad_norm_meter = AverageMeter()
    amp_enabled = scaler is not None and device.type == "cuda"
    for x, y in loader:
        x, y = _to_device(x, y, device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        batch_size = y.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        acc_meter.update(accuracy_from_logits(logits.detach(), y), batch_size)
        grad_norm_meter.update(float(grad_norm), batch_size)
    return loss_meter.avg, acc_meter.avg, grad_norm_meter.avg


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    if "python_random_state" in state:
        random.setstate(state["python_random_state"])
    if "numpy_random_state" in state:
        np.random.set_state(state["numpy_random_state"])
    if "torch_rng_state" in state:
        torch_state = state["torch_rng_state"]
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.cpu()
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in state:
        try:
            cuda_states = [
                cuda_state.cpu() if isinstance(cuda_state, torch.Tensor) else cuda_state
                for cuda_state in state["torch_cuda_rng_state_all"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)
        except RuntimeError:
            # A checkpoint may have been saved with a different CUDA visibility mask.
            pass


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    channels_last: bool = False,
) -> tuple[float, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for x, y in loader:
        x, y = _to_device(x, y, device, channels_last)
        logits = model(x)
        loss = criterion(logits, y)
        batch_size = y.size(0)
        loss_meter.update(float(loss.item()), batch_size)
        acc_meter.update(accuracy_from_logits(logits, y), batch_size)
    return loss_meter.avg, acc_meter.avg


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float | None,
    best_val_loss: float | None,
    config: dict[str, Any],
    selected_weights: str = "raw",
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    ema: ModelEMA | None = None,
    global_step: int = 0,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "ema_state_dict": ema.module.state_dict() if ema is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "selection_metric": "validation accuracy" if best_val_acc is not None else "none",
        "selection_tie_breaker": "validation loss" if best_val_acc is not None else "none",
        "selected_weights": selected_weights,
        "uses_validation_for_selection": best_val_acc is not None,
        "uses_test_during_training": False,
        "official_test_evaluated": False,
        "config": dict(config),
        "rng_state": capture_rng_state(),
    }


def _is_better_val(
    val_acc: float,
    val_loss: float,
    best_val_acc: float,
    best_val_loss: float,
) -> bool:
    if val_acc > best_val_acc:
        return True
    if val_acc == best_val_acc and val_loss < best_val_loss:
        return True
    return False


def _protocol_guard(config: dict[str, Any]) -> None:
    if bool(config.get("use_test_during_training", False)):
        raise ValueError("Training is not allowed to use official_test.")
    train_split = str(config.get("train_split", "dev")).lower()
    selection_split = str(config.get("checkpoint_selection_split", "val")).lower()
    if train_split == "dev" and selection_split != "val":
        raise ValueError("Development training must select checkpoints on val_dev.")
    if train_split == "full":
        if selection_split not in {"none", "last"}:
            raise ValueError("Full-train runs must not perform validation checkpoint selection.")
        lock_path = Path(str(config.get("final_selection_lock", "")))
        if bool(config.get("full_train_requires_lock", True)) and not lock_path.exists():
            raise FileNotFoundError(f"Full-train requires final selection lock: {lock_path}")
    if train_split not in {"dev", "full"}:
        raise ValueError(f"Unsupported train_split: {train_split}")


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    config = merge_config_with_overrides(config, {}, DEFAULT_CONFIG)
    _protocol_guard(config)
    run_name = infer_run_name(config)
    config["run_name"] = run_name
    set_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", True)))
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", False))
    device = get_device(config.get("device"))
    channels_last = bool(config.get("channels_last", False)) and device.type == "cuda"
    train_split = str(config.get("train_split", "dev")).lower()
    is_full_train = train_split == "full"

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
        val_size_per_class=int(config.get("val_size_per_class", 500)),
        train_full=is_full_train,
        include_val=not is_full_train,
        include_test=False,
        train_transform_config=config,
    )
    if loaders.test is not None:
        raise RuntimeError("Training loader unexpectedly includes official_test.")
    if not is_full_train and loaders.val is None:
        raise RuntimeError("Development training requires val_dev loader.")

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
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    ema = ModelEMA(model, decay=float(config.get("ema_decay", 0.999))) if bool(config.get("ema", False)) else None

    save_dir = ensure_dir(config.get("save_dir", "results/checkpoints"))
    log_dir = ensure_dir(config.get("log_dir", "results/logs"))
    metrics_dir = ensure_dir(config.get("metrics_dir", "results/metrics"))
    protocol_dir = ensure_dir(config.get("protocol_dir", "results/protocol"))
    best_path = save_dir / f"{run_name}_best_val.pt"
    best_ema_path = save_dir / f"{run_name}_best_val_ema.pt"
    last_path = save_dir / f"{run_name}_last.pt"
    ema_last_path = save_dir / f"{run_name}_last_ema.pt"
    log_path = log_dir / f"{run_name}_history.csv"
    summary_path = metrics_dir / str(config.get("metrics_filename") or f"{run_name}_results.csv")

    rows: list[dict[str, Any]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_val_epoch = 0
    best_weights = "raw"
    global_step = 0
    start_epoch = 1
    resume_from = config.get("resume_from")
    resume_loaded = False
    if resume_from:
        resume_path = Path(str(resume_from))
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if ema is not None:
            if checkpoint.get("ema_state_dict") is not None:
                ema.module.load_state_dict(checkpoint["ema_state_dict"])
            else:
                ema.module.load_state_dict(model.state_dict())
        best_val_acc = (
            float(checkpoint["best_val_acc"])
            if checkpoint.get("best_val_acc") is not None
            else -1.0
        )
        best_val_loss = (
            float(checkpoint["best_val_loss"])
            if checkpoint.get("best_val_loss") is not None
            else float("inf")
        )
        best_weights = str(checkpoint.get("selected_weights") or "raw")
        global_step = int(checkpoint.get("global_step") or 0)
        start_epoch = int(checkpoint.get("epoch") or 0) + 1
        restore_rng_state(checkpoint.get("rng_state"))
        resume_loaded = True

        resume_history = config.get("resume_history_csv") or (str(log_path) if log_path.exists() else None)
        if resume_history and Path(str(resume_history)).exists():
            previous = pd.read_csv(str(resume_history))
            if "epoch" in previous.columns:
                previous = previous[previous["epoch"].astype(int) < start_epoch]
                rows.extend(previous.to_dict("records"))
                if rows and best_val_acc >= 0 and "selected_val_acc" in previous.columns:
                    selected = pd.to_numeric(previous["selected_val_acc"], errors="coerce")
                    if selected.notna().any():
                        best_val_epoch = int(previous.loc[selected.idxmax(), "epoch"])

    start = time.perf_counter()
    epochs = int(config.get("epochs", 100))
    stop_epoch = int(config.get("stop_epoch") or epochs)
    if stop_epoch > epochs:
        raise ValueError(f"stop_epoch ({stop_epoch}) cannot exceed epochs ({epochs})")
    pbar = tqdm(range(start_epoch, stop_epoch + 1), desc=f"train:{run_name}", unit="epoch")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in pbar:
        epoch_start = time.perf_counter()
        lr_start = current_lr(optimizer)
        train_loss, train_acc, grad_norm = train_one_epoch(
            model,
            loaders.train,
            criterion,
            optimizer,
            device,
            scaler=scaler,
            channels_last=channels_last,
            ema=ema,
        )
        global_step += len(loaders.train)
        val_loss = float("nan")
        val_acc = float("nan")
        val_loss_ema = float("nan")
        val_acc_ema = float("nan")
        if loaders.val is not None:
            val_loss, val_acc = evaluate_model(
                model,
                loaders.val,
                criterion,
                device,
                channels_last=channels_last,
            )
            candidate_acc = val_acc
            candidate_loss = val_loss
            candidate_model = model
            candidate_weights = "raw"
            if ema is not None:
                val_loss_ema, val_acc_ema = evaluate_model(
                    ema.module,
                    loaders.val,
                    criterion,
                    device,
                    channels_last=channels_last,
                )
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
                torch.save(
                    checkpoint_payload(
                        candidate_model,
                        optimizer,
                        epoch,
                        best_val_acc,
                        best_val_loss,
                        config,
                        selected_weights=candidate_weights,
                        scheduler=scheduler,
                        scaler=scaler,
                        ema=ema,
                        global_step=global_step,
                    ),
                    best_ema_path if candidate_weights == "ema" else best_path,
                )

        elapsed_epoch = epoch_time(epoch_start)
        train_images = len(loaders.train.dataset)
        if scheduler is not None:
            scheduler.step()
        lr_end = current_lr(optimizer)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_loss_ema": val_loss_ema,
            "val_acc_ema": val_acc_ema,
            "selected_val_acc": best_val_acc if best_val_acc >= 0 else float("nan"),
            "selected_weights": best_weights if best_val_acc >= 0 else "none",
            "lr": lr_end,
            "lr_start": lr_start,
            "lr_end": lr_end,
            "grad_norm": grad_norm,
            "epoch_time": elapsed_epoch,
            "images_per_second": float(train_images / max(elapsed_epoch, 1e-12)),
            "gpu_peak_memory_gb": _cuda_memory_gb(device),
            "peak_gpu_memory": _cuda_memory_gb(device),
            "total_elapsed_time": time.perf_counter() - start,
        }
        rows.append(row)
        torch.save(
            checkpoint_payload(
                model,
                optimizer,
                epoch,
                best_val_acc if best_val_acc >= 0 else None,
                best_val_loss if best_val_acc >= 0 else None,
                config,
                selected_weights="raw",
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                global_step=global_step,
            ),
            last_path,
        )
        if ema is not None:
            torch.save(
                checkpoint_payload(
                    ema.module,
                    optimizer,
                    epoch,
                    best_val_acc if best_val_acc >= 0 else None,
                    best_val_loss if best_val_acc >= 0 else None,
                    config,
                    selected_weights="ema",
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    global_step=global_step,
                ),
                ema_last_path,
            )
        postfix = {"train_acc": f"{train_acc:.4f}"}
        if loaders.val is not None:
            postfix["val_acc"] = f"{val_acc:.4f}"
            postfix["best_val"] = f"{best_val_acc:.4f}"
        pbar.set_postfix(**postfix)

    history = pd.DataFrame(rows)
    if history.empty:
        raise RuntimeError("Training produced no history rows.")
    history.to_csv(log_path, index=False)
    export_best = config.get("best_checkpoint_name")
    export_best_path = None
    selected_best_path = best_ema_path if best_weights == "ema" else best_path
    if export_best and selected_best_path.exists():
        export_best_path = save_dir / str(export_best)
        shutil.copy2(selected_best_path, export_best_path)
    export_last = config.get("last_checkpoint_name")
    export_last_path = None
    if export_last:
        export_last_path = save_dir / str(export_last)
        shutil.copy2(ema_last_path if bool(config.get("ema", False)) else last_path, export_last_path)

    total_time = time.perf_counter() - start
    final_row = history.iloc[-1].to_dict()
    summary = {
        "run_name": run_name,
        "model": config.get("model"),
        "train_split": train_split,
        "params": count_parameters(model),
        "epochs": epochs,
        "start_epoch": start_epoch,
        "stop_epoch": stop_epoch,
        "scheduler_t_max": int(config.get("scheduler_t_max") or epochs),
        "resume_from": str(resume_from) if resume_from else None,
        "resume_loaded": resume_loaded,
        "optimizer": config.get("optimizer"),
        "scheduler": config.get("scheduler"),
        "lr": float(config.get("lr", 0.0)),
        "weight_decay": float(config.get("weight_decay", 0.0)),
        "activation": config.get("activation"),
        "dropout": float(config.get("dropout", 0.0)),
        "best_val_epoch": best_val_epoch if best_val_epoch else None,
        "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
        "best_val_error": 1.0 - best_val_acc if best_val_acc >= 0 else None,
        "best_val_loss": best_val_loss if best_val_acc >= 0 else None,
        "selected_weights": best_weights if best_val_acc >= 0 else "last",
        "final_val_acc": float(final_row.get("val_acc", float("nan"))),
        "final_val_loss": float(final_row.get("val_loss", float("nan"))),
        "final_train_acc": float(final_row.get("train_acc", float("nan"))),
        "final_train_loss": float(final_row.get("train_loss", float("nan"))),
        "mean_epoch_time": float(history["epoch_time"].mean()),
        "mean_images_per_second": float(history["images_per_second"].mean()),
        "gpu_peak_memory_gb": float(history["gpu_peak_memory_gb"].max()),
        "train_time_seconds": total_time,
        "uses_validation_for_selection": not is_full_train,
        "uses_test_during_training": False,
        "official_test_evaluated": False,
        "checkpoint_selection_split": "val" if not is_full_train else "none",
        "history_csv": str(log_path),
        "best_checkpoint": str(export_best_path or selected_best_path) if not is_full_train else None,
        "last_checkpoint": str(export_last_path or (ema_last_path if bool(config.get("ema", False)) else last_path)),
        "split_file": str(config.get("split_file", DEFAULT_SPLIT_FILE)),
    }
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    save_json(
        {
            "run_name": run_name,
            "train_split": train_split,
            "uses_validation_for_selection": not is_full_train,
            "uses_test_during_training": False,
            "official_test_evaluated": False,
            "checkpoint_selection_split": "val" if not is_full_train else "none",
            "summary_csv": str(summary_path),
            "scheduler_t_max": int(config.get("scheduler_t_max") or epochs),
            "stop_epoch": stop_epoch,
            "resume_from": str(resume_from) if resume_from else None,
            "resume_loaded": resume_loaded,
        },
        protocol_dir / f"{run_name}_protocol.json",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model",
        choices=["simple_cnn", "compact_resnet", "compact_resnet_v2", "vgg_a", "vgg_a_bn", "preact_resnet_cifar"],
    )
    parser.add_argument("--optimizer", choices=["sgd", "adamw", "adam"])
    parser.add_argument("--activation", choices=["relu", "leaky_relu", "silu"])
    parser.add_argument("--loss", choices=["ce", "ce_label_smoothing", "focal"])
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument("--focal_gamma", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop_epoch", type=int)
    parser.add_argument("--scheduler_t_max", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split_seed", type=int)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--prefetch_factor", type=int)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--val_subset_size", type=int)
    parser.add_argument("--train_split", choices=["dev", "full"])
    parser.add_argument("--checkpoint_selection_split", choices=["val", "none", "last"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--cudnn_benchmark", action="store_true")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--metrics_dir", type=str)
    parser.add_argument("--protocol_dir", type=str)
    parser.add_argument("--resume_from", type=str)
    parser.add_argument("--resume_history_csv", type=str)
    parser.add_argument("--run_name", type=str)
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
    merged = merge_config_with_overrides(config, overrides, DEFAULT_CONFIG)
    summary = run_training(merged)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
