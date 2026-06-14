"""Lightweight GPU throughput benchmark for CIFAR-10 training settings."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .data import DEFAULT_SPLIT_FILE, get_cifar10_loaders
from .models import build_model
from .train import build_criterion, build_optimizer, evaluate_model, train_one_epoch
from .utils import count_parameters, ensure_dir, get_device, load_config, merge_config_with_overrides, save_json, set_seed


DEFAULT_BENCHMARK: dict[str, Any] = {
    "model": "compact_resnet_v2",
    "data_dir": "./data",
    "subset_size": 8192,
    "val_subset_size": 1024,
    "benchmark_epochs": 3,
    "batch_sizes": [128, 256, 512],
    "num_workers_list": [4, 8, 12],
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,
    "lr": 0.1,
    "optimizer": "sgd",
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 5e-4,
    "loss": "ce",
    "activation": "silu",
    "dropout": 0.2,
    "channels": [64, 128, 256, 384],
    "blocks_per_stage": [2, 2, 2, 2],
    "drop_path_rate": 0.05,
    "use_eca": True,
    "seed": 42,
    "split_seed": 42,
    "split_file": DEFAULT_SPLIT_FILE,
    "device": None,
    "test_torch_compile": True,
    "cudnn_benchmark": True,
    "output_csv": "results/metrics/gpu_throughput_benchmark.csv",
    "recommendation_json": "results/protocol/gpu_benchmark_recommendation.json",
}


def _gpu_index() -> str | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return None
    return visible.split(",")[0].strip()


def _gpu_utilization() -> float | None:
    gpu = _gpu_index()
    if not gpu:
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return None
    values = [float(line.strip()) for line in output.splitlines() if line.strip()]
    return values[0] if values else None


def _peak_memory_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024**3))


def _build_model(config: dict[str, Any], device: torch.device, channels_last: bool, use_compile: bool) -> torch.nn.Module:
    model = build_model(
        str(config.get("model", "compact_resnet_v2")),
        channels=config.get("channels", [64, 128, 256, 384]),
        blocks_per_stage=config.get("blocks_per_stage", [2, 2, 2, 2]),
        drop_path_rate=config.get("drop_path_rate", 0.05),
        use_eca=config.get("use_eca", True),
        activation=str(config.get("activation", "silu")),
        dropout=float(config.get("dropout", 0.2)),
        depth=config.get("depth"),
        widen_factor=config.get("widen_factor"),
    ).to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if use_compile:
        model = torch.compile(model)
    return model


def run_one_setting(
    config: dict[str, Any],
    batch_size: int,
    num_workers: int,
    amp: bool,
    channels_last: bool,
    torch_compile: bool,
) -> dict[str, Any]:
    set_seed(int(config.get("seed", 42)), deterministic=False)
    torch.backends.cudnn.benchmark = bool(config.get("cudnn_benchmark", True))
    device = get_device(config.get("device"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loaders = get_cifar10_loaders(
        data_dir=str(config.get("data_dir", "./data")),
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        pin_memory=bool(config.get("pin_memory", True)) and device.type == "cuda",
        persistent_workers=bool(config.get("persistent_workers", True)),
        prefetch_factor=config.get("prefetch_factor"),
        subset_size=int(config.get("subset_size", 8192)),
        val_subset_size=int(config.get("val_subset_size", 1024)),
        include_val=True,
        include_test=False,
        split_seed=int(config.get("split_seed", 42)),
        split_file=str(config.get("split_file", DEFAULT_SPLIT_FILE)),
        train_transform_config=config,
        seed=int(config.get("seed", 42)),
    )
    model = _build_model(config, device, channels_last, torch_compile)
    criterion = build_criterion(config)
    optimizer = build_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler() if amp and device.type == "cuda" else None
    epoch_times: list[float] = []
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    start = time.perf_counter()
    status = "ok"
    try:
        for _ in range(int(config.get("benchmark_epochs", 3))):
            epoch_start = time.perf_counter()
            train_loss, train_acc, _grad_norm = train_one_epoch(
                model,
                loaders.train,
                criterion,
                optimizer,
                device,
                scaler=scaler,
                channels_last=channels_last and device.type == "cuda",
                ema=None,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            epoch_times.append(time.perf_counter() - epoch_start)
        if loaders.val is not None:
            val_loss, val_acc = evaluate_model(
                model,
                loaders.val,
                criterion,
                device,
                channels_last=channels_last and device.type == "cuda",
            )
    except Exception as exc:
        status = f"failed:{type(exc).__name__}"
    total = time.perf_counter() - start
    images = int(config.get("subset_size", 8192))
    mean_epoch_time = float(sum(epoch_times) / len(epoch_times)) if epoch_times else float("nan")
    return {
        "model": str(config.get("model", "compact_resnet_v2")),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "amp": bool(amp),
        "channels_last": bool(channels_last),
        "torch_compile": bool(torch_compile),
        "status": status,
        "benchmark_epochs": int(config.get("benchmark_epochs", 3)),
        "subset_size": int(config.get("subset_size", 8192)),
        "val_subset_size": int(config.get("val_subset_size", 1024)),
        "mean_epoch_time": mean_epoch_time,
        "images_per_second": float(images / mean_epoch_time) if mean_epoch_time == mean_epoch_time else float("nan"),
        "total_time_seconds": float(total),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc_after_smoke": float(val_acc),
        "peak_memory_gb": _peak_memory_gb(device),
        "gpu_utilization_percent": _gpu_utilization(),
        "params": count_parameters(model),
    }


def benchmark_matrix(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = [(False, False), (True, False), (True, True)]
    for batch_size, num_workers, (amp, channels_last) in product(
        config.get("batch_sizes", [128, 256, 512]),
        config.get("num_workers_list", [4, 8, 12]),
        variants,
    ):
        rows.append(
            run_one_setting(
                config,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                amp=bool(amp),
                channels_last=bool(channels_last),
                torch_compile=False,
            )
        )
        df = pd.DataFrame(rows)
        output_csv = Path(str(config.get("output_csv", DEFAULT_BENCHMARK["output_csv"])))
        ensure_dir(output_csv.parent)
        df.to_csv(output_csv, index=False)

    df = pd.DataFrame(rows)
    ok = df[df["status"] == "ok"].sort_values("images_per_second", ascending=False)
    compile_row = None
    if bool(config.get("test_torch_compile", True)) and not ok.empty:
        best = ok.iloc[0]
        compile_row = run_one_setting(
            config,
            batch_size=int(best["batch_size"]),
            num_workers=int(best["num_workers"]),
            amp=bool(best["amp"]),
            channels_last=bool(best["channels_last"]),
            torch_compile=True,
        )
        rows.append(compile_row)
    df = pd.DataFrame(rows)
    output_csv = Path(str(config.get("output_csv", DEFAULT_BENCHMARK["output_csv"])))
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    ok = df[df["status"] == "ok"].sort_values("images_per_second", ascending=False)
    recommendation = ok.iloc[0].to_dict() if not ok.empty else {}
    recommendation["source_csv"] = str(output_csv)
    save_json(recommendation, config.get("recommendation_json", DEFAULT_BENCHMARK["recommendation_json"]))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CIFAR-10 training throughput.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output_csv", type=str)
    parser.add_argument("--benchmark_epochs", type=int)
    parser.add_argument("--subset_size", type=int)
    parser.add_argument("--val_subset_size", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--no_torch_compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = merge_config_with_overrides(load_config(args.config), {}, DEFAULT_BENCHMARK)
    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}
    if overrides.pop("no_torch_compile", False):
        overrides["test_torch_compile"] = False
    config.update(overrides)
    df = benchmark_matrix(config)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
