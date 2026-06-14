"""Stage B validation-only medium training for selected CIFAR-10 models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .evaluate import run_evaluation
from .train import run_training
from .utils import ensure_dir, load_config, save_json


COMMON_STAGE_B_OVERRIDES: dict[str, Any] = {
    "epochs": 150,
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


DEFAULT_STAGE_B: dict[str, Any] = {
    "output_csv": "results/metrics/stageB_top_models.csv",
    "decision_json": "results/protocol/stageB_decision_inputs.json",
    "experiments": [
        {
            "role": "champion_candidate",
            "name": "champion_w6_stageB",
            "config": "configs/champion_w6.yaml",
            "run_flip_tta": True,
        },
        {
            "role": "champion_candidate",
            "name": "champion_w4_stageB",
            "config": "configs/champion_w4.yaml",
            "run_flip_tta": True,
        },
        {
            "role": "efficient",
            "name": "compact_resnet_v2_stageB",
            "config": "configs/compact_resnet_v2.yaml",
            "run_flip_tta": False,
        },
    ],
}


def _best_raw_ema_from_history(history_csv: str) -> dict[str, float | str | None]:
    history = pd.read_csv(history_csv)
    raw_best = float(history["val_acc"].max()) if "val_acc" in history else float("nan")
    ema_best: float | None = None
    if "val_acc_ema" in history and history["val_acc_ema"].notna().any():
        ema_best = float(history["val_acc_ema"].max())
    raw_epoch = int(history.loc[history["val_acc"].idxmax(), "epoch"]) if "val_acc" in history else None
    ema_epoch = None
    if ema_best is not None:
        ema_epoch = int(history.loc[history["val_acc_ema"].idxmax(), "epoch"])
    return {
        "best_val_acc_raw": raw_best,
        "best_val_epoch_raw": raw_epoch,
        "best_val_acc_ema": ema_best,
        "best_val_epoch_ema": ema_epoch,
    }


def _overfitting_summary(history_csv: str) -> dict[str, Any]:
    history = pd.read_csv(history_csv)
    tail = history.tail(min(10, len(history)))
    best_idx = int(history["val_acc"].idxmax())
    best_row = history.iloc[best_idx]
    final_row = history.iloc[-1]
    return {
        "best_train_acc_at_raw_best_val": float(best_row["train_acc"]),
        "best_raw_val_acc": float(best_row["val_acc"]),
        "final_train_acc": float(final_row["train_acc"]),
        "final_val_acc": float(final_row["val_acc"]),
        "final_train_val_gap": float(final_row["train_acc"] - final_row["val_acc"]),
        "last10_val_acc_mean": float(tail["val_acc"].mean()),
        "last10_val_acc_std": float(tail["val_acc"].std(ddof=0)),
        "last10_val_loss_mean": float(tail["val_loss"].mean()),
    }


def run_stage_b(config: dict[str, Any]) -> pd.DataFrame:
    stage = dict(DEFAULT_STAGE_B)
    stage.update(config)
    common = dict(COMMON_STAGE_B_OVERRIDES)
    common.update(stage.get("common_overrides", {}))
    rows: list[dict[str, Any]] = []
    for exp in stage["experiments"]:
        exp_config = load_config(exp["config"])
        exp_config.update(common)
        exp_config["run_name"] = exp["name"]
        exp_config["metrics_filename"] = f"{exp['name']}_results.csv"
        summary = run_training(exp_config)
        raw_ema = _best_raw_ema_from_history(summary["history_csv"])
        overfit = _overfitting_summary(summary["history_csv"])
        no_tta_acc = None
        flip_tta_acc = None
        no_tta_loss = None
        flip_tta_loss = None
        if bool(exp.get("run_flip_tta", False)):
            eval_base = {
                **exp_config,
                "checkpoint": summary["best_checkpoint"],
                "split": "val",
                "official_final_eval": False,
                "save_prefix": f"results/metrics/{exp['name']}_val_no_tta",
                "figures_dir": "results/figures",
            }
            no_tta = run_evaluation({**eval_base, "horizontal_flip_tta": False})
            flip_tta = run_evaluation(
                {
                    **eval_base,
                    "horizontal_flip_tta": True,
                    "save_prefix": f"results/metrics/{exp['name']}_val_flip_tta",
                }
            )
            no_tta_acc = float(no_tta["accuracy"])
            no_tta_loss = float(no_tta["loss_standard_ce"])
            flip_tta_acc = float(flip_tta["accuracy"])
            flip_tta_loss = float(flip_tta["loss_standard_ce"])
        rows.append(
            {
                "stage": "B",
                "role": exp.get("role", ""),
                "experiment": exp["name"],
                "model": summary["model"],
                "params": summary["params"],
                "epochs": summary["epochs"],
                "best_val_acc": summary["best_val_acc"],
                "best_val_error": summary["best_val_error"],
                "best_val_loss": summary["best_val_loss"],
                "best_val_epoch": summary["best_val_epoch"],
                "final_val_acc": summary["final_val_acc"],
                "final_val_loss": summary["final_val_loss"],
                "mean_epoch_time": summary["mean_epoch_time"],
                "mean_images_per_second": summary["mean_images_per_second"],
                "gpu_peak_memory_gb": summary["gpu_peak_memory_gb"],
                "train_time_seconds": summary["train_time_seconds"],
                "selected_weights": summary["selected_weights"],
                "best_val_acc_raw": raw_ema["best_val_acc_raw"],
                "best_val_epoch_raw": raw_ema["best_val_epoch_raw"],
                "best_val_acc_ema": raw_ema["best_val_acc_ema"],
                "best_val_epoch_ema": raw_ema["best_val_epoch_ema"],
                "val_no_tta_acc": no_tta_acc,
                "val_no_tta_loss": no_tta_loss,
                "val_flip_tta_acc": flip_tta_acc,
                "val_flip_tta_loss": flip_tta_loss,
                **overfit,
                "checkpoint": summary["best_checkpoint"],
                "history_csv": summary["history_csv"],
            }
        )
        output_csv = Path(str(stage.get("output_csv", DEFAULT_STAGE_B["output_csv"])))
        ensure_dir(output_csv.parent)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    df = pd.DataFrame(rows).sort_values(
        ["role", "best_val_acc", "params"],
        ascending=[True, False, True],
    )
    output_csv = Path(str(stage.get("output_csv", DEFAULT_STAGE_B["output_csv"])))
    df.to_csv(output_csv, index=False)
    decision_data = {
        "stage": "B",
        "official_test_used": False,
        "common_overrides": common,
        "results_csv": str(output_csv),
        "decision_rules": {
            "w6_over_w4_select_w6_threshold": 0.005,
            "w6_over_w4_select_w4_threshold": 0.003,
            "middle_band": "Use Pareto tradeoff, validation curve, stability, and report narrative.",
        },
    }
    save_json(decision_data, stage.get("decision_json", DEFAULT_STAGE_B["decision_json"]))
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage B validation-only training.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output_csv", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.setdefault("common_overrides", {})["epochs"] = args.epochs
    if args.output_csv is not None:
        cfg["output_csv"] = args.output_csv
    df = run_stage_b(cfg)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
