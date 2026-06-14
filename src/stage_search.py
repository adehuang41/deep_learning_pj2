"""Validation-only successive-halving style model search stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .train import run_training
from .utils import ensure_dir, load_config


DEFAULT_STAGE_A: dict[str, Any] = {
    "epochs": 50,
    "output_csv": "results/metrics/stageA_val_search.csv",
    "experiments": [
        {
            "role": "efficient",
            "name": "compact_resnet_v2_stageA",
            "config": "configs/compact_resnet_v2.yaml",
        },
        {
            "role": "champion_candidate",
            "name": "champion_w4_stageA",
            "config": "configs/champion_w4.yaml",
        },
        {
            "role": "champion_candidate",
            "name": "champion_w6_stageA",
            "config": "configs/champion_w6.yaml",
        },
    ],
}


def run_stage_a(config: dict[str, Any]) -> pd.DataFrame:
    stage = dict(DEFAULT_STAGE_A)
    stage.update(config)
    rows: list[dict[str, Any]] = []
    for exp in stage["experiments"]:
        exp_config = load_config(exp["config"])
        exp_config["run_name"] = exp["name"]
        exp_config["epochs"] = int(exp.get("epochs", stage.get("epochs", 50)))
        exp_config["metrics_filename"] = f"{exp['name']}_results.csv"
        exp_config["checkpoint_selection_split"] = "val"
        exp_config["use_test_during_training"] = False
        exp_config["train_split"] = "dev"
        summary = run_training(exp_config)
        rows.append(
            {
                "stage": "A",
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
                "checkpoint": summary["best_checkpoint"],
                "history_csv": summary["history_csv"],
            }
        )
        output_csv = Path(str(stage.get("output_csv", DEFAULT_STAGE_A["output_csv"])))
        ensure_dir(output_csv.parent)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
    df = pd.DataFrame(rows).sort_values(
        ["best_val_acc", "params", "mean_epoch_time"],
        ascending=[False, True, True],
    )
    output_csv = Path(str(stage.get("output_csv", DEFAULT_STAGE_A["output_csv"])))
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A validation search.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output_csv", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.output_csv is not None:
        cfg["output_csv"] = args.output_csv
    df = run_stage_a(cfg)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
