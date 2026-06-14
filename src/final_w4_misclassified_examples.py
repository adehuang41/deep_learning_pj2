"""Post-hoc visualization of final W4 official-test misclassifications.

This script does not write or modify official metrics. It only extracts
predictions from the locked final W4 checkpoint to render a qualitative figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision import datasets

from .data import build_transforms
from .models import build_model
from .utils import CIFAR10_CLASSES, ensure_dir, get_device, set_seed


CHECKPOINT = Path("results/checkpoints/final_w4_fulltrain_raw.pt")
OUTPUT = Path("results/figures/final_w4_official_misclassified_examples.png")


def _build_model_from_checkpoint(checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    cfg = checkpoint.get("config", {})
    model = build_model(
        str(cfg.get("model")),
        num_classes=10,
        channels=cfg.get("channels", [64, 128, 256]),
        activation=str(cfg.get("activation", "relu")),
        dropout=float(cfg.get("dropout", 0.3)),
        blocks_per_stage=cfg.get("blocks_per_stage"),
        drop_path_rate=cfg.get("drop_path_rate"),
        use_eca=cfg.get("use_eca"),
        depth=cfg.get("depth"),
        widen_factor=cfg.get("widen_factor"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    set_seed(42, deterministic=True)
    device = get_device(None)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model = _build_model_from_checkpoint(checkpoint, device)
    transform = build_transforms(train=False)
    dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=None)
    examples: list[tuple[Any, int, int, float]] = []
    batch_images: list[torch.Tensor] = []
    batch_meta: list[tuple[Any, int]] = []

    with torch.no_grad():
        for image, target in dataset:
            batch_images.append(transform(image))
            batch_meta.append((image, int(target)))
            if len(batch_images) < 256:
                continue
            x = torch.stack(batch_images).to(device)
            probs = torch.softmax(model(x), dim=1)
            conf, pred = probs.max(dim=1)
            for (raw_image, true_label), p, c in zip(batch_meta, pred.cpu().tolist(), conf.cpu().tolist()):
                if int(p) != int(true_label):
                    examples.append((raw_image, int(true_label), int(p), float(c)))
                    if len(examples) >= 25:
                        break
            batch_images.clear()
            batch_meta.clear()
            if len(examples) >= 25:
                break
        if len(examples) < 25 and batch_images:
            x = torch.stack(batch_images).to(device)
            probs = torch.softmax(model(x), dim=1)
            conf, pred = probs.max(dim=1)
            for (raw_image, true_label), p, c in zip(batch_meta, pred.cpu().tolist(), conf.cpu().tolist()):
                if int(p) != int(true_label):
                    examples.append((raw_image, int(true_label), int(p), float(c)))
                    if len(examples) >= 25:
                        break

    if not examples:
        raise RuntimeError("No misclassified examples were found.")

    ensure_dir(OUTPUT.parent)
    cols = 5
    rows = (len(examples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.25), dpi=180)
    flat_axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (image, true_label, pred_label, conf) in zip(flat_axes, examples):
        ax.imshow(image)
        ax.set_title(
            f"T:{CIFAR10_CLASSES[true_label]}\nP:{CIFAR10_CLASSES[pred_label]} ({conf:.2f})",
            fontsize=7,
        )
        ax.axis("off")
    for ax in flat_axes[len(examples) :]:
        ax.axis("off")
    fig.suptitle("Final W4 Official Test Misclassified Examples", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT)
    plt.close(fig)
    print(str(OUTPUT))


if __name__ == "__main__":
    main()
