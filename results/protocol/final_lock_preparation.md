# Final Lock Preparation

## Status

- Clean SimpleCNN baseline has completed under the strict `train_dev` / `val_dev` protocol.
- BN P0 has completed.
- P1 activation statistics are skipped by default.
- P2 BN placement ablation is skipped by default.
- Official test has not been used.
- `results/protocol/final_selection_lock.json` has now been created after Stage E completion and before official-test evaluation.

## Champion Decision Before Stage E

- Provisional/final Pareto champion: `CIFAR-PreActResNet-W4`.
- W4 is selected as the Pareto winner because W6 improves validation accuracy by only `0.34` percentage points in clean Stage C-lowLR while using about `2.25x` parameters and more memory.
- W6 remains a validation-stage larger candidate and is not selected for Stage E.
- CompactResNetV2 is dominated by W4 and is retained only as validation-stage development evidence.
- Final policy so far: W4, raw weights, no-TTA.

## Current Official Test Row Plan

Official test remains forbidden until the final official-test lock is explicitly created and approved.

1. SimpleCNN baseline
   - checkpoint: `results/checkpoints/baseline_best_val.pt`
   - source: best validation checkpoint
   - checkpoint exists: yes

2. VGG-A no BN
   - checkpoint: `results/checkpoints/bn_p0_vgg_a_best_val.pt`
   - source: best validation checkpoint from BN controlled comparison
   - checkpoint exists: yes

3. VGG-A BN
   - checkpoint: `results/checkpoints/bn_p0_vgg_a_bn_best_val.pt`
   - source: best validation checkpoint from BN controlled comparison
   - checkpoint exists: yes

4. Final W4 full-train champion
   - checkpoint: `results/checkpoints/final_w4_fulltrain_raw.pt`
   - source: locked full-train recipe
   - checkpoint exists after Stage E: yes

CompactResNetV2 and W6 are planned for validation-stage Pareto/development tables, not as primary official-test rows.

## Stage E Completion

- Stage E W4 full-train completed.
- Output checkpoint: `results/checkpoints/final_w4_fulltrain_raw.pt`.
- History CSV: `results/metrics/final_w4_fulltrain_history.csv`.
- Train curve: `results/figures/final_w4_fulltrain_curves.png`.
- LR curve: `results/figures/final_w4_fulltrain_lr_curve.png`.
- `results/protocol/stageE_training_lock.json` status: `complete`.
- Official test remains unused.
- `results/protocol/final_selection_lock.json` has now been created after Stage E completion and before official-test evaluation.

## Locked Stage E Recipe

- Model: `CIFAR-PreActResNet-W4`.
- Data: full CIFAR-10 official train set, `50,000` images.
- Validation loader: not used during full-train.
- Official test: forbidden.
- Initialization: from scratch.
- EMA: disabled.
- TTA: disabled.
- Weight policy: raw weights only.
- Phase 1: `150` epochs base training with the same cosine schedule family as Stage B.
- Phase 2: low-LR fine-tuning with the same schedule family as Stage C-lowLR.
- Fine-tune epochs: `55`.
- Total epochs: `205`.
- Reason: W4 best validation in clean Stage C-lowLR occurred at fine-tune epoch `55`; therefore the full-train epoch budget is locked before official test.

## Expected Stage E Outputs

- `results/checkpoints/final_w4_fulltrain_raw.pt`
- `results/metrics/final_w4_fulltrain_history.csv`
- `results/figures/final_w4_fulltrain_curves.png`
- `results/figures/final_w4_fulltrain_lr_curve.png`
- `results/protocol/stageE_training_lock.json`
