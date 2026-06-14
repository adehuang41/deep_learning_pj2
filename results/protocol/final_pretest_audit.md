# Final Pre-Test Audit

## Status

- `results/protocol/final_selection_lock.json` exists.
- Official test has not been run before this audit.
- Official test must not be used for tuning, model selection, TTA selection, EMA selection, or adding/removing rows after this lock.
- W6 and CompactResNetV2 are excluded from official-test rows and retained only as validation-stage development evidence.

## Optional W4 Sanity Check

- Checkpoint: `results/checkpoints/final_w4_fulltrain_raw.pt`.
- Split used: CIFAR-10 official train set with eval/clean transform.
- Official test used: no.
- Used for model selection: no.
- Number of examples: `50000`.
- `train_clean_loss`: `0.0692079669`.
- `train_clean_acc`: `0.9857`.
- Metrics path: `results/metrics/final_w4_fulltrain_train_clean_sanity.json`.

## Locked Official-Test Rows

1. SimpleCNN baseline
   - checkpoint: `results/checkpoints/baseline_best_val.pt`
   - source: best validation checkpoint
   - role: baseline
   - checkpoint exists and loads: yes

2. VGG-A no BN
   - checkpoint: `results/checkpoints/bn_p0_vgg_a_best_val.pt`
   - source: best validation checkpoint
   - role: BN control
   - checkpoint exists and loads: yes

3. VGG-A BN
   - checkpoint: `results/checkpoints/bn_p0_vgg_a_bn_best_val.pt`
   - source: best validation checkpoint
   - role: BN treatment
   - checkpoint exists and loads: yes

4. Final W4 full-train champion
   - checkpoint: `results/checkpoints/final_w4_fulltrain_raw.pt`
   - source: locked full-train recipe
   - role: final champion / Pareto winner
   - checkpoint exists and loads: yes

## Lock Field Audit

Required fields present:

- `official_test_used_before_lock`
- `will_not_tune_after_official_test`
- `ablation_uses_official_test`
- `tta_policy`
- `ema_policy`
- `final_champion`
- `final_champion_reason`
- `stageE_recipe`
- `official_test_rows`

Locked policies:

- `official_test_used_before_lock`: `false`
- `will_not_tune_after_official_test`: `true`
- `ablation_uses_official_test`: `false`
- `tta_policy`: `no-TTA`
- `ema_policy`: `no EMA / raw weights`
- `final_champion`: `CIFAR-PreActResNet-W4`

## Leakage Audit

Active results scan command:

`rg -n "best_test|test_acc|official_test_evaluated['\" ]*[:=]['\" ]*true|official_test_used['\" ]*[:=]['\" ]*true|official_test_accuracy|official_test_loss_standard_ce|split['\" ]*[:=]['\" ]*test" results --glob '!results/archive_test_leakage_old_run/**' --glob '!**/*.pt'`

Result: no matches in active results.

Note: `results/archive_test_leakage_old_run` still contains old historical test-leakage artifacts. They are excluded from active results and should not be used in the report except, if needed, as a protocol caution.

## Core Report Artifact Audit

- Baseline metrics: `results/metrics/baseline_results.csv`
- Baseline figure: `results/figures/baseline_train_val_curves.png`
- Stage E W4 metrics: `results/metrics/final_w4_fulltrain_history.csv`
- Stage E W4 train curve: `results/figures/final_w4_fulltrain_curves.png`
- Stage E W4 LR curve: `results/figures/final_w4_fulltrain_lr_curve.png`
- Stage E W4 checkpoint: `results/checkpoints/final_w4_fulltrain_raw.pt`
- BN P0 metrics:
  - `results/metrics/bn_controlled_vgga_comparison.csv`
  - `results/metrics/bn_required_loss_envelope.csv`
  - `results/metrics/bn_lr_robustness.csv`
  - `results/metrics/bn_scale_invariance.csv`
  - `results/metrics/bn_directional_loss_profile.csv`
  - `results/metrics/bn_gradient_predictiveness_v2.csv`
- BN P0 figures:
  - `results/figures/bn_training_dynamics_dashboard.png`
  - `results/figures/bn_required_loss_envelope.png`
  - `results/figures/bn_lr_stability_heatmap.png`
  - `results/figures/bn_scale_invariance_loss.png`
  - `results/figures/bn_directional_loss_profile.png`
  - `results/figures/bn_gradient_predictiveness_vs_alpha.png`
  - `results/figures/bn_gradient_cosine_similarity.png`
  - `results/figures/bn_gradient_lipschitz_violin.png`
- Final lock file: `results/protocol/final_selection_lock.json`

All listed core artifacts exist.
