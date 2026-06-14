# BN P0 Interpretation Summary

## Audit Status

- BN P0 is complete.
- No P1 activation statistics were run.
- No P2 BN placement ablation was run.
- Official test was not used.
- `final_selection_lock.json` has not been created.
- Required LR envelope and extended LR robustness are separated:
  - Required LR envelope raw steps use only base LR values `[1e-4, 5e-4, 1e-3, 2e-3]`.
  - Extended LR robustness uses base LR values `[1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2]`.
  - `5e-2` appears only in `results/metrics/bn_lr_robustness.csv`, not in the required LR envelope raw steps.

## Controlled Comparison Conclusion

VGG-A-BN improves both validation accuracy and convergence behavior under the controlled VGG-A protocol.

- VGG-A best validation accuracy: `0.8848` at epoch `50`.
- VGG-A-BN best validation accuracy: `0.8962` at epoch `48`.
- VGG-A-BN reaches validation accuracy thresholds earlier:
  - `>=0.85`: BN epoch `19`, no-BN epoch `22`.
  - `>=0.88`: BN epoch `35`, no-BN epoch `38`.
  - `>=0.89`: BN epoch `38`; no-BN did not reach this threshold.

Supported claim: BN improves VGG-A validation accuracy and convergence speed in this controlled setup.

## Required LR Envelope Conclusion

The required LR envelope uses only `[1e-4, 5e-4, 1e-3, 2e-3]` as base LR values under the same scheduler rule.

- Mean envelope width:
  - VGG-A: `0.7561`.
  - VGG-A-BN: `0.7671`.
- Final envelope width:
  - VGG-A: `0.9502`.
  - VGG-A-BN: `0.8563`.
- Max envelope width:
  - VGG-A: `1.2806`.
  - VGG-A-BN: `1.1370`.

Interpretation: BN has a narrower final and maximum envelope, but the mean envelope width is not lower. This supports a cautious statement that BN reduces some trajectory-spread measures, not that BN uniformly reduces required LR envelope width.

## Extended LR Robustness Conclusion

Extended LR robustness is a short LR stress test, not a full training sweep. It includes the larger LR values up to `5e-2`.

- Largest stable LR under the numerical divergence criteria:
  - VGG-A: `0.05`.
  - VGG-A-BN: `0.05`.
- BN has higher 5-epoch validation accuracy than no-BN for base LR values from `1e-4` through `2e-2`.
- At `5e-2`, BN has lower validation accuracy than no-BN in this short stress test.

Interpretation: BN improves short-run optimization effectiveness over low-to-mid LR values, but this experiment does not show a larger maximum stable LR for BN.

## Scale-Invariance Conclusion

The scale-invariance probe strongly supports the reparameterization interpretation of BN under train-mode batch statistics.

- VGG-A-BN train-mode batch-statistics probe:
  - mean absolute loss change: about `1.4e-5`.
  - prediction flip rate: `0`.
- VGG-A no-BN changes strongly under conv-weight rescaling.
- VGG-A-BN eval-mode running-statistics behavior also changes strongly under conv-weight rescaling.

Interpretation: BN train-mode batch statistics create strong scale-invariance under conv-weight rescaling. This is evidence for BN as an optimization reparameterization mechanism. The report should distinguish this from eval-mode running-statistics behavior.

## Directional Local Loss Profile Conclusion

The directional local profile uses fixed `train_dev` probe batches as the main optimization-landscape probe. Both `d_grad` and `d_random` are L2-normalized, and rows record `delta_norm`, `weight_norm`, and relative perturbation norm.

At best-val snapshots:

- `d_grad` mean absolute loss delta:
  - VGG-A: `0.4370`.
  - VGG-A-BN: `0.3503`.
- `d_random` mean absolute loss delta:
  - VGG-A: `0.000234`.
  - VGG-A-BN: `0.000072`.

Interpretation: BN shows more stable local directional perturbation behavior in this probe.

## Gradient Predictiveness Conclusion

Gradient predictiveness 2.0 uses fixed `train_dev` probe batches as the main result. BN layers use train-mode batch statistics, while Dropout is disabled/eval mode to avoid dropout noise.

At best-val snapshots:

- Relative prediction error:
  - VGG-A: `3.3905`.
  - VGG-A-BN: `2.4389`.
- Gradient cosine similarity:
  - VGG-A: `0.3553`.
  - VGG-A-BN: `0.4352`.
- Relative gradient change:
  - VGG-A: `3.0477`.
  - VGG-A-BN: `1.0037`.
- Grad Lipschitz estimate:
  - mean VGG-A: `32.1262`.
  - mean VGG-A-BN: `32.7714`.
  - max VGG-A: `41.3750`.
  - max VGG-A-BN: `54.3651`.

Interpretation: BN improves relative prediction error, gradient cosine similarity, and relative gradient change in this probe. However, it does not lower the gradient Lipschitz estimate in this experiment.

## Strong Supported Claims

- BN improves VGG-A validation accuracy and convergence speed.
- BN reaches accuracy thresholds earlier.
- BN train-mode batch statistics create strong scale-invariance under conv-weight rescaling.
- BN shows more stable local directional perturbation behavior.
- BN improves relative prediction error, gradient cosine similarity, and relative gradient change in our probe.

## Explicit Non-Claims

- Do not claim BN uniformly reduces required LR envelope width.
- Do not claim BN has a larger maximum stable LR in this experiment.
- Do not claim BN universally lowers gradient Lipschitz estimates.
- Do not claim BN always makes gradients smoother.

## Limitations

- The LR robustness experiment is a short stress test, not a full training sweep for every LR.
- Gradient and directional probes depend on snapshot choice, perturbation definition, probe batch, and BN/dropout mode.
- BN train-mode scale-invariance is not the same as eval-mode running-statistics behavior.
- These results support an optimization reparameterization story for this controlled VGG-A protocol; they should not be generalized as universal BN claims.
