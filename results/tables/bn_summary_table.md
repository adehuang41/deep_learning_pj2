# BatchNorm P0 Summary Table

| Experiment | No-BN result | BN result | Interpretation |
|---|---:|---:|---|
| Controlled VGG-A best val acc | 88.48% | 89.62% | BN improves validation accuracy by 1.14 pp. |
| Epoch reaching 85% val acc | 22 | 19 | BN reaches the threshold earlier. |
| Epoch reaching 88% val acc | 38 | 35 | BN reaches the threshold earlier. |
| Epoch reaching 89% val acc | not reached | 38 | BN reaches a higher threshold no-BN did not reach. |
| Required LR envelope mean width | 0.7561 | 0.7671 | Do not claim uniform envelope reduction. |
| Required LR envelope final width | 0.9502 | 0.8563 | BN has narrower final envelope width. |
| Required LR envelope max width | 1.2806 | 1.1370 | BN has narrower maximum envelope width. |
| Extended LR largest stable LR | 0.05 | 0.05 | No evidence that BN increases maximum stable LR here. |
| Train-mode scale-invariance loss change | changes strongly | ~1.4e-5 | BN train-mode batch statistics are nearly scale-invariant under conv-weight rescaling. |
| Train-mode scale-invariance flip rate | changes strongly | 0 | Supports the reparameterization interpretation. |
| Directional profile d_grad mean abs loss delta | 0.4370 | 0.3503 | BN is more stable under this gradient-direction probe. |
| Directional profile d_random mean abs loss delta | 0.000234 | 0.000072 | BN is more stable under this random-direction probe. |
| Gradient relative prediction error | 3.3905 | 2.4389 | BN improves local linear prediction error. |
| Gradient cosine similarity | 0.3553 | 0.4352 | BN improves gradient direction consistency. |
| Relative gradient change | 3.0477 | 1.0037 | BN reduces relative gradient change in this probe. |
| Gradient Lipschitz estimate mean | 32.1262 | 32.7714 | Do not claim BN universally lowers Lipschitz estimates. |

Strong supported claim: BN improves VGG-A optimization behavior in this controlled setup. Explicit non-claim: this experiment does not prove universally smoother gradients or larger maximum stable learning rates.
