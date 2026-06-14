# Model Efficiency Table

Training/runtime metrics are taken from the clean active runs. Official test accuracy is included only for the four locked final-test rows.

| Model | Params | Train time | Mean epoch time | Images/sec | Peak memory | Official test acc |
|---|---:|---:|---:|---:|---:|---:|
| SimpleCNN baseline | 620,362 | 530.49 s | 10.58 s | 4251.99 | 0.09 GiB | 86.72% |
| VGG-A no BN | 9,750,922 | 531.92 s | 10.39 s | 4332.52 | 0.27 GiB | 87.85% |
| VGG-A BN | 9,753,674 | 533.16 s | 10.41 s | 4325.23 | 0.33 GiB | 89.17% |
| Final W4 full-train champion | 5,849,050 | 4950.79 s | 23.96 s | 2086.65 | 3.41 GiB | 95.52% |

Note: final W4 uses full-train Stage E with 205 total epochs; the other rows are validation-controlled development checkpoints.
