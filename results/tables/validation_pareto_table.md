# Validation-Stage Pareto Table

Validation-stage rows were selected using `train_dev / val_dev` only. They are not additional official-test rows.

| Model / run | Role | Best val acc | Best val epoch | Params | Mean epoch time | Images/sec | Peak memory | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CompactResNetV2 Stage B | efficient candidate | 94.64% | 138 | 7,747,170 | 23.73 s | 1896.67 | 4.98 GiB | Dominated by W4; development evidence only. |
| W4 Stage B | champion candidate | 95.86% | 148 | 5,849,050 | 23.56 s | 1909.73 | 3.43 GiB | Continued into clean low-LR Stage C. |
| W6 Stage B | larger candidate | 96.16% | 142 | 13,144,794 | 23.64 s | 1903.53 | 5.56 GiB | Continued into clean low-LR Stage C. |
| W4 Stage C-lowLR | provisional/final Pareto champion | 95.98% | 55 fine-tune | 5,849,050 | 23.25 s | 1935.68 | 3.47 GiB | Selected for Stage E. |
| W6 Stage C-lowLR | larger candidate | 96.32% | 43 fine-tune | 13,144,794 | 23.81 s | 1890.44 | 5.67 GiB | Not selected: +0.34 pp val acc but 2.25x params and more memory. |

Pareto decision: W4 was selected because W6's validation gain was only 0.34 percentage points while requiring 2.25x parameters and more memory.
