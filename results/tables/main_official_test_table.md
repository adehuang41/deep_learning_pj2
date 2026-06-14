# Main Official Test Results

Final CIFAR-10 official test results. These rows were locked in `results/protocol/final_selection_lock.json` before the single official test run.

| Row | Model | Role | Checkpoint source | Test accuracy | Test error | Test loss |
|---:|---|---|---|---:|---:|---:|
| 1 | SimpleCNN baseline | baseline | best validation checkpoint | 86.72% | 13.28% | 0.3891 |
| 2 | VGG-A no BN | BN control | best validation checkpoint | 87.85% | 12.15% | 0.4783 |
| 3 | VGG-A BN | BN treatment | best validation checkpoint | 89.17% | 10.83% | 0.4291 |
| 4 | Final W4 champion | final champion / Pareto winner | locked full-train recipe | 95.52% | 4.48% | 0.1523 |

Protocol note: no TTA, no EMA, clean/eval transform, no post-test tuning.
