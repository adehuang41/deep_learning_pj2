# 神经网络与深度学习 Project 2

本仓库对应复旦大学《神经网络与深度学习》课程 Project 2，主题为 CIFAR-10 图像分类、模型结构比较与 Batch Normalization 分析。

仓库中保留了：

- 项目代码
- 最终中文 / 英文报告 PDF
- 报告中使用的小型结果图表与 CSV / JSON

仓库中不包含：

- CIFAR-10 数据集原始文件
- 训练得到的大型模型权重 / checkpoint
- 报告 LaTeX 编译中间文件
- 大型训练目录、日志和中间缓存文件

模型权重与 checkpoint 已单独上传到 ModelScope，并在报告中提供链接。

## 仓库结构

```text
configs/
```

训练配置文件。

```text
scripts/
```

主要训练、评估和分析入口脚本。

```text
src/
```

数据划分、模型定义、训练、评估、BatchNorm 分析与可视化代码。

```text
splits/
```

CIFAR-10 训练集划分文件。

```text
results/figures/
```

报告中使用的结果图。

```text
results/metrics/
```

报告中使用的实验结果 CSV / JSON。

```text
results/tables/
```

报告中使用的汇总表格。

## 主要实验内容

### Model Comparison

- 训练 SimpleCNN baseline
- 训练并比较 VGG-A 与 VGG-A BatchNorm
- 训练并比较若干 CIFAR-10 residual-style 模型
- 基于验证集结果、模型规模和训练成本选择最终模型

### Final Evaluation

- 在模型选择完成后锁定最终评估设置
- 使用完整 CIFAR-10 official train set 训练最终模型
- 在 official test set 上进行一次最终测试

### Batch Normalization Study

- 比较 VGG-A with / without BatchNorm
- 分析训练动态、学习率稳定性、scale invariance、loss profile 与 gradient behavior
- 生成报告中 BatchNorm 部分使用的图表和汇总结果

## 主要脚本

```text
scripts/run_baseline.sh
```

运行 SimpleCNN baseline。

```text
scripts/run_stageA_search.sh
scripts/run_stageB.sh
scripts/run_stageC.sh
scripts/run_stageC_lowLR.sh
```

运行验证阶段模型筛选与后续比较实验。

```text
scripts/run_stageE_w4_fulltrain.sh
```

运行最终模型的 full-train 训练。

```text
scripts/run_final_official_test.sh
```

运行最终 official test 评估。

```text
scripts/run_bn_p0.sh
```

运行 BatchNorm 分析实验。

## 数据与权重说明

GitHub 仓库不上传 CIFAR-10 数据集与大型模型权重。

数据集请参考课程要求或 CIFAR-10 官方页面：

```text
https://www.cs.toronto.edu/~kriz/cifar.html
```

训练权重 / checkpoint 请参考报告中的 ModelScope 链接：

```text
https://www.modelscope.cn/models/adehuang41/deep_learning_pj2
```

## 报告

最终报告文件直接放在仓库根目录：

```text
main.pdf
main_en.pdf
```

## 提交说明

本仓库用于提供：

- 课程项目代码
- 复现实验所需的小型结果文件
- 最终报告 PDF

最终报告中的 GitHub 链接为：

```text
https://github.com/adehuang41/deep_learning_pj2
```
