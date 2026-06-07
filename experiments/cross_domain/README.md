# 跨工况实验

这个目录用于实现受 DVPF 论文启发的跨工况无监督软测量实验。

## 核心思想

把不同运行工况看成不同领域：

```text
source domain：源工况，有 X 和 y，用来监督训练
target domain：目标工况，训练时只使用 X，不使用 y
test：训练结束后，才用目标工况 y 计算 RMSE、MAE、R²
```

当前默认目标工况是：

```text
late_disturbance: 原始样本索引 158000 到 183000
```

## 先画清晰工况图

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\plot_conditions.py
```

输出位置：

```text
results/cross_domain/condition_overview/figures/stacked_normalized_variables.png
results/cross_domain/condition_overview/figures/selected_raw_variables.png
results/cross_domain/condition_overview/metrics/condition_intervals.csv
```

这两张图都直接从 `data/Unit03_1_5_select.csv` 生成，不依赖之前那张截图。

## 运行训练

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_source_only.py
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_coral_adaptation.py --alignment-weight 1.0
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_mmd_adaptation.py --alignment-weight 0.05
```

## 脚本说明

```text
common.py                 # 工况划分、数据标准化、训练循环、CORAL/MMD、画图
plot_conditions.py        # 只画工况图，不训练
run_source_only.py        # 只用源工况标签训练，直接测试目标工况
run_coral_adaptation.py   # 源域回归损失 + CORAL 特征对齐
run_mmd_adaptation.py     # 源域回归损失 + MMD 特征对齐
```

## 结果位置

```text
results/cross_domain/comparison_summary.csv
results/cross_domain/source_only/late_disturbance/
results/cross_domain/coral/late_disturbance/
results/cross_domain/mmd/late_disturbance/
```

每个方法目录下都有：

```text
metrics/summary.csv
metrics/loss_history.csv
figures/loss_curve.png
figures/target_prediction_curve.png
figures/target_true_pred_scatter.png
figures/target_residual.png
figures/feature_tsne.png
figures/stacked_normalized_variables.png
figures/selected_raw_variables.png
models/best_xxx.pt
```

## 当前结果

```text
CORAL       Target RMSE = 2.1719, Target MAE = 1.6640, Target R² = -3.6644
Source-only Target RMSE = 2.2593, Target MAE = 1.7464, Target R² = -4.0473
MMD         Target RMSE = 2.3456, Target MAE = 1.7903, Target R² = -4.4403
```

当前结论：

```text
跨工况测试明显比同分布测试困难很多。CORAL 特征对齐相对 source-only 有一定提升，MMD 在当前权重下没有提升。
这说明目标工况 late_disturbance 与源工况分布差异明显，简单特征对齐只能部分缓解跨工况泛化问题。
```
