# Unit03 Soft Sensor Regression

本项目用于 Unit03 化工软测量回归实验。数据前 13 列作为输入特征 `X`，最后 1 列作为预测目标 `y`。

`EXPERIMENTS.md` 是实验索引，用来查看每个脚本的作用、运行方式和结果位置。

## 目录说明

```text
unit03_soft_sensor/               # 可复用源码：数据、模型、训练、评价、画图

experiments/                      # 所有实验入口
  baseline/                       # 主 baseline：MLP / GMM / AE+MLP
  aemlp_ablations/                # AE+MLP 专用消融实验
  cross_domain/                   # DVPF 启发的跨工况实验

results/                          # 所有实验结果
  baseline/                       # baseline 结果
  aemlp_ablations/                # AE+MLP 消融结果
  cross_domain/                   # 跨工况实验结果
```

## 运行 baseline

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\baseline\run_baseline.py
```

结果保存到：

```text
results/baseline/
```

当前 baseline 测试集结果：

```text
MLP     RMSE = 1.4514, R² = 0.9116
GMM     RMSE = 1.8824, R² = 0.8513
AE+MLP  RMSE = 1.8861, R² = 0.8507
```

## 运行 AE+MLP 消融实验

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_latent_dim_ablation.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_dropout_ablation.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_latent_dropout_grid.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_finetune_ablation.py
```

结果保存到：

```text
results/aemlp_ablations/
```

## 运行跨工况实验

先画清晰工况图：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\plot_conditions.py
```

再运行三个跨工况模型：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_source_only.py
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_coral_adaptation.py --alignment-weight 1.0
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_mmd_adaptation.py --alignment-weight 0.05
```

结果保存到：

```text
results/cross_domain/
```

当前默认目标工况是 `late_disturbance`，训练时只使用目标工况的 `X`，不使用目标工况的 `y`。目标工况的 `y` 只在最后评价 RMSE、MAE、R² 时使用。

当前跨工况结果：

```text
单目标 late_disturbance：CORAL RMSE = 2.1719, R² = -3.6644
leave-one startup_ramp：CORAL RMSE = 2.1850, R² = 0.8400
leave-one restart_transition：CORAL RMSE = 2.2473, R² = 0.8249
leave-one long_stable：Source-only RMSE = 1.6621, R² = -0.0782
```

跨工况总览图：

```text
results/cross_domain/summary_figures/leave_one_condition_summary.png
```

## 结果解释

`results/baseline/metrics/metrics_summary.csv` 用于比较三个 baseline 模型的同分布测试表现。

`results/cross_domain/comparison_summary.csv` 用于比较 source-only、CORAL、MMD 在目标工况上的跨工况表现。

AE+MLP 消融实验的 summary 文件用于选择更合适的超参数，建议主要根据验证集 RMSE 选择，而不是直接看测试集。

