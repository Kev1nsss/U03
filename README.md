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
    metrics/
    figures/
    models/
  aemlp_ablations/                # AE+MLP 消融结果
  cross_domain/                   # 跨工况实验结果
```

## 运行 baseline

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\baseline\run_baseline.py
```

当前 baseline 测试集结果：

```text
MLP     RMSE = 1.4514, R² = 0.9116
GMM     RMSE = 1.8824, R² = 0.8513
AE+MLP  RMSE = 1.8861, R² = 0.8507
```

结果保存到：

```text
results/baseline/
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

## 跨工况实验

`experiments/cross_domain/` 用于后续实现受 DVPF 论文启发的跨工况无监督软测量实验。

计划顺序：

```text
1. source-only MLP
2. CORAL feature alignment
3. MMD feature alignment
```

训练时目标工况只使用 `X`，不使用目标工况的 `y`。目标工况的 `y` 只在最后评价 RMSE、MAE、R² 时使用。

## 结果解释

`results/baseline/metrics/metrics_summary.csv` 用于比较三个 baseline 模型的最终测试集表现。

`*_split_metrics.csv` 用于查看 Train / Valid / Test 指标是否接近，从而判断模型是否存在明显过拟合。

AE+MLP 消融实验的 summary 文件用于选择更合适的超参数，建议主要根据验证集 RMSE 选择，而不是直接看测试集。
