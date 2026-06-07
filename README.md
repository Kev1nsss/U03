# Unit03 Soft Sensor Regression

本项目用于 Unit03 化工软测量回归实验。数据前 13 列作为输入特征 `X`，最后 1 列作为预测目标 `y`。

## 目录说明

```text
run_unit03_experiment.py          # baseline 主入口：MLP / GMM / AE+MLP

unit03_soft_sensor/
  config.py                       # 统一配置
  data.py                         # 数据读取、抽样、划分、标准化
  train.py                        # 训练、Early Stopping、少标签 mask
  evaluation.py                   # MSE / RMSE / MAE / R²
  plotting.py                     # loss、预测、残差图
  models/
    mlp.py                        # MLP baseline
    gmm.py                        # GMM-based regression
    ae_mlp.py                     # AutoEncoder + MLP

ablations/
  run_latent_dim_ablation.py      # AE latent_dim 消融
  run_dropout_ablation.py         # AE+MLP 回归头 dropout 消融

results/
  metrics/                        # baseline 指标
  figures/                        # baseline 图像
  models/                         # baseline 模型权重
  ablations/                      # 消融实验结果
```

## 运行 baseline

```powershell
D:\miniconda3\envs\d2l\python.exe run_unit03_experiment.py
```

当前 baseline 测试集结果：

```text
MLP     RMSE = 1.4514, R² = 0.9116
GMM     RMSE = 1.8824, R² = 0.8513
AE+MLP  RMSE = 1.8861, R² = 0.8507
```

## 运行消融实验

### latent_dim 消融

```powershell
D:\miniconda3\envs\d2l\python.exe ablations\run_latent_dim_ablation.py
```

结果保存到：

```text
results/ablations/latent_dim/
```

### dropout 消融

```powershell
D:\miniconda3\envs\d2l\python.exe ablations\run_dropout_ablation.py
```

结果保存到：

```text
results/ablations/dropout/
```

## 结果解释

`metrics_summary.csv` 用于比较三个 baseline 模型的最终测试集表现。

`*_split_metrics.csv` 用于查看 Train / Valid / Test 指标是否接近，从而判断模型是否存在明显过拟合。

消融实验的 summary 文件用于选择更合适的 AE+MLP 超参数，建议主要根据验证集 RMSE 选择，而不是直接看测试集。

### latent_dim + dropout 二维组合消融

```powershell
D:\miniconda3\envs\d2l\python.exe ablations\run_latent_dropout_grid.py
```

结果保存到：

```text
results/ablations/latent_dropout_grid/
```

### AE pretrain + Encoder fine-tune 消融

```powershell
D:\miniconda3\envs\d2l\python.exe ablations\run_finetune_ablation.py
```

结果保存到：

```text
results/ablations/finetune/
```
