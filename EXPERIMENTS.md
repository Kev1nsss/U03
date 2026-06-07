# 实验索引

这个文件是 Unit03 软测量项目的实验导航。你可以把它当成“先看这里”的说明书，用来确认每个脚本的作用、运行命令和结果位置。

## 1. Baseline 主实验

入口脚本：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\baseline\run_baseline.py
```

实验目的：

```text
在当前同分布划分方式下，对比 MLP、GMM 和 AE+MLP 三个基础模型。
```

结果位置：

```text
results/baseline/metrics/metrics_summary.csv
results/baseline/metrics/mlp_split_metrics.csv
results/baseline/metrics/gmm_split_metrics.csv
results/baseline/metrics/aemlp_split_metrics.csv
results/baseline/figures/
results/baseline/models/
```

当前结论：

```text
在当前周期式 train/valid/test 划分下，MLP 的测试集表现最好。
```

## 2. AE+MLP 消融实验

入口脚本：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_latent_dim_ablation.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_dropout_ablation.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_latent_dropout_grid.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_finetune_ablation.py
```

实验目的：

```text
分析 AE+MLP 为什么弱于直接 MLP，并尝试通过 latent_dim、dropout 和 Encoder fine-tune 改善效果。
```

结果位置：

```text
results/aemlp_ablations/latent_dim/
results/aemlp_ablations/dropout/
results/aemlp_ablations/latent_dropout_grid/
results/aemlp_ablations/finetune/
```

当前结论：

```text
增大 latent_dim 和微调 Encoder 都能明显改善 AE+MLP，说明原始 AE+MLP 的主要问题是特征压缩和冻结 Encoder 限制了对 y 的适应能力。
不过当前最好的 fine-tuned AE+MLP 仍然没有超过直接 MLP baseline。
```

## 3. 跨工况实验

入口脚本：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_source_only.py
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_coral_adaptation.py
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_mmd_adaptation.py
```

当前状态：

```text
已建立目录框架，训练逻辑尚未实现。
```

实验目的：

```text
结合 DVPF 论文思想，把不同运行工况视为不同域。
训练阶段使用源工况的有标签数据，同时使用目标工况的无标签 X 做领域适应。
目标工况的 y 只在最后评价 RMSE、MAE 和 R² 时使用。
```

计划结果位置：

```text
results/cross_domain/source_only/
results/cross_domain/coral/
results/cross_domain/mmd/
```

推荐实施顺序：

```text
1. 先根据原始曲线图定义工况区间。
2. 实现 source-only MLP，作为跨工况 baseline。
3. 实现 CORAL 或 MMD 特征对齐。
4. 比较目标工况上的 RMSE、MAE 和 R²。
```

## 4. 怎么理解这三类实验

```text
baseline：回答“同分布划分下，哪个基础模型最好？”
AE+MLP 消融：回答“AE+MLP 为什么不好，怎么改会更好？”
跨工况实验：回答“结合 DVPF 思想，模型能不能适应目标工况无标签的情况？”
```
