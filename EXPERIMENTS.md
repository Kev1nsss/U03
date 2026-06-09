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
增大 latent_dim 和微调 Encoder 都能明显改善 AE+MLP。
不过当前最好的 fine-tuned AE+MLP 仍然没有超过直接 MLP baseline。
```

## 3. 跨工况实验

先画清晰工况图：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\plot_conditions.py
```

训练入口：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_source_only.py
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_coral_adaptation.py --alignment-weight 1.0
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_mmd_adaptation.py --alignment-weight 0.05
```

实验目的：

```text
结合 DVPF 论文思想，把不同运行工况视为不同域。
训练阶段使用源工况的有标签数据，同时使用目标工况的无标签 X 做领域适应。
目标工况的 y 只在最后评价 RMSE、MAE 和 R² 时使用。
```

结果位置：

```text
results/cross_domain/condition_overview/
results/cross_domain/comparison_summary.csv
results/cross_domain/source_only/late_disturbance/
results/cross_domain/coral/late_disturbance/
results/cross_domain/mmd/late_disturbance/
```

当前结果：

```text
CORAL       Target RMSE = 2.1719, Target MAE = 1.6640, Target R² = -3.6644
Source-only Target RMSE = 2.2593, Target MAE = 1.7464, Target R² = -4.0473
MMD         Target RMSE = 2.3456, Target MAE = 1.7903, Target R² = -4.4403
```

当前结论：

```text
跨工况测试明显比同分布测试困难。CORAL 相对 source-only 有一定提升，MMD 在当前设置下没有提升。
这说明简单特征对齐能缓解一部分域差异，但还不足以完全解决目标工况泛化问题。
```

## 4. 怎么理解这三类实验

```text
baseline：回答“同分布划分下，哪个基础模型最好？”
AE+MLP 消融：回答“AE+MLP 为什么不好，怎么改会更好？”
跨工况实验：回答“结合 DVPF 思想，模型能不能适应目标工况无标签的情况？”
```


## 5. GRU 时序跨工况实验

新增入口脚本：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_gru_source_only.py
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_gru_coral_adaptation.py
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_gru_mmd_adaptation.py
```

实验目的：

```text
进一步贴近 DVPF 论文中的时序建模思想，把输入从单时刻 X(t) 改为一段窗口 [X(t-L+1), ..., X(t)]，再用 GRU 提取动态特征。
训练时仍然只使用源工况 y，目标工况只提供无标签 X。
```

当前最优结果：

```text
目标工况：late_stable
方法：GRU + CORAL
seq_len = 40
hidden_dim = 64
dropout = 0.05
alignment_weight = 0.1
Target RMSE = 1.0741
Target MAE  = 0.8532
Target R²   = -0.0509
```

与同分布 MLP baseline 对比：

```text
同分布 MLP Test RMSE = 1.4514
GRU+CORAL late_stable Target RMSE = 1.0741
```

解释：

```text
GRU+CORAL 在 late_stable 目标工况上实现了 RMSE 层面的明显突破，说明时序动态特征和目标域无标签 X 的特征对齐是有价值的。
但由于 late_stable 本身 y 方差较小，R² 仍略低于 0，所以不能简单说模型整体拟合能力全面优于同分布 MLP。
更严谨的结论是：在某些目标工况上，时序跨域方法可以显著降低绝对误差，但跨工况泛化仍需结合 R²、残差图和预测曲线共同分析。
```

结果位置：

```text
results/cross_domain/gru_comparison_summary.csv
results/cross_domain/gru_coral/late_stable/
```
