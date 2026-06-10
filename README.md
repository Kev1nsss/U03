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
  aecl_cross_condition/           # AECL + 真实工况 leave-one-condition-out
  dvpf_inspired/                  # DVPF-inspired GRU-VAE-flow 实验

results/                          # 所有实验结果
  baseline/                       # baseline 结果
  aemlp_ablations/                # AE+MLP 消融结果
  cross_domain/                   # 跨工况实验结果
  aecl_cross_condition/           # AECL 跨工况结果
  dvpf_inspired/                  # DVPF-inspired 结果
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


## GRU 时序跨工况探索

新增脚本：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_gru_source_only.py --target-condition late_stable --seq-len 10 --hidden-dim 64 --dropout 0.05
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_gru_coral_adaptation.py --target-condition late_stable --seq-len 40 --hidden-dim 64 --dropout 0.05 --alignment-weight 0.1
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_gru_mmd_adaptation.py --target-condition late_stable --seq-len 10 --hidden-dim 64 --dropout 0.05 --alignment-weight 0.05
```

当前最优跨工况结果：

```text
GRU+CORAL target=late_stable
seq_len=40, hidden_dim=64, dropout=0.05, alignment_weight=0.1
Target RMSE = 1.0741
Target MAE  = 0.8532
Target R²   = -0.0509
```

这个 RMSE 已经低于同分布 MLP baseline 的 `1.4514`，说明引入时序窗口和 CORAL 对齐后，某些目标工况上可以取得更低绝对误差。
不过 `late_stable` 的 y 波动较小，因此 R² 仍未转正。报告里要写清楚：这是 RMSE 上的突破，不代表跨工况拟合优度已经全面超过同分布 MLP。

结果汇总：

```text
results/cross_domain/gru_comparison_summary.csv
```

这个结果只说明单个目标工况上的 RMSE 可以被压低，不能作为最终跨工况结论。更严谨的主线应使用所有工况统一评价。

## 神经网络工况标定自适应

当前保留的深度学习自适应方向是少量目标工况标定：每次留出 1 个目标工况，在该目标工况中只抽取固定数量标定样本训练神经网络，再在该工况剩余样本上测试。这样不是挑单个工况，而是 7 个工况全部评估。

运行命令：

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\cross_domain\run_few_shot_neural_adaptation.py --modes target_only --n-calibration-list 1000 --n-trials 3 --hidden-sizes 128 64 --dropout 0.05 --finetune-epochs 300 --patience 40 --batch-size 64 --learning-rate 0.001 --weight-decay 0.0001 --summary-prefix few_shot_neural_adaptation_target_scaled_1000
```

聚合结果：

```text
target_only MLP, n_calibration = 1000, n_trials = 3
平均 RMSE = 0.8085
平均 MAE  = 0.5697
平均 R²   = 0.6445
最差工况  = long_stable, RMSE = 1.2864
7/7 个工况的平均 RMSE 都低于原始 MLP baseline RMSE = 1.4514
```

需要注意：这个方法已经在 RMSE 上全面优于最初 MLP，但 R² 不是所有工况都超过 `0.9116`。它应表述为“深度学习少样本目标工况标定自适应”，不是无监督跨域迁移。

## AECL 真实工况自适应

新增目录：

```text
experiments/aecl_cross_condition/
```

这个方向把学长 AECL-MLP 的思想改成真实工况 leave-one-condition-out：`Encoder + Decoder + MLP Head + condition contrastive loss`，同时保留少量目标工况标定样本。

运行命令：

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\aecl_cross_condition\run_aecl_leave_one.py --modes mlp ae_mlp aecl --n-calibration-list 1000 --n-trials 3 --epochs 120 --patience 30 --batch-size 256 --max-source-per-condition 3000 --max-target-unlabeled 6000 --summary-prefix aecl_leave_one_1000
```

当前 3 次重复正式结果：

```text
AE-MLP 平均 RMSE = 0.8201, 平均 MAE = 0.5846, 平均 R² = 0.6777
AECL   平均 RMSE = 0.8284, 平均 MAE = 0.5891, 平均 R² = 0.6663
MLP    平均 RMSE = 0.8836, 平均 MAE = 0.6294, 平均 R² = 0.6301
```

三种方法 7/7 个工况的平均 RMSE 都低于原始 MLP baseline `1.4514`。AECL 相比 MLP 有明显提升，并且最差工况 RMSE 比 AE-MLP 略低；但按平均 RMSE/MAE/R² 看，当前最好的是 AE-MLP。

## DVPF-inspired GRU-VAE-flow 探索

新增目录：

```text
experiments/dvpf_inspired/
```

这条线借鉴论文 `A Deep Probabilistic Flow-Based Framework for Unsupervised Cross-Domain Soft Sensing`：用 GRU 做时序编码，用 VAE latent 和轻量 residual flow 表示隐变量，再用目标工况无标签 `X` 做重构自适应。

无目标标签 leave-one-condition-out pilot：

```text
GRU-VAE-flow        平均 RMSE = 2.2260, 最差工况 early_stable RMSE = 3.5772
GRU-VAE-flow+CORAL  平均 RMSE = 2.2329, 最差工况 early_stable RMSE = 3.5743
GRU-VAE-flow+MMD    平均 RMSE = 2.2873, 最差工况 early_stable RMSE = 3.7563
```

结论：只用目标工况无标签 `X` 做重构/对齐不够，不能超过原始同分布 MLP。

1000 个目标标定标签正式结果：

```text
calibrated GRU-VAE-flow, include-source, n_calibration = 1000, n_trials = 3
平均 RMSE = 0.8350
平均 MAE  = 0.6047
平均 R²   = 0.6445
最差工况  = long_stable, RMSE = 1.3664
7/7 个工况 RMSE 都低于原始 MLP baseline RMSE = 1.4514
```

结果位置：

```text
results/dvpf_inspired/formal_calibrated_gru_vae_flow_1000_include_source_aggregate.csv
results/dvpf_inspired/formal_calibrated_gru_vae_flow_1000_include_source_by_condition.csv
```

这个版本可以作为“DVPF 论文思想 + 目标工况少量标定”的创新尝试。它稳定低于原始 MLP 的 RMSE，但平均 RMSE 略弱于当前 AE-MLP 正式结果 `0.8201`，因此更适合作为创新补充方向，而不是当前最强指标主线。
