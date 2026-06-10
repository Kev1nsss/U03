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

注意：

```text
这个结果只是在单一目标工况 late_stable 上的 RMSE 突破，不能作为整体跨工况成功结论。
最终比较应优先使用所有工况统一评估的 leave-one-condition-out 或目标工况标定自适应结果。
```

## 6. 神经网络工况标定自适应实验

入口脚本：

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\cross_domain\run_few_shot_neural_adaptation.py --modes target_only --n-calibration-list 1000 --n-trials 3 --hidden-sizes 128 64 --dropout 0.05 --finetune-epochs 300 --patience 40 --batch-size 64 --learning-rate 0.001 --weight-decay 0.0001 --summary-prefix few_shot_neural_adaptation_target_scaled_1000
```

实验目的：

```text
不再只挑单个目标工况，而是对 7 个工况逐一做目标工况标定自适应。
每次从目标工况中抽取相同数量的有标签标定样本训练 MLP，再在该工况剩余样本上测试。
这是深度学习少样本自适应，不是树模型，也不是无监督领域适应。
```

结果位置：

```text
results/cross_domain/few_shot_neural_adaptation_target_scaled_1000_summary.csv
results/cross_domain/few_shot_neural_adaptation_target_scaled_1000_by_condition.csv
results/cross_domain/few_shot_neural_adaptation_target_scaled_1000_aggregate.csv
```

当前结果：

```text
target_only MLP, n_calibration = 1000, n_trials = 3
平均 RMSE = 0.8085
平均 MAE  = 0.5697
平均 R²   = 0.6445
最差工况  = long_stable, RMSE = 1.2864
7/7 个工况的平均 RMSE 都低于原始 MLP baseline RMSE = 1.4514
```

当前结论：

```text
在 RMSE 指标上，少量目标工况标定后的神经网络已经全面优于最初同分布 MLP baseline。
但 R² 只有 2/7 个工况超过 0.9116，因此报告中不能写成所有指标全面超过。
更稳妥的表述是：引入目标工况标定信息后，深度学习模型在所有工况上都获得了更低的绝对预测误差。
```

## 7. AECL 真实工况自适应实验

入口脚本：

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\aecl_cross_condition\run_aecl_leave_one.py --modes mlp ae_mlp aecl --n-calibration-list 1000 --n-trials 3 --epochs 120 --patience 30 --batch-size 256 --max-source-per-condition 3000 --max-target-unlabeled 6000 --summary-prefix aecl_leave_one_1000
```

实验目的：

```text
把学长 AECL-MLP 的思想改造成真实工况 leave-one-condition-out 实验。
模型由 Encoder、Decoder、MLP 回归头和工况对比学习损失组成。
目标是让方法不只是普通 MLP 标定，而是利用无标签 X 重构和工况表征约束提升跨工况自适应。
```

三个对照模式：

```text
mlp:    只用目标工况 calibration y 训练 MLP。
ae_mlp: 目标工况 calibration y + 目标工况无标签 X 重构，不加对比学习。
aecl:   源工况 + 目标工况 calibration + AE 重构 + 工况对比学习。
```

结果位置：

```text
results/aecl_cross_condition/
results/aecl_cross_condition/aecl_leave_one_1000_summary.csv
results/aecl_cross_condition/aecl_leave_one_1000_by_condition.csv
results/aecl_cross_condition/aecl_leave_one_1000_aggregate.csv
```

当前 3 次重复正式结果：

```text
AE-MLP 平均 RMSE = 0.8201, 平均 MAE = 0.5846, 平均 R² = 0.6777, 最差工况 long_stable RMSE = 1.3078
AECL   平均 RMSE = 0.8284, 平均 MAE = 0.5891, 平均 R² = 0.6663, 最差工况 long_stable RMSE = 1.2972
MLP    平均 RMSE = 0.8836, 平均 MAE = 0.6294, 平均 R² = 0.6301, 最差工况 long_stable RMSE = 1.3224
```

当前结论：

```text
三种方法 7/7 个工况的平均 RMSE 都低于原始 MLP baseline 1.4514。
AE-MLP 当前平均 RMSE、平均 MAE 和平均 R² 最好；AECL 相比普通 MLP 有明显提升，并且最差工况 RMSE 略优于 AE-MLP。
因此当前最稳妥的写法是：AE 重构带来了主要收益，工况对比学习进一步改善了最差工况，但当前参数下没有全面超过 AE-MLP。
```

## 8. DVPF-inspired GRU-VAE-flow 实验

入口脚本：

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\dvpf_inspired\run_gru_vae_flow_leave_one.py --seq-len 20 --epochs 40 --patience 10 --batch-size 128 --max-source-per-condition 1000 --max-target-unlabeled 2000 --summary-prefix pilot_gru_vae_flow_all_conditions

D:\miniconda3\envs\d2l\python.exe -B experiments\dvpf_inspired\run_calibrated_gru_vae_flow_leave_one.py --include-source --n-calibration-list 1000 --n-trials 3 --seq-len 20 --epochs 100 --patience 25 --batch-size 256 --max-source-per-condition 3000 --max-target-unlabeled 6000 --summary-prefix formal_calibrated_gru_vae_flow_1000_include_source
```

实验目的：

```text
把 DVPF 论文中的时序建模、概率 latent 表征和 flow 更新思想简化成 GRU-VAE-flow。
先测试严格的无目标标签跨工况自适应；如果无标签版本不足，再加入 1000 个目标工况标定标签，形成更实用的半监督标定自适应。
```

模型结构：

```text
X sequence -> GRU Encoder -> mu/logvar -> latent z -> residual latent flow -> MLP Head -> y
                                                   -> Decoder -> X_rec
```

无目标标签 pilot 结果：

```text
GRU-VAE-flow        平均 RMSE = 2.2260, 平均 MAE = 1.7447, 平均 R² = -11.4577
GRU-VAE-flow+CORAL  平均 RMSE = 2.2329, 平均 MAE = 1.7516, 平均 R² = -11.5212
GRU-VAE-flow+MMD    平均 RMSE = 2.2873, 平均 MAE = 1.8089, 平均 R² = -9.7773
```

结论：

```text
严格无目标标签版本没有成功。目标工况 X 的重构和 latent 对齐不能保证 X -> y 关系自动迁移。
这条结果反而说明：当前数据上，要取得稳定低 RMSE，需要少量目标工况 y 做标定。
```

1000 标定标签正式结果：

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
results/dvpf_inspired/pilot_gru_vae_flow_all_conditions_aggregate.csv
results/dvpf_inspired/pilot_gru_vae_flow_coral_all_conditions_aggregate.csv
results/dvpf_inspired/pilot_gru_vae_flow_mmd_all_conditions_aggregate.csv
results/dvpf_inspired/formal_calibrated_gru_vae_flow_1000_include_source_aggregate.csv
results/dvpf_inspired/formal_calibrated_gru_vae_flow_1000_include_source_by_condition.csv
```

当前结论：

```text
DVPF-inspired 的无标签简化版不适合作为最终结果主线。
更适合写成：受 DVPF 启发的时序概率表征 + 目标工况少量标定自适应。
正式 3 次重复已经在 7/7 个工况上低于原始 MLP 的 RMSE；但平均 RMSE 略弱于 AE-MLP 的 0.8201 和 AECL 的 0.8284，因此建议作为有创新解释力的补充模型，而不是当前最强指标模型。
```
