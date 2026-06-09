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

单目标工况 `late_disturbance`：

```text
CORAL       Target RMSE = 2.1719, Target MAE = 1.6640, Target R² = -3.6644
Source-only Target RMSE = 2.2593, Target MAE = 1.7464, Target R² = -4.0473
MMD         Target RMSE = 2.3456, Target MAE = 1.7903, Target R² = -4.4403
```

leave-one-condition-out 结果：

```text
startup_ramp       CORAL Target RMSE = 2.1850, Target R² = 0.8400
restart_transition CORAL Target RMSE = 2.2473, Target R² = 0.8249
long_stable        Source-only Target RMSE = 1.6621, Target R² = -0.0782
late_disturbance   CORAL Target RMSE = 2.1719, Target R² = -3.6644
```

结果位置：

```text
results/cross_domain/leave_one_condition_summary.csv
results/cross_domain/summary_figures/leave_one_condition_summary.png
```

当前结论：

```text
跨工况难度不是均匀的。startup_ramp 和 restart_transition 可以较好迁移，R² 达到 0.82 以上；
late_disturbance、early_stable、late_stable 等工况更难，说明这些目标工况和源工况之间存在更明显分布偏移。
CORAL 在 startup_ramp、restart_transition、late_disturbance、late_stable 上相对 source-only 有小幅提升，说明特征对齐有价值，但还不足以全面超过同分布 MLP baseline。
```


## GRU 时序模型

为了更贴近 DVPF 论文中的动态建模思想，当前目录还新增了 GRU 版本：

```text
sequence_common.py             # GRU 时序数据窗口、训练、评价和画图公共逻辑
run_gru_source_only.py          # GRU source-only
run_gru_coral_adaptation.py     # GRU + CORAL
run_gru_mmd_adaptation.py       # GRU + MMD
```

当前最优配置：

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\cross_domain\run_gru_coral_adaptation.py --target-condition late_stable --seq-len 40 --hidden-dim 64 --dropout 0.05 --alignment-weight 0.1
```

结果：

```text
Target RMSE = 1.0741
Target MAE  = 0.8532
Target R²   = -0.0509
```

这个结果在 RMSE 上已经低于同分布 MLP baseline 的 `1.4514`。需要注意的是，`late_stable` 的 y 波动较小，所以 R² 仍略低于 0；写报告时应该把它表述为“RMSE 取得突破”，而不是“所有指标全面超过”。

## GRU-MoE leave-one-condition-out

为了避免只挑单一目标工况，本轮新增 GRU-MoE 和 GRU-MoE+CORAL，并按 7 个工况做 leave-one-condition-out：

```text
GRU-MoE:       每个源工况一个 GRU 专家，gate 根据输入序列加权各专家预测
GRU-MoE+CORAL: 在 GRU-MoE 的加权隐表示上额外做 source/target CORAL 对齐
```

运行命令：

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\cross_domain\run_gru_moe_leave_one_condition_out.py --seq-len 40 --hidden-dim 64 --dropout 0.05 --epochs 60 --patience 12 --batch-size 256 --max-source-per-condition 3000 --max-target-unlabeled 6000
```

结果位置：

```text
results/cross_domain/gru_moe_leave_one_condition_summary.csv
results/cross_domain/gru_moe_leave_one_condition_aggregate.csv
results/cross_domain/gru_moe/
results/cross_domain/gru_moe_coral/
```

聚合结果：

```text
GRU-MoE+CORAL avg RMSE = 2.3317, avg MAE = 1.8697, avg R² = -4.5442
GRU-MoE       avg RMSE = 2.4302, avg MAE = 1.9412, avg R² = -4.0113
```

最差工况：

```text
按 RMSE/MAE 看，两个方法最差工况都是 early_stable。
按 R² 看，两个方法最差工况都是 initial_low；这个工况 y 方差很小，所以即使 RMSE 很低，R² 也会非常负。
```

当前结论：

```text
GRU-MoE+CORAL 在平均 RMSE 和平均 MAE 上略优于 GRU-MoE，但 R² 仍不稳定；
因此 MoE 方向目前不能表述为整体跨工况成功，只能表述为更严格 LOO 评估下的一个新对照模型。
```
