# AECL Cross-Condition Experiments

这个目录是新的主方法方向：把学长 AECL-MLP 的思想改造成真实工况 leave-one-condition-out 评价。

## 核心设定

每次选择 1 个真实工况作为目标工况：

```text
target calibration: 从目标工况抽取少量有标签 y，用于标定/微调
target unlabeled:   目标工况剩余样本的 X，可用于重构和对比学习，不使用 y
target eval:        目标工况剩余样本的 y 只在最后评价 RMSE、MAE、R2
source conditions:  其他 6 个工况，可为 AECL 提供源域样本和工况对比信息
```

## 三个模式

```text
mlp:
  X -> MLP -> y
  只用目标工况 calibration 样本监督训练。

ae_mlp:
  X -> Encoder -> Z -> MLP Head -> y
  X -> Encoder -> Z -> Decoder -> X_rec
  使用目标工况 calibration y 做监督，使用目标工况无标签 X 做重构。
  不使用工况对比学习。

aecl:
  X -> Encoder -> Z -> MLP Head -> y
  X -> Encoder -> Z -> Decoder -> X_rec
  使用源工况 + 目标工况 calibration y 做监督。
  使用源工况和目标工况的工况编号做 supervised contrastive learning。
```

AECL 的总损失：

```text
Loss = lambda_rec * reconstruction_loss
     + lambda_sup * supervised_regression_loss
     + lambda_con * condition_contrastive_loss
```

## 快速 smoke test

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\aecl_cross_condition\run_aecl_leave_one.py --targets initial_low --modes mlp ae_mlp aecl --n-calibration-list 100 --n-trials 1 --epochs 2 --patience 2 --batch-size 128 --max-source-per-condition 200 --max-target-unlabeled 200 --summary-prefix smoke_aecl
```

## 正式 7 工况实验

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\aecl_cross_condition\run_aecl_leave_one.py --modes mlp ae_mlp aecl --n-calibration-list 1000 --n-trials 3 --epochs 120 --patience 30 --batch-size 256 --max-source-per-condition 3000 --max-target-unlabeled 6000 --x-scaler-mode target_unlabeled --lambda-rec 0.2 --lambda-con 0.01 --target-supervision-weight 10.0 --source-supervision-weight 0.1 --dropout 0.05 --summary-prefix aecl_leave_one_1000
```

当前 7 工况 3 次重复正式结果：

```text
AE-MLP 平均 RMSE = 0.8201, 平均 MAE = 0.5846, 平均 R2 = 0.6777, 最差工况 long_stable RMSE = 1.3078
AECL   平均 RMSE = 0.8284, 平均 MAE = 0.5891, 平均 R2 = 0.6663, 最差工况 long_stable RMSE = 1.2972
MLP    平均 RMSE = 0.8836, 平均 MAE = 0.6294, 平均 R2 = 0.6301, 最差工况 long_stable RMSE = 1.3224
```

三种方法 7/7 个工况的平均 RMSE 都低于原始 MLP baseline `1.4514`。当前 AE-MLP 的平均指标最好，AECL 相比 MLP 有提升，并且最差工况 RMSE 略优于 AE-MLP。

## 结果位置

```text
results/aecl_cross_condition/<mode>/<target_condition>/
results/aecl_cross_condition/<summary_prefix>_summary.csv
results/aecl_cross_condition/<summary_prefix>_by_condition.csv
results/aecl_cross_condition/<summary_prefix>_aggregate.csv
```

## 写报告时的边界

这个方法不是纯无监督跨域，因为目标工况使用了少量 y 标定样本。更准确的表述是：

```text
半监督/少样本目标工况自适应软测量模型
```

它的创新点在于把无标签 X 重构、工况对比学习和真实工况 leave-one-condition-out 评价结合起来。

