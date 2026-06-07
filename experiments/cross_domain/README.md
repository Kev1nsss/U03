# 跨工况实验

这个目录用于实现受 DVPF 论文启发的跨工况无监督软测量实验。

## 核心思想

把不同运行工况看成不同领域：

```text
source domain：源工况，有 X 和 y，用来监督训练
target domain：目标工况，训练时只使用 X，不使用 y
test：训练结束后，才用目标工况 y 计算 RMSE、MAE、R²
```

这和普通随机划分不同。普通 baseline 主要回答“同分布下模型好不好”，跨工况实验回答“换到新工况以后模型还能不能泛化”。

## 脚本规划

```text
run_source_only.py       # 只用源工况标签训练，直接测试目标工况
run_coral_adaptation.py  # 源域回归损失 + CORAL 特征对齐
run_mmd_adaptation.py    # 源域回归损失 + MMD 特征对齐
```

## 推荐实现顺序

```text
1. 先根据原始曲线图定义工况区间。
2. 实现 source-only MLP，作为跨工况 baseline。
3. 实现 CORAL 或 MMD 特征对齐。
4. 比较目标工况上的 RMSE、MAE、R²。
```

## 结果位置

```text
results/cross_domain/source_only/
results/cross_domain/coral/
results/cross_domain/mmd/
```

## 和 DVPF 的关系

DVPF 的重点是无监督跨域软测量：源域有标签，目标域无标签，模型需要利用目标域 X 适应新工况。

这里先做轻量版：

```text
Source-only baseline
CORAL feature alignment
MMD feature alignment
```

等轻量版跑通后，再考虑时序模型、RNN、变分推断或更接近 DVPF 的结构。
