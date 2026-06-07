# AE+MLP 消融实验

这个目录只放和 AE+MLP 有关的消融实验。它不负责主 baseline，也不负责后续跨工况实验。

## 为什么单独放这里

主实验里 AE+MLP 的效果没有超过 MLP，所以这里专门分析原因：

```text
1. latent_dim 是否压缩太强
2. dropout 是否设置合适
3. 冻结 Encoder 是否限制了回归效果
4. AE 预训练后微调 Encoder 是否能改善预测
```

## 脚本说明

```text
common.py                         # AE+MLP 消融共用函数
run_latent_dim_ablation.py        # 改 latent_dim，观察 AE 压缩维度影响
run_dropout_ablation.py           # 改 dropout，观察正则化强度影响
run_latent_dropout_grid.py        # 同时搜索 latent_dim 和 dropout
run_finetune_ablation.py          # AE 预训练后微调 Encoder + 回归头
```

## 运行命令

```powershell
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_latent_dim_ablation.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_dropout_ablation.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_latent_dropout_grid.py
D:\miniconda3\envs\d2l\python.exe experiments\aemlp_ablations\run_finetune_ablation.py
```

## 结果位置

```text
results/aemlp_ablations/latent_dim/
results/aemlp_ablations/dropout/
results/aemlp_ablations/latent_dropout_grid/
results/aemlp_ablations/finetune/
```

## 当前结论

```text
增大 latent_dim 和 fine-tune Encoder 都能明显改善 AE+MLP。
但是在当前同分布划分下，AE+MLP 仍然没有超过直接使用原始特征的 MLP。
```
