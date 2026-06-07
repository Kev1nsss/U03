# Experiments

这个目录只放实验入口，不放核心模型实现。

```text
baseline/           # 主 baseline：MLP / GMM / AE+MLP
aemlp_ablations/   # AE+MLP 专用消融实验
cross_domain/       # DVPF 启发的跨工况实验
```

核心源码仍然放在 `unit03_soft_sensor/`，实验脚本只负责调用这些可复用函数并保存结果。
