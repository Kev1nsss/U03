run_unit03_experiment.py          # 主入口，负责串联完整实验
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
results/                          # 结果图、数据汇总、PyTorch参数


MLP     RMSE = 1.4514, R² = 0.9116
GMM     RMSE = 1.8824, R² = 0.8513
AE+MLP  RMSE = 1.8861, R² = 0.8507

MLP 在训练集上表现最好
并且 split metrics => 三者均没有明显过拟合

注：我 py 版本用的是 3.9 可能效果不好