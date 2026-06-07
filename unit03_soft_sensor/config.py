from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ExperimentConfig:
    """实验统一配置。

    所有模型共用这里的抽样、划分、标准化和训练参数，保证对比公平。
    """

    csv_path: str = os.path.join("data", "Unit03_1_5_select.csv")
    fallback_csv_path: str = "Unit03_1_5_select.csv"

    results_dir: str = "results"
    metrics_dir: str = os.path.join("results", "metrics")
    models_dir: str = os.path.join("results", "models")
    mlp_figures_dir: str = os.path.join("results", "figures", "mlp")
    gmm_figures_dir: str = os.path.join("results", "figures", "gmm")
    aemlp_figures_dir: str = os.path.join("results", "figures", "ae_mlp")

    n_sample: int = 12000
    random_seed: int = 42

    # 4:3:3 表示每 10 个样本中，4 个进训练集、3 个进验证集、3 个进测试集。
    # 三个模型必须共用同一种划分方式，结果才有可比性。
    split_pattern: tuple = (4, 3, 3)

    # 下面是监督回归器的通用训练参数。
    # epochs 是最多训练轮数；patience 控制 Early Stopping 的等待轮数。
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 20

    # 少标签比例。0.2 表示训练集中只有 20% 的 y 标签参与监督损失。
    labeled_ratio: float = 0.2

    # MLP baseline 的结构和正则化参数。
    # hidden_sizes=(128, 64) 对应 13 -> 128 -> 64 -> 1。
    # dropout 和 weight_decay 都是为了降低过拟合风险，但过大可能导致欠拟合。
    mlp_hidden_sizes: tuple = (128, 64)
    mlp_activation_name: str = "relu"
    mlp_dropout_rate: float = 0.1
    supervised_weight_decay: float = 1e-5

    n_components_list: tuple = (2, 3, 5, 8, 10)

    latent_dim: int = 6

    # AE+MLP 中 AE 负责无监督重构，回归头负责 z -> y。
    # 这里让回归头和 MLP baseline 使用相同的激活函数/Dropout 设定，
    # 这样二者的主要差异就集中在“是否使用 AE 特征提取”。
    aemlp_activation_name: str = "relu"
    aemlp_dropout_rate: float = 0.1

    ae_epochs: int = 80
    ae_batch_size: int = 64
    ae_learning_rate: float = 0.001
    ae_patience: int = 20

    def ensure_dirs(self):
        """创建结果目录，避免保存图像或模型时路径不存在。"""
        for path in [
            self.results_dir,
            self.metrics_dir,
            self.models_dir,
            self.mlp_figures_dir,
            self.gmm_figures_dir,
            self.aemlp_figures_dir,
        ]:
            os.makedirs(path, exist_ok=True)
