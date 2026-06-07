from torch import nn

from unit03_soft_sensor.models.common import get_activation


class AutoEncoder(nn.Module):
    """AutoEncoder 特征提取器。

    Encoder 把 13 维过程变量压缩到 latent_dim 维；
    Decoder 尝试从 latent feature 重构原始 X。
    重构训练不使用 y，所以它是无监督特征学习。
    """

    def __init__(self, input_dim=13, latent_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 10),
            nn.ReLU(),
            nn.Linear(10, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class AEMLPRegressor(nn.Module):
    """基于 AE latent feature 的 MLP 回归器。

    输入不再是原始 13 维 X，而是 Encoder 输出的 z。
    默认结构：latent_dim -> 64 -> 32 -> 1。

    和 MLP baseline 保持一致：
    - activation_name 控制隐藏层非线性激活函数；
    - dropout_rate 控制 Dropout 正则化强度；
    - weight_decay 在训练函数中设置，不在模型结构中设置。

    这样做的好处是：MLP baseline 和 AE+MLP 的回归头训练设置更公平。
    差异主要来自 AE 特征提取，而不是回归头配置不一致。
    """

    def __init__(self, latent_dim=6, activation_name="relu", dropout_rate=0.0):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in (64, 32):
            # Linear: 将 AE 提取的 latent feature 映射到隐藏空间。
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation: 为回归头增加非线性表达能力。
            layers.append(get_activation(activation_name))

            # Dropout: 随机屏蔽部分隐藏单元，降低过拟合风险。
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = hidden_dim

        # 输出 1 个连续值，对应原始数据最后一列 y。
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
