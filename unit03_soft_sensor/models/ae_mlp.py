from torch import nn

from unit03_soft_sensor.models.common import get_activation


class AutoEncoder(nn.Module):
    """AutoEncoder 特征提取器。

    Encoder 把 13 维过程变量压缩到 latent_dim 维；
    Decoder 尝试从 latent feature 重构原始 X。
    重构训练不使用 y，所以它是无监督/自监督特征学习。
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
    """

    def __init__(self, latent_dim=6, activation_name="relu", dropout_rate=0.0):
        super().__init__()

        layers = []
        prev_dim = latent_dim

        for hidden_dim in (64, 32):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_name))
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class AEMLPFineTuner(nn.Module):
    """AE 预训练后的端到端微调模型。

    普通 AE+MLP 的流程是：
    1. AE 用 X_train 重构 X，训练 Encoder；
    2. 固定 Encoder，提取 z；
    3. 只训练 z -> y 的 MLP 回归头。

    微调版本的区别是：
    AE 预训练后，Encoder 不再固定，而是和 MLP 回归头一起用 20% 标签继续训练。
    这样 Encoder 会根据 y 的预测误差调整特征，更接近任务导向的半监督学习。
    """

    def __init__(self, encoder, latent_dim=6, activation_name="relu", dropout_rate=0.0):
        super().__init__()
        self.encoder = encoder
        self.regressor = AEMLPRegressor(
            latent_dim=latent_dim,
            activation_name=activation_name,
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.regressor(z)
