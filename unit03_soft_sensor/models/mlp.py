from torch import nn

from unit03_soft_sensor.models.common import get_activation


class MLPRegressor(nn.Module):
    """MLP baseline 回归器。

    MLP 通过多层线性变换 + 非线性激活函数拟合 X 到 y 的复杂映射。
    默认结构：13 -> 128 -> 64 -> 1。

    这里吸收了参考 MLP notebook 的优点：
    - activation_name 可配置，便于尝试 ReLU / GELU 等激活函数；
    - dropout_rate 可配置，用于缓解过拟合；
    - hidden_sizes 仍保留为统一配置，方便后续做消融实验。
    """

    def __init__(
        self,
        input_dim=13,
        hidden_sizes=(128, 64),
        output_dim=1,
        activation_name="relu",
        dropout_rate=0.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_sizes:
            # Linear: 把上一层特征映射到 hidden_dim 维。
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Activation: 增加非线性表达能力，否则多层线性层仍等价于一层线性层。
            layers.append(get_activation(activation_name))

            # Dropout: 训练时随机丢弃一部分隐藏单元，减少模型对少数特征的依赖。
            # 回归任务中 dropout 不宜盲目过大；本实验默认 0.1，比较保守。
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))

            prev_dim = hidden_dim

        # 输出层维度为 1，对应最后一列软测量目标 y。
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
