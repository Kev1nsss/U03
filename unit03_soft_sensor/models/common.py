from torch import nn


def get_activation(activation_name):
    """根据配置名称创建激活函数。

    为什么需要激活函数：
    - 线性层只能表达线性关系；
    - 多个线性层叠在一起，本质上仍然等价于一个线性层；
    - 加入 ReLU / GELU / tanh 等非线性激活后，神经网络才能拟合复杂非线性关系。

    本实验默认使用 ReLU，因为它简单、稳定、训练速度快。
    """
    activation_name = activation_name.lower()

    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if activation_name == "gelu":
        return nn.GELU()
    if activation_name == "tanh":
        return nn.Tanh()

    raise ValueError(
        f"Unsupported activation_name={activation_name}. "
        "Choose from: relu, leaky_relu, gelu, tanh."
    )

