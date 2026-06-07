import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def to_float_tensor(array):
    return torch.as_tensor(array, dtype=torch.float32)


def make_labeled_mask(n_samples, labeled_ratio, seed):
    """少标签设置。

    labeled_ratio=0.2 表示训练集中只有 20% 样本的 y 参与监督损失。
    这里可称为 low-label / semi-supervised setting，但本质仍是少标签监督训练，
    没有额外的一致性正则或伪标签机制。
    """
    n_labeled = max(1, int(round(n_samples * labeled_ratio)))
    rng = np.random.default_rng(seed)
    labeled_indices = rng.choice(n_samples, size=n_labeled, replace=False)

    mask = np.zeros(n_samples, dtype=bool)
    mask[labeled_indices] = True
    return mask


def train_supervised_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    labeled_mask,
    epochs,
    batch_size,
    learning_rate,
    patience,
    device,
    seed,
    weight_decay=0.0,
    model_path=None,
):
    """训练回归模型并使用验证集 early stopping。

    Early Stopping 的理论动机：
    当验证集 loss 多轮不下降时，模型继续训练可能只是在拟合训练集噪声。
    因此保存验证集 loss 最低的参数，并用它做最终测试集评价。
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    # Adam 是常用的自适应学习率优化器。
    # weight_decay 相当于 L2 正则化，会惩罚过大的参数，通常有助于降低过拟合风险。
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 少标签训练的核心：训练时只拿 labeled_mask=True 的样本计算监督损失。
    # 注意：未标注训练样本在这个 MLP baseline 中暂时不参与 loss，
    # 因此它不是复杂半监督算法，而是 low-label supervised baseline。
    X_labeled = X_train[labeled_mask]
    y_labeled = y_train[labeled_mask]

    dataset = TensorDataset(to_float_tensor(X_labeled), to_float_tensor(y_labeled))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    X_valid_tensor = to_float_tensor(X_valid).to(device)
    y_valid_tensor = to_float_tensor(y_valid).to(device)

    history = {"train_loss": [], "valid_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            # PyTorch 默认会累加梯度，所以每个 batch 反向传播前都要清零。
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)

            # backward 根据 loss 计算每个参数的梯度；step 根据梯度更新参数。
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        # 验证集只用于模型选择，不参与参数更新。
        model.eval()
        with torch.no_grad():
            valid_pred = model(X_valid_tensor)
            valid_loss = criterion(valid_pred, y_valid_tensor).item()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        # 记录验证集 loss 最低的模型参数。
        # 最终测试集评价必须使用这个 best_state，而不是最后一个 epoch 的参数。
        if valid_loss < best_valid_loss - 1e-8:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    if model_path is not None:
        torch.save(best_state, model_path)

    return model, history, best_epoch, best_valid_loss


def train_autoencoder(
    autoencoder,
    X_train,
    X_valid,
    epochs,
    batch_size,
    learning_rate,
    patience,
    device,
    seed,
    model_path=None,
):
    """训练 AutoEncoder。

    AE 不使用 y，只学习 X 的压缩表示 z。
    如果 z 能保留主要过程变量结构，后续回归器可以在更低维空间中学习 y。
    """
    autoencoder = autoencoder.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

    dataset = TensorDataset(to_float_tensor(X_train))
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    X_valid_tensor = to_float_tensor(X_valid).to(device)

    history = {"train_loss": [], "valid_loss": []}
    best_state = copy.deepcopy(autoencoder.state_dict())
    best_valid_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, epochs + 1):
        autoencoder.train()
        batch_losses = []

        for (xb,) in loader:
            xb = xb.to(device)

            optimizer.zero_grad()
            x_hat = autoencoder(xb)
            loss = criterion(x_hat, xb)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))

        autoencoder.eval()
        with torch.no_grad():
            valid_hat = autoencoder(X_valid_tensor)
            valid_loss = criterion(valid_hat, X_valid_tensor).item()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid_loss - 1e-8:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(autoencoder.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"AE Epoch {epoch:03d} | train_loss={train_loss:.6f} | valid_loss={valid_loss:.6f}")

        if wait >= patience:
            print(f"AE early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    autoencoder.load_state_dict(best_state)
    if model_path is not None:
        torch.save(best_state, model_path)

    return autoencoder, history, best_epoch, best_valid_loss


def predict_torch_model(model, X, device):
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(to_float_tensor(X).to(device)).cpu().numpy()
    return pred


def extract_encoder_features(autoencoder, X, device):
    autoencoder = autoencoder.to(device)
    autoencoder.eval()
    with torch.no_grad():
        z = autoencoder.encoder(to_float_tensor(X).to(device)).cpu().numpy()
    return z
