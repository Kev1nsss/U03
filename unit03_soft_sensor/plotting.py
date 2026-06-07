import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_true_pred(y_true, y_pred, title, save_path):
    """真实值-预测值散点图；y=x 越接近，说明预测越准确。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.45, edgecolors="none")
    plt.plot([min_v, max_v], [min_v, max_v], color="crimson", linewidth=1.5, label="y = x")
    plt.xlabel("True y")
    plt.ylabel("Predicted y")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_residuals(y_true, y_pred, title, save_path):
    """残差分布图；残差越集中在 0 附近，模型系统偏差越小。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residual = y_true - y_pred

    plt.figure(figsize=(7, 5))
    plt.hist(residual, bins=50, color="steelblue", alpha=0.8)
    plt.axvline(0.0, color="crimson", linewidth=1.5)
    plt.xlabel("Residual = True - Predicted")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_loss_curve(history, title, save_path, best_epoch=None, y_label="MSE Loss"):
    """训练/验证 loss 曲线；Best Epoch 对应验证集 loss 最低点。"""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    plt.plot(epochs, history["valid_loss"], label="Valid Loss", linewidth=1.5)
    if best_epoch is not None:
        plt.axvline(
            best_epoch,
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=f"Best Epoch = {best_epoch}",
        )
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

