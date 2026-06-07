import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def inverse_y(y_scaler, y_scaled):
    """把标准化尺度的预测值还原到原始 y 尺度后再评价。"""
    return y_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1))


def evaluate_regression(y_true, y_pred):
    """计算回归任务常用指标：MSE / RMSE / MAE / R²。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R²": float(r2_score(y_true, y_pred)),
    }


def print_metrics(model_name, metrics):
    print(f"{model_name} Test Metrics")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


def print_metrics_table(title, split_metrics):
    """按数据集划分打印指标表。

    新手读结果时建议重点看：
    - Train 好、Valid/Test 差：可能过拟合；
    - Train/Valid/Test 都差：可能欠拟合或特征不足；
    - Train/Valid/Test 接近且指标较好：泛化通常更可靠。
    """
    print("\n" + title)
    print("-" * 78)
    print(f"{'Split':>8} | {'MSE':>14} | {'RMSE':>14} | {'MAE':>14} | {'R²':>10}")
    print("-" * 78)

    for split_name, metrics in split_metrics.items():
        print(
            f"{split_name:>8} | "
            f"{metrics['MSE']:14.6f} | "
            f"{metrics['RMSE']:14.6f} | "
            f"{metrics['MAE']:14.6f} | "
            f"{metrics['R²']:10.6f}"
        )

    print("-" * 78)
