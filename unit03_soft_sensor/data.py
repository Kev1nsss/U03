from dataclasses import dataclass
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    """保存一次实验所需的全部数据切分结果。"""

    df_shape: tuple
    sampled_indices: np.ndarray

    X_train_raw: np.ndarray
    X_valid_raw: np.ndarray
    X_test_raw: np.ndarray
    y_train_raw: np.ndarray
    y_valid_raw: np.ndarray
    y_test_raw: np.ndarray

    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray

    X_scaler: StandardScaler
    y_scaler: StandardScaler


def set_random_seed(seed):
    """固定随机种子，降低实验复现时的随机波动。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_csv_path(config):
    if os.path.exists(config.csv_path):
        return config.csv_path
    if os.path.exists(config.fallback_csv_path):
        return config.fallback_csv_path
    raise FileNotFoundError(f"CSV not found: {config.csv_path} or {config.fallback_csv_path}")


def load_and_sample_data(config):
    """读取 CSV 并抽样。

    前 13 列始终作为过程变量 X，最后 1 列始终作为软测量目标 y。
    抽样索引排序后保留原数据中的时间顺序，和参考 notebook 的思路一致。
    """
    csv_path = resolve_csv_path(config)
    df = pd.read_csv(csv_path)
    df = df.dropna().reset_index(drop=True)

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)

    if config.n_sample > len(df):
        raise ValueError(f"n_sample={config.n_sample} is larger than available rows={len(df)}")

    rng = np.random.default_rng(config.random_seed)
    sampled_indices = np.sort(rng.choice(len(df), size=config.n_sample, replace=False))

    return df.shape, sampled_indices, X[sampled_indices], y[sampled_indices]


def split_by_pattern(X, y, pattern):
    """按 4:3:3 周期划分 train/valid/test。

    这种划分不是完全随机切分，而是在抽样后的时序上周期分配，
    可以让三类集合在不同时间段上都有覆盖。
    """
    train_idx, valid_idx, test_idx = [], [], []

    for start in range(0, len(X), sum(pattern)):
        end = min(start + sum(pattern), len(X))
        current = list(range(start, end))

        train_idx.extend(current[: pattern[0]])
        valid_idx.extend(current[pattern[0] : pattern[0] + pattern[1]])
        test_idx.extend(current[pattern[0] + pattern[1] :])

    return (
        X[train_idx],
        X[valid_idx],
        X[test_idx],
        y[train_idx],
        y[valid_idx],
        y[test_idx],
    )


def standardize_splits(X_train_raw, X_valid_raw, X_test_raw, y_train_raw, y_valid_raw, y_test_raw):
    """标准化输入和输出。

    关键防泄漏原则：
    - X_scaler 只能在 X_train 上 fit
    - y_scaler 只能在 y_train 上 fit
    - valid/test 只能 transform，不能参与任何 fit
    """
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train_raw).astype(np.float32)
    X_valid = X_scaler.transform(X_valid_raw).astype(np.float32)
    X_test = X_scaler.transform(X_test_raw).astype(np.float32)

    y_train = y_scaler.fit_transform(y_train_raw).astype(np.float32)
    y_valid = y_scaler.transform(y_valid_raw).astype(np.float32)
    y_test = y_scaler.transform(y_test_raw).astype(np.float32)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, X_scaler, y_scaler


def prepare_data(config):
    df_shape, sampled_indices, X_sampled_raw, y_sampled_raw = load_and_sample_data(config)

    (
        X_train_raw,
        X_valid_raw,
        X_test_raw,
        y_train_raw,
        y_valid_raw,
        y_test_raw,
    ) = split_by_pattern(X_sampled_raw, y_sampled_raw, config.split_pattern)

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
        X_scaler,
        y_scaler,
    ) = standardize_splits(
        X_train_raw,
        X_valid_raw,
        X_test_raw,
        y_train_raw,
        y_valid_raw,
        y_test_raw,
    )

    return PreparedData(
        df_shape=df_shape,
        sampled_indices=sampled_indices,
        X_train_raw=X_train_raw,
        X_valid_raw=X_valid_raw,
        X_test_raw=X_test_raw,
        y_train_raw=y_train_raw,
        y_valid_raw=y_valid_raw,
        y_test_raw=y_test_raw,
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
    )

