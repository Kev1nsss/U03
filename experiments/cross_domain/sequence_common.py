import argparse
import copy
import os
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cross_domain.common import (
    CONDITION_INTERVALS,
    DEFAULT_TARGET_CONDITION,
    build_condition_table,
    condition_names,
    coral_loss,
    mmd_loss,
    plot_prediction_curve,
    plot_tsne_features,
    save_condition_overview,
    update_global_comparison,
)
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import resolve_csv_path, set_random_seed
from unit03_soft_sensor.evaluation import evaluate_regression, inverse_y
from unit03_soft_sensor.plotting import plot_loss_curve, plot_residuals, plot_true_pred
from unit03_soft_sensor.train import to_float_tensor


@dataclass
class SequenceCrossDomainData:
    target_condition: str
    source_conditions: list
    condition_table: pd.DataFrame
    X_source_train: np.ndarray
    y_source_train: np.ndarray
    X_source_valid: np.ndarray
    y_source_valid: np.ndarray
    y_source_valid_raw: np.ndarray
    X_target_unlabeled: np.ndarray
    X_target_eval: np.ndarray
    y_target_eval_raw: np.ndarray
    target_eval_indices: np.ndarray
    X_scaler: StandardScaler
    y_scaler: StandardScaler
    source_train_condition_ids: np.ndarray = None
    source_valid_condition_ids: np.ndarray = None


class GRUDomainRegressor(nn.Module):
    """GRU regressor that exposes the final hidden state as domain features."""

    def __init__(self, input_dim=13, hidden_dim=64, num_layers=1, dropout_rate=0.0):
        super().__init__()
        gru_dropout = dropout_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def extract_features(self, x):
        _, hidden = self.gru(x)
        features = hidden[-1]
        return self.dropout(features)

    def forward(self, x, return_features=False):
        features = self.extract_features(x)
        pred = self.regressor(features)
        if return_features:
            return pred, features
        return pred


class _GRUExpert(nn.Module):
    """One GRU expert used by the mixture model."""

    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.0):
        super().__init__()
        gru_dropout = dropout_rate if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def extract_features(self, x):
        _, hidden = self.gru(x)
        return self.dropout(hidden[-1])

    def forward(self, x):
        features = self.extract_features(x)
        return self.regressor(features), features


class GRUMoERegressor(nn.Module):
    """GRU mixture-of-experts with one source-condition expert per condition."""

    def __init__(
        self,
        input_dim=13,
        hidden_dim=64,
        num_experts=6,
        num_layers=1,
        dropout_rate=0.05,
        gate_temperature=1.0,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = num_experts
        self.gate_temperature = max(float(gate_temperature), 1e-6)
        self.experts = nn.ModuleList(
            [
                _GRUExpert(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_experts)
            ]
        )
        gru_dropout = dropout_rate if num_layers > 1 else 0.0
        self.gate_gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.gate_dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.gate_head = nn.Linear(hidden_dim, num_experts)

    def extract_gate_features(self, x):
        _, hidden = self.gate_gru(x)
        return self.gate_dropout(hidden[-1])

    def forward(self, x, return_features=False, return_aux=False):
        expert_preds = []
        expert_features = []
        for expert in self.experts:
            pred, features = expert(x)
            expert_preds.append(pred)
            expert_features.append(features)

        expert_preds = torch.cat(expert_preds, dim=1)
        expert_features = torch.stack(expert_features, dim=1)
        gate_features = self.extract_gate_features(x)
        gate_logits = self.gate_head(gate_features)
        gate_weights = torch.softmax(gate_logits / self.gate_temperature, dim=1)
        pred = torch.sum(expert_preds * gate_weights, dim=1, keepdim=True)
        mixture_features = torch.sum(expert_features * gate_weights.unsqueeze(-1), dim=1)

        if return_aux:
            return pred, mixture_features, {
                "expert_preds": expert_preds,
                "expert_features": expert_features,
                "gate_logits": gate_logits,
                "gate_weights": gate_weights,
            }
        if return_features:
            return pred, mixture_features
        return pred

    def extract_features(self, x):
        _, features = self.forward(x, return_features=True)
        return features


def build_sequence_parser(method_name, default_alignment_weight=0.0):
    parser = argparse.ArgumentParser(description=f"Run GRU {method_name} cross-domain experiment.")
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--source-conditions", nargs="+", default=None)
    parser.add_argument("--x-scaler-mode", choices=["source", "source_target"], default="source")
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--alignment-weight", type=float, default=default_alignment_weight)
    parser.add_argument("--max-source-per-condition", type=int, default=4000)
    parser.add_argument("--max-target-unlabeled", type=int, default=8000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--tsne-max-per-domain", type=int, default=600)
    return parser


def build_gru_moe_parser(method_name, default_alignment_weight=0.0):
    parser = build_sequence_parser(method_name, default_alignment_weight=default_alignment_weight)
    parser.set_defaults(seq_len=40)
    parser.add_argument("--expert-loss-weight", type=float, default=0.5)
    parser.add_argument("--gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--make-plots", action="store_true")
    return parser


def _float_tag(value):
    return str(value).replace("-", "m").replace(".", "p")


def make_sequence_output_dirs(project_root, method_name, target_condition, seq_len, hidden_dim, dropout, alignment_weight, x_scaler_mode):
    dropout_tag = _float_tag(dropout)
    align_tag = _float_tag(alignment_weight)
    setting_tag = f"seq_len_{seq_len}_hidden_{hidden_dim}_dropout_{dropout_tag}_align_{align_tag}_scaler_{x_scaler_mode}"
    output_root = project_root / "results" / "cross_domain" / method_name / target_condition / setting_tag
    dirs = {
        "root": output_root,
        "metrics": output_root / "metrics",
        "figures": output_root / "figures",
        "models": output_root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def make_gru_moe_output_dirs(
    project_root,
    method_name,
    target_condition,
    seq_len,
    hidden_dim,
    dropout,
    alignment_weight,
    x_scaler_mode,
    expert_loss_weight,
    gate_loss_weight,
    gate_temperature,
):
    setting_tag = (
        f"seq_len_{seq_len}_hidden_{hidden_dim}_dropout_{_float_tag(dropout)}"
        f"_align_{_float_tag(alignment_weight)}_scaler_{x_scaler_mode}"
        f"_expert_{_float_tag(expert_loss_weight)}_gate_{_float_tag(gate_loss_weight)}"
        f"_temp_{_float_tag(gate_temperature)}"
    )
    output_root = project_root / "results" / "cross_domain" / method_name / target_condition / setting_tag
    dirs = {
        "root": output_root,
        "metrics": output_root / "metrics",
        "figures": output_root / "figures",
        "models": output_root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _clip_interval(interval, n_rows):
    start = max(0, min(int(interval["start"]), n_rows))
    end = max(start, min(int(interval["end"]), n_rows))
    return start, end


def _make_window_end_indices(start, end, seq_len):
    first_end = start + seq_len - 1
    last_end = end - 1
    if first_end > last_end:
        return np.array([], dtype=np.int64)
    return np.arange(first_end, last_end + 1, dtype=np.int64)


def _sample_sorted(indices, max_count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if max_count is not None and max_count > 0 and len(indices) > max_count:
        indices = rng.choice(indices, size=max_count, replace=False)
    return np.sort(indices)


def _split_train_valid(indices, valid_ratio):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    n_valid = max(1, int(round(len(indices) * valid_ratio)))
    n_train = max(1, len(indices) - n_valid)
    train_indices = indices[:n_train]
    valid_indices = indices[n_train:]
    if len(valid_indices) == 0:
        valid_indices = train_indices[-1:]
        train_indices = train_indices[:-1]
    return train_indices, valid_indices


def _concat_sorted_with_labels(index_parts, label_parts):
    indices = np.concatenate(index_parts)
    labels = np.concatenate(label_parts)
    order = np.argsort(indices)
    return indices[order], labels[order]


def _build_windows(X_all, y_all, end_indices, seq_len):
    n_features = X_all.shape[1]
    windows = np.empty((len(end_indices), seq_len, n_features), dtype=np.float32)
    for i, end_idx in enumerate(end_indices):
        start_idx = end_idx - seq_len + 1
        windows[i] = X_all[start_idx : end_idx + 1]
    y = y_all[end_indices]
    return windows, y


def _transform_windows(scaler, windows):
    original_shape = windows.shape
    flat = windows.reshape(-1, original_shape[-1])
    transformed = scaler.transform(flat).astype(np.float32)
    return transformed.reshape(original_shape)


def load_sequence_cross_domain_data(
    config,
    target_condition,
    seq_len=10,
    max_source_per_condition=4000,
    max_target_unlabeled=8000,
    source_valid_ratio=0.2,
    source_condition_names=None,
    x_scaler_mode="source",
):
    csv_path = resolve_csv_path(config)
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)

    rng = np.random.default_rng(config.random_seed)
    condition_table = build_condition_table(len(df), target_condition)
    source_condition_names = set(source_condition_names) if source_condition_names is not None else None
    train_parts, valid_parts = [], []
    train_label_parts, valid_label_parts = [], []
    target_end_indices = None
    source_conditions = []

    for interval in CONDITION_INTERVALS:
        start, end = _clip_interval(interval, len(df))
        end_indices = _make_window_end_indices(start, end, seq_len)
        name = interval["name"]
        if name == target_condition:
            target_end_indices = end_indices
        elif source_condition_names is None or name in source_condition_names:
            condition_id = len(source_conditions)
            source_conditions.append(name)
            sampled = _sample_sorted(end_indices, max_source_per_condition, rng)
            train_idx, valid_idx = _split_train_valid(sampled, source_valid_ratio)
            train_parts.append(train_idx)
            valid_parts.append(valid_idx)
            train_label_parts.append(np.full(len(train_idx), condition_id, dtype=np.int64))
            valid_label_parts.append(np.full(len(valid_idx), condition_id, dtype=np.int64))

    if not train_parts:
        raise ValueError("No source conditions selected. Use --source-conditions with valid non-target names.")
    if target_end_indices is None or len(target_end_indices) == 0:
        raise ValueError(f"Unknown or empty target_condition={target_condition}. Choose from: {condition_names()}")

    if source_condition_names is not None:
        condition_table.loc[
            ~condition_table["condition"].isin(source_conditions + [target_condition]),
            "role",
        ] = "excluded"

    source_train_end, source_train_condition_ids = _concat_sorted_with_labels(train_parts, train_label_parts)
    source_valid_end, source_valid_condition_ids = _concat_sorted_with_labels(valid_parts, valid_label_parts)
    target_unlabeled_end = _sample_sorted(target_end_indices, max_target_unlabeled, rng)
    target_eval_end = np.sort(target_end_indices)

    X_source_train_raw, y_source_train_raw = _build_windows(X_all, y_all, source_train_end, seq_len)
    X_source_valid_raw, y_source_valid_raw = _build_windows(X_all, y_all, source_valid_end, seq_len)
    X_target_unlabeled_raw, _ = _build_windows(X_all, y_all, target_unlabeled_end, seq_len)
    X_target_eval_raw, y_target_eval_raw = _build_windows(X_all, y_all, target_eval_end, seq_len)

    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    source_flat = X_source_train_raw.reshape(-1, X_source_train_raw.shape[-1])
    if x_scaler_mode == "source_target":
        target_flat = X_target_unlabeled_raw.reshape(-1, X_target_unlabeled_raw.shape[-1])
        X_scaler.fit(np.vstack([source_flat, target_flat]))
    else:
        X_scaler.fit(source_flat)
    y_scaler.fit(y_source_train_raw)

    X_source_train = _transform_windows(X_scaler, X_source_train_raw)
    X_source_valid = _transform_windows(X_scaler, X_source_valid_raw)
    X_target_unlabeled = _transform_windows(X_scaler, X_target_unlabeled_raw)
    X_target_eval = _transform_windows(X_scaler, X_target_eval_raw)
    y_source_train = y_scaler.transform(y_source_train_raw).astype(np.float32)
    y_source_valid = y_scaler.transform(y_source_valid_raw).astype(np.float32)

    return SequenceCrossDomainData(
        target_condition=target_condition,
        source_conditions=source_conditions,
        condition_table=condition_table,
        X_source_train=X_source_train,
        y_source_train=y_source_train,
        X_source_valid=X_source_valid,
        y_source_valid=y_source_valid,
        y_source_valid_raw=y_source_valid_raw,
        X_target_unlabeled=X_target_unlabeled,
        X_target_eval=X_target_eval,
        y_target_eval_raw=y_target_eval_raw,
        target_eval_indices=target_eval_end,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
        source_train_condition_ids=source_train_condition_ids,
        source_valid_condition_ids=source_valid_condition_ids,
    )


def train_sequence_domain_model(
    model,
    data,
    method_name,
    epochs,
    batch_size,
    learning_rate,
    patience,
    weight_decay,
    alignment_weight,
    device,
    seed,
    model_path,
):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    source_dataset = TensorDataset(to_float_tensor(data.X_source_train), to_float_tensor(data.y_source_train))
    target_dataset = TensorDataset(to_float_tensor(data.X_target_unlabeled))
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 1))

    X_valid_tensor = to_float_tensor(data.X_source_valid).to(device)
    y_valid_tensor = to_float_tensor(data.y_source_valid).to(device)

    history = {"train_loss": [], "valid_loss": [], "alignment_loss": [], "total_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        target_iter = cycle(target_loader)
        source_losses, alignment_losses, total_losses = [], [], []

        for xb_source, yb_source in source_loader:
            xb_target = next(target_iter)[0]
            xb_source = xb_source.to(device)
            yb_source = yb_source.to(device)
            xb_target = xb_target.to(device)

            optimizer.zero_grad()
            pred_source, features_source = model(xb_source, return_features=True)
            source_loss = criterion(pred_source, yb_source)

            if method_name.endswith("source_only"):
                alignment = pred_source.new_tensor(0.0)
            else:
                _, features_target = model(xb_target, return_features=True)
                if method_name.endswith("coral"):
                    alignment = coral_loss(features_source, features_target)
                elif method_name.endswith("mmd"):
                    alignment = mmd_loss(features_source, features_target)
                else:
                    raise ValueError(f"Unknown method_name={method_name}")

            total_loss = source_loss + alignment_weight * alignment
            total_loss.backward()
            optimizer.step()

            source_losses.append(source_loss.item())
            alignment_losses.append(alignment.item())
            total_losses.append(total_loss.item())

        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(X_valid_tensor), y_valid_tensor).item()

        history["train_loss"].append(float(np.mean(source_losses)))
        history["valid_loss"].append(valid_loss)
        history["alignment_loss"].append(float(np.mean(alignment_losses)))
        history["total_loss"].append(float(np.mean(total_losses)))

        if valid_loss < best_valid_loss - 1e-8:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | source_loss={history['train_loss'][-1]:.6f} "
                f"| align_loss={history['alignment_loss'][-1]:.6f} | valid_loss={valid_loss:.6f}"
            )
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, model_path)
    return model, history, best_epoch, best_valid_loss


def train_gru_moe_domain_model(
    model,
    data,
    method_name,
    epochs,
    batch_size,
    learning_rate,
    patience,
    weight_decay,
    alignment_weight,
    expert_loss_weight,
    gate_loss_weight,
    device,
    seed,
    model_path,
):
    if data.source_train_condition_ids is None:
        raise ValueError("MoE training needs source_train_condition_ids from load_sequence_cross_domain_data.")

    model = model.to(device)
    regression_criterion = nn.MSELoss()
    gate_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    source_dataset = TensorDataset(
        to_float_tensor(data.X_source_train),
        to_float_tensor(data.y_source_train),
        torch.as_tensor(data.source_train_condition_ids, dtype=torch.long),
    )
    target_dataset = TensorDataset(to_float_tensor(data.X_target_unlabeled))
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed + 1))

    X_valid_tensor = to_float_tensor(data.X_source_valid).to(device)
    y_valid_tensor = to_float_tensor(data.y_source_valid).to(device)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "expert_loss": [],
        "gate_loss": [],
        "alignment_loss": [],
        "total_loss": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        target_iter = cycle(target_loader)
        mixture_losses, expert_losses, gate_losses, alignment_losses, total_losses = [], [], [], [], []

        for xb_source, yb_source, condition_ids in source_loader:
            xb_target = next(target_iter)[0]
            xb_source = xb_source.to(device)
            yb_source = yb_source.to(device)
            condition_ids = condition_ids.to(device)
            xb_target = xb_target.to(device)

            optimizer.zero_grad()
            pred_source, features_source, aux_source = model(xb_source, return_aux=True)
            mixture_loss = regression_criterion(pred_source, yb_source)
            selected_expert_pred = aux_source["expert_preds"].gather(1, condition_ids.view(-1, 1))
            expert_loss = regression_criterion(selected_expert_pred, yb_source)
            gate_loss = gate_criterion(aux_source["gate_logits"], condition_ids)

            if method_name.endswith("coral"):
                _, features_target = model(xb_target, return_features=True)
                alignment = coral_loss(features_source, features_target)
            else:
                alignment = pred_source.new_tensor(0.0)

            total_loss = (
                mixture_loss
                + expert_loss_weight * expert_loss
                + gate_loss_weight * gate_loss
                + alignment_weight * alignment
            )
            total_loss.backward()
            optimizer.step()

            mixture_losses.append(mixture_loss.item())
            expert_losses.append(expert_loss.item())
            gate_losses.append(gate_loss.item())
            alignment_losses.append(alignment.item())
            total_losses.append(total_loss.item())

        model.eval()
        with torch.no_grad():
            valid_loss = regression_criterion(model(X_valid_tensor), y_valid_tensor).item()

        history["train_loss"].append(float(np.mean(mixture_losses)))
        history["valid_loss"].append(valid_loss)
        history["expert_loss"].append(float(np.mean(expert_losses)))
        history["gate_loss"].append(float(np.mean(gate_losses)))
        history["alignment_loss"].append(float(np.mean(alignment_losses)))
        history["total_loss"].append(float(np.mean(total_losses)))

        if valid_loss < best_valid_loss - 1e-8:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | mix_loss={history['train_loss'][-1]:.6f} "
                f"| expert_loss={history['expert_loss'][-1]:.6f} | gate_loss={history['gate_loss'][-1]:.6f} "
                f"| align_loss={history['alignment_loss'][-1]:.6f} | valid_loss={valid_loss:.6f}"
            )
        if wait >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, model_path)
    return model, history, best_epoch, best_valid_loss


def evaluate_sequence_model(model, X, y_raw, y_scaler, device):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(to_float_tensor(X)), batch_size=1024, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    pred_scaled = np.vstack(preds)
    pred = inverse_y(y_scaler, pred_scaled)
    metrics = evaluate_regression(y_raw, pred)
    return pred, metrics


def extract_sequence_features(model, X, device, batch_size=1024):
    model.eval()
    features = []
    loader = DataLoader(TensorDataset(to_float_tensor(X)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            features.append(model.extract_features(xb.to(device)).cpu().numpy())
    return np.vstack(features)


def plot_sequence_tsne(model, data, device, output_path, max_per_domain=600, seed=42):
    rng = np.random.default_rng(seed)
    n_source = min(max_per_domain, len(data.X_source_valid))
    n_target = min(max_per_domain, len(data.X_target_eval))
    source_idx = rng.choice(len(data.X_source_valid), size=n_source, replace=False)
    target_idx = rng.choice(len(data.X_target_eval), size=n_target, replace=False)

    source_features = extract_sequence_features(model, data.X_source_valid[source_idx], device)
    target_features = extract_sequence_features(model, data.X_target_eval[target_idx], device)
    features = np.vstack([source_features, target_features])
    labels = np.array(["Source valid"] * n_source + ["Target eval"] * n_target)
    if len(features) < 10:
        return

    perplexity = max(5, min(30, (len(features) - 1) // 3))
    embedding = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate=200.0, random_state=seed).fit_transform(features)

    plt.figure(figsize=(8, 6))
    for label, color in [("Source valid", "tab:blue"), ("Target eval", "tab:red")]:
        mask = labels == label
        plt.scatter(embedding[mask, 0], embedding[mask, 1], s=12, alpha=0.65, label=label, c=color)
    plt.title("t-SNE of GRU hidden features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def predict_moe_gate_weights(model, X, device, batch_size=1024):
    model.eval()
    weights = []
    loader = DataLoader(TensorDataset(to_float_tensor(X)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            _, _, aux = model(xb.to(device), return_aux=True)
            weights.append(aux["gate_weights"].cpu().numpy())
    return np.vstack(weights)


def save_moe_gate_summary(model, data, device, output_path):
    rows = []
    source_weights = predict_moe_gate_weights(model, data.X_source_valid, device)
    target_weights = predict_moe_gate_weights(model, data.X_target_eval, device)

    for condition_id, condition_name in enumerate(data.source_conditions):
        mask = data.source_valid_condition_ids == condition_id
        if not np.any(mask):
            continue
        mean_weights = source_weights[mask].mean(axis=0)
        for expert_id, weight in enumerate(mean_weights):
            rows.append(
                {
                    "domain": "source_valid",
                    "condition": condition_name,
                    "expert_id": expert_id,
                    "expert_condition": data.source_conditions[expert_id],
                    "mean_gate_weight": float(weight),
                }
            )

    mean_target_weights = target_weights.mean(axis=0)
    for expert_id, weight in enumerate(mean_target_weights):
        rows.append(
            {
                "domain": "target_eval",
                "condition": data.target_condition,
                "expert_id": expert_id,
                "expert_condition": data.source_conditions[expert_id],
                "mean_gate_weight": float(weight),
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


def update_gru_comparison(project_root, metrics_row):
    path = project_root / "results" / "cross_domain" / "gru_comparison_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([metrics_row])
    if path.exists():
        old = pd.read_csv(path)
        for column in new_row.columns:
            if column not in old.columns:
                old[column] = np.nan
        for column in old.columns:
            if column not in new_row.columns:
                new_row[column] = np.nan
        new_row = new_row[old.columns]
        key_columns = [
            "method",
            "target_condition",
            "seq_len",
            "hidden_dim",
            "num_layers",
            "dropout",
            "x_scaler_mode",
            "alignment_weight",
        ]
        for optional_column in ["expert_loss_weight", "gate_loss_weight", "gate_temperature"]:
            if optional_column in old.columns:
                key_columns.append(optional_column)
        old_keys = old[key_columns].fillna("").astype(str).agg("|".join, axis=1)
        new_key = new_row[key_columns].fillna("").astype(str).agg("|".join, axis=1).iloc[0]
        combined = pd.concat([old[old_keys != new_key], new_row], ignore_index=True)
    else:
        combined = new_row
    combined = combined.sort_values(["Target RMSE", "target_condition", "method"])
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_gru_moe_cross_domain_experiment(project_root, method_name, args):
    if method_name not in {"gru_moe", "gru_moe_coral"}:
        raise ValueError("method_name must be 'gru_moe' or 'gru_moe_coral'.")

    config = ExperimentConfig()
    set_random_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = make_gru_moe_output_dirs(
        project_root=project_root,
        method_name=method_name,
        target_condition=args.target_condition,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        alignment_weight=args.alignment_weight,
        x_scaler_mode=args.x_scaler_mode,
        expert_loss_weight=args.expert_loss_weight,
        gate_loss_weight=args.gate_loss_weight,
        gate_temperature=args.gate_temperature,
    )

    data = load_sequence_cross_domain_data(
        config=config,
        target_condition=args.target_condition,
        seq_len=args.seq_len,
        max_source_per_condition=args.max_source_per_condition,
        max_target_unlabeled=args.max_target_unlabeled,
        source_valid_ratio=args.source_valid_ratio,
        source_condition_names=args.source_conditions,
        x_scaler_mode=args.x_scaler_mode,
    )
    data.condition_table.to_csv(dirs["metrics"] / "condition_split.csv", index=False, encoding="utf-8-sig")
    if getattr(args, "make_plots", False):
        save_condition_overview(config, data.target_condition, dirs)

    print(f"Using device: {device}")
    print(f"Method: {method_name}")
    print(f"Target condition: {data.target_condition}")
    print(f"Source conditions / experts: {', '.join(data.source_conditions)}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Source train: {data.X_source_train.shape}, source valid: {data.X_source_valid.shape}")
    print(f"Target unlabeled: {data.X_target_unlabeled.shape}, target eval: {data.X_target_eval.shape}")

    model = GRUMoERegressor(
        input_dim=data.X_source_train.shape[-1],
        hidden_dim=args.hidden_dim,
        num_experts=len(data.source_conditions),
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        gate_temperature=args.gate_temperature,
    )

    model, history, best_epoch, best_valid_loss = train_gru_moe_domain_model(
        model=model,
        data=data,
        method_name=method_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        weight_decay=config.supervised_weight_decay,
        alignment_weight=args.alignment_weight,
        expert_loss_weight=args.expert_loss_weight,
        gate_loss_weight=args.gate_loss_weight,
        device=device,
        seed=config.random_seed,
        model_path=dirs["models"] / f"best_{method_name}.pt",
    )

    _, source_valid_metrics = evaluate_sequence_model(model, data.X_source_valid, data.y_source_valid_raw, data.y_scaler, device)
    target_pred, target_metrics = evaluate_sequence_model(model, data.X_target_eval, data.y_target_eval_raw, data.y_scaler, device)

    metrics_row = {
        "method": method_name,
        "target_condition": data.target_condition,
        "source_conditions": ";".join(data.source_conditions),
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "x_scaler_mode": args.x_scaler_mode,
        "alignment_weight": args.alignment_weight,
        "num_experts": len(data.source_conditions),
        "expert_loss_weight": args.expert_loss_weight,
        "gate_loss_weight": args.gate_loss_weight,
        "gate_temperature": args.gate_temperature,
        "best_epoch": best_epoch,
        "best_source_valid_loss_scaled": best_valid_loss,
        "Source Valid RMSE": source_valid_metrics["RMSE"],
        "Source Valid MAE": source_valid_metrics["MAE"],
        "Source Valid R2": source_valid_metrics["R²"],
        "Target RMSE": target_metrics["RMSE"],
        "Target MAE": target_metrics["MAE"],
        "Target R2": target_metrics["R²"],
    }
    pd.DataFrame([metrics_row]).to_csv(dirs["metrics"] / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(dirs["metrics"] / "loss_history.csv", index_label="epoch", encoding="utf-8-sig")
    save_moe_gate_summary(model, data, device, dirs["metrics"] / "gate_summary.csv")

    pd.DataFrame(
        {
            "target_eval_index": data.target_eval_indices,
            "y_true": data.y_target_eval_raw.reshape(-1),
            "y_pred": target_pred.reshape(-1),
        }
    ).to_csv(dirs["metrics"] / "target_predictions.csv", index=False, encoding="utf-8-sig")

    if getattr(args, "make_plots", False):
        plot_loss_curve(history, f"{method_name} GRU-MoE Train / Valid Loss", dirs["figures"] / "loss_curve.png", best_epoch=best_epoch)
        plot_prediction_curve(data.target_eval_indices, data.y_target_eval_raw, target_pred, f"{method_name} target prediction curve", dirs["figures"] / "target_prediction_curve.png")
        plot_true_pred(data.y_target_eval_raw, target_pred, f"{method_name} target true vs predicted", dirs["figures"] / "target_true_pred_scatter.png")
        plot_residuals(data.y_target_eval_raw, target_pred, f"{method_name} target residual distribution", dirs["figures"] / "target_residual.png")
        plot_sequence_tsne(model, data, device, dirs["figures"] / "feature_tsne.png", max_per_domain=args.tsne_max_per_domain, seed=config.random_seed)

    comparison_path = update_gru_comparison(project_root, metrics_row)
    print("\nTarget metrics:")
    print(pd.DataFrame([metrics_row]).to_string(index=False))
    print(f"\nSaved GRU-MoE method summary to: {dirs['metrics'] / 'summary.csv'}")
    print(f"Updated GRU comparison summary: {comparison_path}")
    return metrics_row


def run_gru_cross_domain_experiment(project_root, method_name, args):
    config = ExperimentConfig()
    set_random_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = make_sequence_output_dirs(project_root, method_name, args.target_condition, args.seq_len, args.hidden_dim, args.dropout, args.alignment_weight, args.x_scaler_mode)

    data = load_sequence_cross_domain_data(
        config=config,
        target_condition=args.target_condition,
        seq_len=args.seq_len,
        max_source_per_condition=args.max_source_per_condition,
        max_target_unlabeled=args.max_target_unlabeled,
        source_valid_ratio=args.source_valid_ratio,
        source_condition_names=args.source_conditions,
        x_scaler_mode=args.x_scaler_mode,
    )
    data.condition_table.to_csv(dirs["metrics"] / "condition_split.csv", index=False, encoding="utf-8-sig")
    if getattr(args, "make_plots", True):
        save_condition_overview(config, data.target_condition, dirs)

    print(f"Using device: {device}")
    print(f"Method: {method_name}")
    print(f"Target condition: {data.target_condition}")
    print(f"Source conditions: {', '.join(data.source_conditions)}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Source train: {data.X_source_train.shape}, source valid: {data.X_source_valid.shape}")
    print(f"Target unlabeled: {data.X_target_unlabeled.shape}, target eval: {data.X_target_eval.shape}")

    model = GRUDomainRegressor(
        input_dim=data.X_source_train.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
    )

    model, history, best_epoch, best_valid_loss = train_sequence_domain_model(
        model=model,
        data=data,
        method_name=method_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        weight_decay=config.supervised_weight_decay,
        alignment_weight=args.alignment_weight,
        device=device,
        seed=config.random_seed,
        model_path=dirs["models"] / f"best_{method_name}.pt",
    )

    _, source_valid_metrics = evaluate_sequence_model(model, data.X_source_valid, data.y_source_valid_raw, data.y_scaler, device)
    target_pred, target_metrics = evaluate_sequence_model(model, data.X_target_eval, data.y_target_eval_raw, data.y_scaler, device)

    metrics_row = {
        "method": method_name,
        "target_condition": data.target_condition,
        "source_conditions": ";".join(data.source_conditions),
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "x_scaler_mode": args.x_scaler_mode,
        "alignment_weight": args.alignment_weight,
        "best_epoch": best_epoch,
        "best_source_valid_loss_scaled": best_valid_loss,
        "Source Valid RMSE": source_valid_metrics["RMSE"],
        "Source Valid MAE": source_valid_metrics["MAE"],
        "Source Valid R2": source_valid_metrics["R²"],
        "Target RMSE": target_metrics["RMSE"],
        "Target MAE": target_metrics["MAE"],
        "Target R2": target_metrics["R²"],
    }
    pd.DataFrame([metrics_row]).to_csv(dirs["metrics"] / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(dirs["metrics"] / "loss_history.csv", index_label="epoch", encoding="utf-8-sig")

    if getattr(args, "make_plots", True):
        plot_loss_curve(history, f"{method_name} GRU Train / Valid Loss", dirs["figures"] / "loss_curve.png", best_epoch=best_epoch)
        plot_prediction_curve(data.target_eval_indices, data.y_target_eval_raw, target_pred, f"{method_name} target prediction curve", dirs["figures"] / "target_prediction_curve.png")
        plot_true_pred(data.y_target_eval_raw, target_pred, f"{method_name} target true vs predicted", dirs["figures"] / "target_true_pred_scatter.png")
        plot_residuals(data.y_target_eval_raw, target_pred, f"{method_name} target residual distribution", dirs["figures"] / "target_residual.png")
        plot_sequence_tsne(model, data, device, dirs["figures"] / "feature_tsne.png", max_per_domain=args.tsne_max_per_domain, seed=config.random_seed)

    path = project_root / "results" / "cross_domain" / "gru_comparison_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([metrics_row])
    if path.exists():
        old = pd.read_csv(path)
        if "x_scaler_mode" not in old.columns:
            old["x_scaler_mode"] = "source"
        keep = ~(
            (old["method"] == metrics_row["method"])
            & (old["target_condition"] == metrics_row["target_condition"])
            & (old["seq_len"].astype(int) == int(metrics_row["seq_len"]))
            & (old["hidden_dim"].astype(int) == int(metrics_row["hidden_dim"]))
            & (old["num_layers"].astype(int) == int(metrics_row["num_layers"]))
            & (old["dropout"].astype(float) == float(metrics_row["dropout"]))
            & (old["x_scaler_mode"] == metrics_row["x_scaler_mode"])
            & (old["alignment_weight"].astype(float) == float(metrics_row["alignment_weight"]))
        )
        combined = pd.concat([old[keep], row_df], ignore_index=True)
    else:
        combined = row_df
    combined = combined.sort_values(["Target RMSE", "target_condition", "method"])
    combined.to_csv(path, index=False, encoding="utf-8-sig")

    print("\nTarget metrics:")
    print(pd.DataFrame([metrics_row]).to_string(index=False))
    print(f"\nSaved GRU method summary to: {dirs['metrics'] / 'summary.csv'}")
    print(f"Updated GRU comparison summary: {path}")
    return metrics_row









