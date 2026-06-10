import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cross_domain.common import CONDITION_INTERVALS, build_condition_table, condition_names, load_raw_dataframe
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.evaluation import evaluate_regression, inverse_y
from unit03_soft_sensor.models import MLPRegressor
from unit03_soft_sensor.train import to_float_tensor


MLP_BASELINE_RMSE = 1.4514
MLP_BASELINE_R2 = 0.9116


@dataclass
class AECLConditionData:
    target_condition: str
    source_conditions: list
    condition_table: pd.DataFrame
    condition_id_to_name: dict
    X_train_all: np.ndarray
    y_train_all: np.ndarray
    label_mask: np.ndarray
    condition_ids: np.ndarray
    supervision_weights: np.ndarray
    X_target_train: np.ndarray
    y_target_train: np.ndarray
    X_target_valid: np.ndarray
    y_target_valid: np.ndarray
    X_target_eval: np.ndarray
    y_target_eval_raw: np.ndarray
    target_eval_indices: np.ndarray
    X_scaler: StandardScaler
    y_scaler: StandardScaler


class AECLRegressor(nn.Module):
    def __init__(
        self,
        input_dim=13,
        latent_dim=32,
        encoder_hidden_sizes=(128, 64),
        regressor_hidden_sizes=(64, 64),
        dropout_rate=0.1,
    ):
        super().__init__()
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in encoder_hidden_sizes:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(tuple(encoder_hidden_sizes)):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        regressor_layers = []
        prev_dim = latent_dim
        for hidden_dim in regressor_hidden_sizes:
            regressor_layers.append(nn.Linear(prev_dim, hidden_dim))
            regressor_layers.append(nn.ReLU())
            if dropout_rate > 0:
                regressor_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        regressor_layers.append(nn.Linear(prev_dim, 1))
        self.regressor = nn.Sequential(*regressor_layers)

    def forward(self, x, return_latent=False):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        y_pred = self.regressor(z)
        if return_latent:
            return x_rec, z, y_pred
        return y_pred

    def predict(self, x):
        return self.forward(x)


def _clip_interval(interval, n_rows):
    start = max(0, min(int(interval["start"]), n_rows))
    end = max(start, min(int(interval["end"]), n_rows))
    return start, end


def _sample_sorted(indices, max_count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if max_count is not None and max_count > 0 and len(indices) > max_count:
        indices = rng.choice(indices, size=max_count, replace=False)
    return np.sort(indices)


def _split_train_valid(indices, valid_ratio, seed):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    if len(indices) < 2:
        raise ValueError("Need at least two calibration samples for train/valid split.")
    rng = np.random.default_rng(seed)
    shuffled = np.array(indices, copy=True)
    rng.shuffle(shuffled)
    n_valid = max(1, int(round(len(shuffled) * valid_ratio)))
    valid_idx = np.sort(shuffled[:n_valid])
    train_idx = np.sort(shuffled[n_valid:])
    if len(train_idx) == 0:
        train_idx = valid_idx[:1]
        valid_idx = valid_idx[1:]
    if len(valid_idx) == 0:
        valid_idx = train_idx[-1:]
        train_idx = train_idx[:-1]
    return train_idx, valid_idx


def _condition_index_ranges(n_rows):
    rows = []
    for interval in CONDITION_INTERVALS:
        start, end = _clip_interval(interval, n_rows)
        rows.append((interval["name"], np.arange(start, end, dtype=np.int64)))
    return rows


def _fit_x_scaler(mode, X_source, X_target_train, X_target_unlabeled):
    scaler = StandardScaler()
    if mode == "target":
        fit_X = X_target_train
    elif mode == "source":
        fit_X = X_source if len(X_source) > 0 else X_target_train
    elif mode == "target_unlabeled":
        fit_X = np.vstack([X_target_train, X_target_unlabeled])
    elif mode == "source_target":
        chunks = [X_target_train, X_target_unlabeled]
        if len(X_source) > 0:
            chunks.insert(0, X_source)
        fit_X = np.vstack(chunks)
    else:
        raise ValueError(f"Unknown x_scaler_mode={mode}")
    scaler.fit(fit_X)
    return scaler


def _fit_y_scaler(mode, y_source, y_target_train):
    scaler = StandardScaler()
    if mode == "target":
        fit_y = y_target_train
    elif mode == "labeled":
        fit_y = np.vstack([y_source, y_target_train]) if len(y_source) > 0 else y_target_train
    else:
        raise ValueError(f"Unknown y_scaler_mode={mode}")
    scaler.fit(fit_y)
    return scaler


def load_aecl_condition_data(
    config: ExperimentConfig,
    target_condition,
    n_calibration,
    seed,
    valid_ratio=0.2,
    max_source_per_condition=3000,
    max_target_unlabeled=6000,
    x_scaler_mode="source_target",
    y_scaler_mode="target",
    include_source=True,
    target_supervision_weight=3.0,
    source_supervision_weight=1.0,
):
    df = load_raw_dataframe(config)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
    rng = np.random.default_rng(seed)

    target_indices = None
    source_indices_parts = []
    source_condition_parts = []
    source_conditions = []
    condition_names_all = condition_names()
    condition_to_id = {name: idx for idx, name in enumerate(condition_names_all)}

    for name, indices in _condition_index_ranges(len(df)):
        if name == target_condition:
            target_indices = indices
        else:
            source_conditions.append(name)
            if include_source:
                sampled = _sample_sorted(indices, max_source_per_condition, rng)
                source_indices_parts.append(sampled)
                source_condition_parts.append(np.full(len(sampled), condition_to_id[name], dtype=np.int64))

    if target_indices is None or len(target_indices) < 2:
        raise ValueError(f"Unknown or empty target_condition={target_condition}. Choose from: {condition_names_all}")

    n_cal = min(int(n_calibration), len(target_indices) - 1)
    if n_cal < 2:
        raise ValueError("n_calibration must leave both calibration and evaluation samples.")
    calibration_indices = np.sort(rng.choice(target_indices, size=n_cal, replace=False))
    target_train_idx, target_valid_idx = _split_train_valid(calibration_indices, valid_ratio, seed + 1)
    eval_mask = np.ones(len(target_indices), dtype=bool)
    eval_mask[np.searchsorted(target_indices, calibration_indices)] = False
    target_eval_idx = np.sort(target_indices[eval_mask])
    target_unlabeled_idx = _sample_sorted(target_eval_idx, max_target_unlabeled, rng)

    if include_source and source_indices_parts:
        source_indices = np.sort(np.concatenate(source_indices_parts))
        source_condition_ids = np.concatenate(source_condition_parts)[np.argsort(np.concatenate(source_indices_parts))]
    else:
        source_indices = np.array([], dtype=np.int64)
        source_condition_ids = np.array([], dtype=np.int64)

    X_source_raw = X_all[source_indices] if len(source_indices) > 0 else np.empty((0, X_all.shape[1]), dtype=np.float32)
    y_source_raw = y_all[source_indices] if len(source_indices) > 0 else np.empty((0, 1), dtype=np.float32)
    X_target_train_raw = X_all[target_train_idx]
    y_target_train_raw = y_all[target_train_idx]
    X_target_valid_raw = X_all[target_valid_idx]
    y_target_valid_raw = y_all[target_valid_idx]
    X_target_unlabeled_raw = X_all[target_unlabeled_idx]
    X_target_eval_raw = X_all[target_eval_idx]
    y_target_eval_raw = y_all[target_eval_idx]

    X_scaler = _fit_x_scaler(x_scaler_mode, X_source_raw, X_target_train_raw, X_target_unlabeled_raw)
    y_scaler = _fit_y_scaler(y_scaler_mode, y_source_raw, y_target_train_raw)

    X_source = X_scaler.transform(X_source_raw).astype(np.float32) if len(X_source_raw) else X_source_raw
    y_source = y_scaler.transform(y_source_raw).astype(np.float32) if len(y_source_raw) else y_source_raw
    X_target_train = X_scaler.transform(X_target_train_raw).astype(np.float32)
    y_target_train = y_scaler.transform(y_target_train_raw).astype(np.float32)
    X_target_valid = X_scaler.transform(X_target_valid_raw).astype(np.float32)
    y_target_valid = y_scaler.transform(y_target_valid_raw).astype(np.float32)
    X_target_unlabeled = X_scaler.transform(X_target_unlabeled_raw).astype(np.float32)
    X_target_eval = X_scaler.transform(X_target_eval_raw).astype(np.float32)

    target_condition_id = condition_to_id[target_condition]
    chunks_X = []
    chunks_y = []
    chunks_mask = []
    chunks_cond = []
    chunks_weight = []

    if include_source and len(X_source) > 0 and source_supervision_weight > 0:
        chunks_X.append(X_source)
        chunks_y.append(y_source)
        chunks_mask.append(np.ones(len(X_source), dtype=np.float32))
        chunks_cond.append(source_condition_ids)
        chunks_weight.append(np.full(len(X_source), float(source_supervision_weight), dtype=np.float32))

    chunks_X.append(X_target_train)
    chunks_y.append(y_target_train)
    chunks_mask.append(np.ones(len(X_target_train), dtype=np.float32))
    chunks_cond.append(np.full(len(X_target_train), target_condition_id, dtype=np.int64))
    chunks_weight.append(np.full(len(X_target_train), float(target_supervision_weight), dtype=np.float32))

    if len(X_target_unlabeled) > 0:
        chunks_X.append(X_target_unlabeled)
        chunks_y.append(np.zeros((len(X_target_unlabeled), 1), dtype=np.float32))
        chunks_mask.append(np.zeros(len(X_target_unlabeled), dtype=np.float32))
        chunks_cond.append(np.full(len(X_target_unlabeled), target_condition_id, dtype=np.int64))
        chunks_weight.append(np.zeros(len(X_target_unlabeled), dtype=np.float32))

    X_train_all = np.vstack(chunks_X).astype(np.float32)
    y_train_all = np.vstack(chunks_y).astype(np.float32)
    label_mask = np.concatenate(chunks_mask).astype(np.float32)
    condition_ids = np.concatenate(chunks_cond).astype(np.int64)
    supervision_weights = np.concatenate(chunks_weight).astype(np.float32)

    return AECLConditionData(
        target_condition=target_condition,
        source_conditions=source_conditions,
        condition_table=build_condition_table(len(df), target_condition),
        condition_id_to_name={idx: name for name, idx in condition_to_id.items()},
        X_train_all=X_train_all,
        y_train_all=y_train_all,
        label_mask=label_mask,
        condition_ids=condition_ids,
        supervision_weights=supervision_weights,
        X_target_train=X_target_train,
        y_target_train=y_target_train,
        X_target_valid=X_target_valid,
        y_target_valid=y_target_valid,
        X_target_eval=X_target_eval,
        y_target_eval_raw=y_target_eval_raw,
        target_eval_indices=target_eval_idx,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
    )


def weighted_masked_mse(y_pred, y_true, label_mask, sample_weights):
    mask = label_mask.view(-1) > 0.5
    if mask.sum() == 0:
        return y_pred.new_tensor(0.0)
    weights = sample_weights.view(-1)[mask]
    loss = (y_pred[mask] - y_true[mask]).pow(2).view(-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


def supervised_contrastive_loss(z, condition_ids, label_mask=None, temperature=0.2):
    labels = condition_ids.view(-1)
    valid = labels >= 0
    if label_mask is not None:
        valid = valid & (label_mask.view(-1) > 0.5)
    z = z[valid]
    labels = labels[valid]
    if z.shape[0] < 2:
        return z.new_tensor(0.0)

    z = nn.functional.normalize(z, dim=1)
    logits = z.matmul(z.t()) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    pos_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & (~self_mask)
    valid_anchor = pos_mask.sum(dim=1) > 0
    if valid_anchor.sum() == 0:
        return z.new_tensor(0.0)
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[valid_anchor].mean()


def train_mlp_calibration(data, args, device, seed, model_path=None):
    model = MLPRegressor(
        input_dim=data.X_target_train.shape[1],
        hidden_sizes=tuple(args.mlp_hidden_sizes),
        activation_name="relu",
        dropout_rate=args.mlp_dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    dataset = TensorDataset(to_float_tensor(data.X_target_train), to_float_tensor(data.y_target_train))
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    X_valid = to_float_tensor(data.X_target_valid).to(device)
    y_valid = to_float_tensor(data.y_target_valid).to(device)
    history = {"train_loss": [], "valid_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_valid = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(X_valid), y_valid).item()
        history["train_loss"].append(float(np.mean(losses)))
        history["valid_loss"].append(valid_loss)
        if valid_loss < best_valid - 1e-8:
            best_valid = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break

    model.load_state_dict(best_state)
    if model_path is not None:
        torch.save(best_state, model_path)
    return model, history, best_epoch, best_valid


def train_aecl(data, args, device, seed, lambda_con, model_path=None):
    model = AECLRegressor(
        input_dim=data.X_train_all.shape[1],
        latent_dim=args.latent_dim,
        encoder_hidden_sizes=tuple(args.encoder_hidden_sizes),
        regressor_hidden_sizes=tuple(args.regressor_hidden_sizes),
        dropout_rate=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    reconstruction_loss = nn.MSELoss()
    dataset = TensorDataset(
        to_float_tensor(data.X_train_all),
        to_float_tensor(data.y_train_all),
        to_float_tensor(data.label_mask),
        torch.as_tensor(data.condition_ids, dtype=torch.long),
        to_float_tensor(data.supervision_weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    X_valid = to_float_tensor(data.X_target_valid).to(device)
    y_valid = to_float_tensor(data.y_target_valid).to(device)

    history = {
        "train_total": [],
        "train_rec": [],
        "train_sup": [],
        "train_con": [],
        "valid_sup": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_valid = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_losses, rec_losses, sup_losses, con_losses = [], [], [], []
        for xb, yb, mb, cb, wb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            cb = cb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad()
            x_rec, z, y_pred = model(xb, return_latent=True)
            rec = reconstruction_loss(x_rec, xb)
            sup = weighted_masked_mse(y_pred, yb, mb, wb)
            con_mask = mb if args.contrast_labeled_only else None
            con = supervised_contrastive_loss(z, cb, label_mask=con_mask, temperature=args.temperature)
            total = args.lambda_rec * rec + args.lambda_sup * sup + lambda_con * con
            total.backward()
            optimizer.step()
            total_losses.append(total.item())
            rec_losses.append(rec.item())
            sup_losses.append(sup.item())
            con_losses.append(con.item())

        model.eval()
        with torch.no_grad():
            valid_pred = model.predict(X_valid)
            valid_sup = nn.functional.mse_loss(valid_pred, y_valid).item()

        history["train_total"].append(float(np.mean(total_losses)))
        history["train_rec"].append(float(np.mean(rec_losses)))
        history["train_sup"].append(float(np.mean(sup_losses)))
        history["train_con"].append(float(np.mean(con_losses)))
        history["valid_sup"].append(valid_sup)

        if valid_sup < best_valid - 1e-8:
            best_valid = valid_sup
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break

    model.load_state_dict(best_state)
    if model_path is not None:
        torch.save(best_state, model_path)
    return model, history, best_epoch, best_valid


def predict_scaled(model, X, device, batch_size=2048):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(to_float_tensor(X)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            if hasattr(model, "predict"):
                pred = model.predict(xb.to(device))
            else:
                pred = model(xb.to(device))
            preds.append(pred.cpu().numpy())
    return np.vstack(preds)


def get_metric_r2(metrics):
    for key, value in metrics.items():
        if key not in {"MSE", "RMSE", "MAE"}:
            return value
    raise KeyError("R2 metric not found.")


def evaluate_target(model, data, device):
    pred_scaled = predict_scaled(model, data.X_target_eval, device)
    pred = inverse_y(data.y_scaler, pred_scaled)
    metrics = evaluate_regression(data.y_target_eval_raw, pred)
    return pred, metrics


def make_run_dirs(project_root: Path, mode, target, n_calibration, trial, args):
    tag = (
        f"ncal_{n_calibration}_trial_{trial}_latent_{args.latent_dim}"
        f"_dropout_{str(args.dropout).replace('.', 'p')}"
        f"_con_{str(args.lambda_con).replace('.', 'p')}"
    )
    root = project_root / "results" / "aecl_cross_condition" / mode / target / tag
    dirs = {"root": root, "metrics": root / "metrics", "models": root / "models"}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def aggregate_summary(summary):
    rows = []
    for (mode, n_calibration), group in summary.groupby(["mode", "n_calibration"], sort=True):
        by_condition = group.groupby("target_condition", as_index=False).agg(
            mean_rmse=("Target RMSE", "mean"),
            mean_mae=("Target MAE", "mean"),
            mean_r2=("Target R2", "mean"),
            worst_rmse=("Target RMSE", "max"),
        )
        worst_row = by_condition.loc[by_condition["worst_rmse"].idxmax()]
        rows.append(
            {
                "mode": mode,
                "n_calibration": int(n_calibration),
                "n_targets": int(by_condition["target_condition"].nunique()),
                "n_trials": int(group["trial"].nunique()),
                "avg_target_rmse": float(by_condition["mean_rmse"].mean()),
                "avg_target_mae": float(by_condition["mean_mae"].mean()),
                "avg_target_r2": float(by_condition["mean_r2"].mean()),
                "worst_rmse_condition": worst_row["target_condition"],
                "worst_target_rmse": float(worst_row["worst_rmse"]),
                "count_conditions_mean_rmse_below_mlp": int((by_condition["mean_rmse"] < MLP_BASELINE_RMSE).sum()),
                "count_conditions_mean_r2_above_mlp": int((by_condition["mean_r2"] > MLP_BASELINE_R2).sum()),
                "all_conditions_mean_rmse_below_mlp": bool((by_condition["mean_rmse"] < MLP_BASELINE_RMSE).all()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["all_conditions_mean_rmse_below_mlp", "avg_target_rmse", "worst_target_rmse"],
        ascending=[False, True, True],
    )


def condition_summary(summary):
    return (
        summary.groupby(["mode", "n_calibration", "target_condition"], as_index=False)
        .agg(
            mean_rmse=("Target RMSE", "mean"),
            std_rmse=("Target RMSE", "std"),
            mean_mae=("Target MAE", "mean"),
            mean_r2=("Target R2", "mean"),
            worst_trial_rmse=("Target RMSE", "max"),
            count_trials_rmse_below_mlp=("rmse_below_mlp", "sum"),
        )
        .sort_values(["mode", "n_calibration", "target_condition"])
    )
