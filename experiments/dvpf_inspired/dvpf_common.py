import copy
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.aecl_cross_condition.aecl_common import supervised_contrastive_loss
from experiments.cross_domain.common import (
    CONDITION_INTERVALS,
    build_condition_table,
    condition_names,
    coral_loss,
    load_raw_dataframe,
    mmd_loss,
)
from experiments.cross_domain.sequence_common import evaluate_sequence_model
from unit03_soft_sensor.train import to_float_tensor


MLP_BASELINE_RMSE = 1.4514
MLP_BASELINE_R2 = 0.9116


@dataclass
class DVPFTrainResult:
    model: nn.Module
    history: dict
    best_epoch: int
    best_valid_loss: float


@dataclass
class DVPFCalibrationData:
    target_condition: str
    source_conditions: list
    condition_table: pd.DataFrame
    condition_id_to_name: dict
    X_labeled_train: np.ndarray
    y_labeled_train: np.ndarray
    labeled_condition_ids: np.ndarray
    supervision_weights: np.ndarray
    X_target_valid: np.ndarray
    y_target_valid: np.ndarray
    y_target_valid_raw: np.ndarray
    X_target_unlabeled: np.ndarray
    X_target_eval: np.ndarray
    y_target_eval_raw: np.ndarray
    target_eval_indices: np.ndarray
    X_scaler: StandardScaler
    y_scaler: StandardScaler


class ResidualLatentFlow(nn.Module):
    """Lightweight residual latent flow inspired by potential-flow updates."""

    def __init__(self, latent_dim, hidden_dim=64, n_steps=1, step_size=0.2):
        super().__init__()
        self.n_steps = int(n_steps)
        self.step_size = float(step_size)
        self.field = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z):
        if self.n_steps <= 0:
            return z
        out = z
        for _ in range(self.n_steps):
            out = out + self.step_size * self.field(out)
        return out


class GRUVAEFlowRegressor(nn.Module):
    """GRU-VAE soft sensor with an optional residual latent flow."""

    def __init__(
        self,
        input_dim=13,
        seq_len=20,
        hidden_dim=64,
        latent_dim=32,
        num_layers=1,
        dropout_rate=0.05,
        flow_steps=1,
        flow_hidden_dim=64,
        flow_step_size=0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        gru_dropout = dropout_rate if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.flow = ResidualLatentFlow(
            latent_dim=latent_dim,
            hidden_dim=flow_hidden_dim,
            n_steps=flow_steps,
            step_size=flow_step_size,
        )
        self.regressor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(hidden_dim, seq_len * input_dim),
        )

    def encode(self, x):
        _, hidden = self.encoder(x)
        h = self.dropout(hidden[-1])
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-8.0, 6.0)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        rec = self.decoder(z)
        return rec.view(-1, self.seq_len, self.input_dim)

    def forward(self, x, return_latent=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z_flow = self.flow(z)
        y_pred = self.regressor(z_flow)
        x_rec = self.decode(z_flow)
        if return_latent:
            return y_pred, x_rec, mu, logvar, z_flow
        return y_pred

    def extract_features(self, x):
        mu, _ = self.encode(x)
        return self.flow(mu)


def kl_normal(mu, logvar):
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    return kl.sum(dim=1).mean()


def evaluate_dvpf_model(model, X, y_raw, y_scaler, device):
    return evaluate_sequence_model(model, X, y_raw, y_scaler, device)


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


def _split_train_valid(indices, valid_ratio, seed):
    indices = np.sort(np.asarray(indices, dtype=np.int64))
    if len(indices) < 2:
        raise ValueError("Need at least two calibration windows for train/valid split.")
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


def _fit_window_x_scaler(mode, X_source, X_target_train, X_target_unlabeled):
    scaler = StandardScaler()
    chunks = []
    if mode == "target":
        chunks = [X_target_train]
    elif mode == "source":
        chunks = [X_source] if len(X_source) > 0 else [X_target_train]
    elif mode == "target_unlabeled":
        chunks = [X_target_train, X_target_unlabeled]
    elif mode == "source_target":
        if len(X_source) > 0:
            chunks.append(X_source)
        chunks.extend([X_target_train, X_target_unlabeled])
    else:
        raise ValueError(f"Unknown x_scaler_mode={mode}")
    flat_chunks = [chunk.reshape(-1, chunk.shape[-1]) for chunk in chunks if len(chunk) > 0]
    scaler.fit(np.vstack(flat_chunks))
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


def load_dvpf_calibration_data(
    config,
    target_condition,
    n_calibration,
    seed,
    seq_len=20,
    valid_ratio=0.2,
    max_source_per_condition=3000,
    max_target_unlabeled=6000,
    x_scaler_mode="target_unlabeled",
    y_scaler_mode="target",
    include_source=True,
    target_supervision_weight=10.0,
    source_supervision_weight=0.1,
):
    df = load_raw_dataframe(config)
    X_all = df.iloc[:, :-1].values.astype(np.float32)
    y_all = df.iloc[:, -1].values.reshape(-1, 1).astype(np.float32)
    rng = np.random.default_rng(seed)

    target_end_indices = None
    source_parts, source_label_parts = [], []
    source_conditions = []
    names = condition_names()
    condition_to_id = {name: idx for idx, name in enumerate(names)}

    for interval in CONDITION_INTERVALS:
        start, end = _clip_interval(interval, len(df))
        end_indices = _make_window_end_indices(start, end, seq_len)
        name = interval["name"]
        if name == target_condition:
            target_end_indices = end_indices
        else:
            source_conditions.append(name)
            if include_source:
                sampled = _sample_sorted(end_indices, max_source_per_condition, rng)
                source_parts.append(sampled)
                source_label_parts.append(np.full(len(sampled), condition_to_id[name], dtype=np.int64))

    if target_end_indices is None or len(target_end_indices) < 2:
        raise ValueError(f"Unknown or empty target_condition={target_condition}. Choose from: {names}")

    n_cal = min(int(n_calibration), len(target_end_indices) - 1)
    if n_cal < 2:
        raise ValueError("n_calibration must leave both calibration and evaluation windows.")
    calibration_end = np.sort(rng.choice(target_end_indices, size=n_cal, replace=False))
    target_train_end, target_valid_end = _split_train_valid(calibration_end, valid_ratio, seed + 1)
    eval_mask = ~np.isin(target_end_indices, calibration_end)
    target_eval_end = np.sort(target_end_indices[eval_mask])
    target_unlabeled_end = _sample_sorted(target_eval_end, max_target_unlabeled, rng)

    if include_source and source_parts:
        source_end, source_condition_ids = _concat_sorted_with_labels(source_parts, source_label_parts)
    else:
        source_end = np.array([], dtype=np.int64)
        source_condition_ids = np.array([], dtype=np.int64)

    X_source_raw, y_source_raw = (
        _build_windows(X_all, y_all, source_end, seq_len)
        if len(source_end) > 0
        else (np.empty((0, seq_len, X_all.shape[1]), dtype=np.float32), np.empty((0, 1), dtype=np.float32))
    )
    X_target_train_raw, y_target_train_raw = _build_windows(X_all, y_all, target_train_end, seq_len)
    X_target_valid_raw, y_target_valid_raw = _build_windows(X_all, y_all, target_valid_end, seq_len)
    X_target_unlabeled_raw, _ = _build_windows(X_all, y_all, target_unlabeled_end, seq_len)
    X_target_eval_raw, y_target_eval_raw = _build_windows(X_all, y_all, target_eval_end, seq_len)

    X_scaler = _fit_window_x_scaler(x_scaler_mode, X_source_raw, X_target_train_raw, X_target_unlabeled_raw)
    y_scaler = _fit_y_scaler(y_scaler_mode, y_source_raw, y_target_train_raw)

    X_source = _transform_windows(X_scaler, X_source_raw) if len(X_source_raw) else X_source_raw
    y_source = y_scaler.transform(y_source_raw).astype(np.float32) if len(y_source_raw) else y_source_raw
    X_target_train = _transform_windows(X_scaler, X_target_train_raw)
    y_target_train = y_scaler.transform(y_target_train_raw).astype(np.float32)
    X_target_valid = _transform_windows(X_scaler, X_target_valid_raw)
    y_target_valid = y_scaler.transform(y_target_valid_raw).astype(np.float32)
    X_target_unlabeled = _transform_windows(X_scaler, X_target_unlabeled_raw)
    X_target_eval = _transform_windows(X_scaler, X_target_eval_raw)

    labeled_X, labeled_y, labeled_cond, weights = [], [], [], []
    if include_source and len(X_source) > 0 and source_supervision_weight > 0:
        labeled_X.append(X_source)
        labeled_y.append(y_source)
        labeled_cond.append(source_condition_ids)
        weights.append(np.full(len(X_source), float(source_supervision_weight), dtype=np.float32))

    target_condition_id = condition_to_id[target_condition]
    labeled_X.append(X_target_train)
    labeled_y.append(y_target_train)
    labeled_cond.append(np.full(len(X_target_train), target_condition_id, dtype=np.int64))
    weights.append(np.full(len(X_target_train), float(target_supervision_weight), dtype=np.float32))

    return DVPFCalibrationData(
        target_condition=target_condition,
        source_conditions=source_conditions,
        condition_table=build_condition_table(len(df), target_condition),
        condition_id_to_name={idx: name for name, idx in condition_to_id.items()},
        X_labeled_train=np.vstack(labeled_X).astype(np.float32),
        y_labeled_train=np.vstack(labeled_y).astype(np.float32),
        labeled_condition_ids=np.concatenate(labeled_cond).astype(np.int64),
        supervision_weights=np.concatenate(weights).astype(np.float32),
        X_target_valid=X_target_valid,
        y_target_valid=y_target_valid,
        y_target_valid_raw=y_target_valid_raw,
        X_target_unlabeled=X_target_unlabeled,
        X_target_eval=X_target_eval,
        y_target_eval_raw=y_target_eval_raw,
        target_eval_indices=target_eval_end,
        X_scaler=X_scaler,
        y_scaler=y_scaler,
    )


def train_dvpf_model(
    model,
    data,
    args,
    device,
    seed,
    model_path=None,
):
    model = model.to(device)
    regression_loss = nn.MSELoss()
    reconstruction_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    source_dataset = TensorDataset(to_float_tensor(data.X_source_train), to_float_tensor(data.y_source_train))
    target_dataset = TensorDataset(to_float_tensor(data.X_target_unlabeled))
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    X_valid = to_float_tensor(data.X_source_valid).to(device)
    y_valid = to_float_tensor(data.y_source_valid).to(device)

    history = {
        "train_total": [],
        "source_regression": [],
        "source_reconstruction": [],
        "target_reconstruction": [],
        "kl_loss": [],
        "alignment_loss": [],
        "valid_loss": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_valid = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        target_iter = cycle(target_loader)
        total_losses = []
        reg_losses = []
        src_rec_losses = []
        tgt_rec_losses = []
        kl_losses = []
        align_losses = []

        for xb_source, yb_source in source_loader:
            xb_target = next(target_iter)[0]
            xb_source = xb_source.to(device)
            yb_source = yb_source.to(device)
            xb_target = xb_target.to(device)

            optimizer.zero_grad()
            pred_source, rec_source, mu_source, logvar_source, z_source = model(xb_source, return_latent=True)
            _, rec_target, mu_target, logvar_target, z_target = model(xb_target, return_latent=True)

            reg = regression_loss(pred_source, yb_source)
            src_rec = reconstruction_loss(rec_source, xb_source)
            tgt_rec = reconstruction_loss(rec_target, xb_target)
            kl = 0.5 * (kl_normal(mu_source, logvar_source) + kl_normal(mu_target, logvar_target))

            if args.alignment == "coral":
                align = coral_loss(z_source, z_target)
            elif args.alignment == "mmd":
                align = mmd_loss(z_source, z_target)
            elif args.alignment == "none":
                align = pred_source.new_tensor(0.0)
            else:
                raise ValueError(f"Unknown alignment={args.alignment}")

            total = (
                args.lambda_reg * reg
                + args.lambda_source_rec * src_rec
                + args.lambda_target_rec * tgt_rec
                + args.lambda_kl * kl
                + args.alignment_weight * align
            )
            total.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_losses.append(total.item())
            reg_losses.append(reg.item())
            src_rec_losses.append(src_rec.item())
            tgt_rec_losses.append(tgt_rec.item())
            kl_losses.append(kl.item())
            align_losses.append(align.item())

        model.eval()
        with torch.no_grad():
            valid_pred = model(X_valid)
            valid_loss = regression_loss(valid_pred, y_valid).item()

        history["train_total"].append(float(np.mean(total_losses)))
        history["source_regression"].append(float(np.mean(reg_losses)))
        history["source_reconstruction"].append(float(np.mean(src_rec_losses)))
        history["target_reconstruction"].append(float(np.mean(tgt_rec_losses)))
        history["kl_loss"].append(float(np.mean(kl_losses)))
        history["alignment_loss"].append(float(np.mean(align_losses)))
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid - 1e-8:
            best_valid = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | total={history['train_total'][-1]:.5f} "
                f"| reg={history['source_regression'][-1]:.5f} "
                f"| tgt_rec={history['target_reconstruction'][-1]:.5f} "
                f"| kl={history['kl_loss'][-1]:.5f} | valid={valid_loss:.5f}"
            )
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    if model_path is not None:
        torch.save(best_state, model_path)
    return DVPFTrainResult(model=model, history=history, best_epoch=best_epoch, best_valid_loss=best_valid)


def weighted_mse(y_pred, y_true, sample_weights):
    weights = sample_weights.view(-1)
    loss = (y_pred - y_true).pow(2).view(-1)
    return (loss * weights).sum() / weights.sum().clamp_min(1e-8)


def train_dvpf_calibration_model(
    model,
    data,
    args,
    device,
    seed,
    model_path=None,
):
    model = model.to(device)
    reconstruction_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    labeled_dataset = TensorDataset(
        to_float_tensor(data.X_labeled_train),
        to_float_tensor(data.y_labeled_train),
        torch.as_tensor(data.labeled_condition_ids, dtype=torch.long),
        to_float_tensor(data.supervision_weights),
    )
    unlabeled_dataset = TensorDataset(to_float_tensor(data.X_target_unlabeled))
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=min(args.batch_size, len(labeled_dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=min(args.batch_size, len(unlabeled_dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    X_valid = to_float_tensor(data.X_target_valid).to(device)
    y_valid = to_float_tensor(data.y_target_valid).to(device)

    history = {
        "train_total": [],
        "supervised_loss": [],
        "labeled_reconstruction": [],
        "target_reconstruction": [],
        "kl_loss": [],
        "contrastive_loss": [],
        "valid_loss": [],
    }
    best_state = copy.deepcopy(model.state_dict())
    best_valid = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        target_iter = cycle(unlabeled_loader)
        total_losses = []
        sup_losses = []
        labeled_rec_losses = []
        target_rec_losses = []
        kl_losses = []
        con_losses = []

        for xb_labeled, yb_labeled, condition_ids, weights in labeled_loader:
            xb_target = next(target_iter)[0]
            xb_labeled = xb_labeled.to(device)
            yb_labeled = yb_labeled.to(device)
            condition_ids = condition_ids.to(device)
            weights = weights.to(device)
            xb_target = xb_target.to(device)

            optimizer.zero_grad()
            pred_labeled, rec_labeled, mu_labeled, logvar_labeled, z_labeled = model(
                xb_labeled,
                return_latent=True,
            )
            _, rec_target, mu_target, logvar_target, _ = model(xb_target, return_latent=True)

            sup = weighted_mse(pred_labeled, yb_labeled, weights)
            labeled_rec = reconstruction_loss(rec_labeled, xb_labeled)
            target_rec = reconstruction_loss(rec_target, xb_target)
            kl = 0.5 * (kl_normal(mu_labeled, logvar_labeled) + kl_normal(mu_target, logvar_target))
            con = supervised_contrastive_loss(z_labeled, condition_ids, temperature=args.temperature)

            total = (
                args.lambda_sup * sup
                + args.lambda_labeled_rec * labeled_rec
                + args.lambda_target_rec * target_rec
                + args.lambda_kl * kl
                + args.lambda_con * con
            )
            total.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_losses.append(total.item())
            sup_losses.append(sup.item())
            labeled_rec_losses.append(labeled_rec.item())
            target_rec_losses.append(target_rec.item())
            kl_losses.append(kl.item())
            con_losses.append(con.item())

        model.eval()
        with torch.no_grad():
            valid_pred = model(X_valid)
            valid_loss = nn.functional.mse_loss(valid_pred, y_valid).item()

        history["train_total"].append(float(np.mean(total_losses)))
        history["supervised_loss"].append(float(np.mean(sup_losses)))
        history["labeled_reconstruction"].append(float(np.mean(labeled_rec_losses)))
        history["target_reconstruction"].append(float(np.mean(target_rec_losses)))
        history["kl_loss"].append(float(np.mean(kl_losses)))
        history["contrastive_loss"].append(float(np.mean(con_losses)))
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid - 1e-8:
            best_valid = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | total={history['train_total'][-1]:.5f} "
                f"| sup={history['supervised_loss'][-1]:.5f} "
                f"| tgt_rec={history['target_reconstruction'][-1]:.5f} "
                f"| con={history['contrastive_loss'][-1]:.5f} | valid={valid_loss:.5f}"
            )
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    if model_path is not None:
        torch.save(best_state, model_path)
    return DVPFTrainResult(model=model, history=history, best_epoch=best_epoch, best_valid_loss=best_valid)


def make_run_dirs(project_root: Path, method, target_condition, args):
    tag = (
        f"seq_{args.seq_len}_hid_{args.hidden_dim}_lat_{args.latent_dim}"
        f"_flow_{args.flow_steps}_align_{args.alignment}_aw_{str(args.alignment_weight).replace('.', 'p')}"
        f"_kl_{str(args.lambda_kl).replace('.', 'p')}"
    )
    root = project_root / "results" / "dvpf_inspired" / method / target_condition / tag
    dirs = {"root": root, "metrics": root / "metrics", "models": root / "models"}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def get_metric_r2(metrics):
    for key, value in metrics.items():
        if key not in {"MSE", "RMSE", "MAE"}:
            return value
    raise KeyError("R2 metric not found.")


def aggregate_summary(summary):
    rows = []
    for method, group in summary.groupby("method", sort=True):
        by_condition = group.groupby("target_condition", as_index=False).agg(
            mean_rmse=("Target RMSE", "mean"),
            mean_mae=("Target MAE", "mean"),
            mean_r2=("Target R2", "mean"),
            worst_rmse=("Target RMSE", "max"),
        )
        worst_row = by_condition.loc[by_condition["worst_rmse"].idxmax()]
        rows.append(
            {
                "method": method,
                "n_targets": int(by_condition["target_condition"].nunique()),
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
        summary.groupby(["method", "target_condition"], as_index=False)
        .agg(
            mean_rmse=("Target RMSE", "mean"),
            mean_mae=("Target MAE", "mean"),
            mean_r2=("Target R2", "mean"),
            worst_target_rmse=("Target RMSE", "max"),
        )
        .sort_values(["method", "target_condition"])
    )


def aggregate_calibration_summary(summary):
    rows = []
    for (method, n_calibration), group in summary.groupby(["method", "n_calibration"], sort=True):
        by_condition = group.groupby("target_condition", as_index=False).agg(
            mean_rmse=("Target RMSE", "mean"),
            mean_mae=("Target MAE", "mean"),
            mean_r2=("Target R2", "mean"),
            worst_rmse=("Target RMSE", "max"),
        )
        worst_row = by_condition.loc[by_condition["worst_rmse"].idxmax()]
        rows.append(
            {
                "method": method,
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


def condition_calibration_summary(summary):
    return (
        summary.groupby(["method", "n_calibration", "target_condition"], as_index=False)
        .agg(
            mean_rmse=("Target RMSE", "mean"),
            std_rmse=("Target RMSE", "std"),
            mean_mae=("Target MAE", "mean"),
            mean_r2=("Target R2", "mean"),
            worst_trial_rmse=("Target RMSE", "max"),
            count_trials_rmse_below_mlp=("rmse_below_mlp", "sum"),
        )
        .sort_values(["method", "n_calibration", "target_condition"])
    )
