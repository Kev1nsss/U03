import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.cross_domain.common import condition_names
from experiments.cross_domain.sequence_common import load_sequence_cross_domain_data
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.evaluation import evaluate_regression, inverse_y
from unit03_soft_sensor.models import MLPRegressor
from unit03_soft_sensor.train import to_float_tensor


MLP_BASELINE_RMSE = 1.4514
MLP_BASELINE_R2 = 0.9116


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot neural target adaptation under leave-one-condition-out.")
    parser.add_argument("--targets", nargs="+", default=condition_names())
    parser.add_argument("--modes", nargs="+", default=["target_only", "source_finetune"], choices=["target_only", "source_finetune"])
    parser.add_argument("--n-calibration-list", nargs="+", type=int, default=[100, 200])
    parser.add_argument("--n-trials", type=int, default=3)
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pretrain-epochs", type=int, default=80)
    parser.add_argument("--finetune-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--finetune-learning-rate", type=float, default=0.0005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--max-source-per-condition", type=int, default=5000)
    parser.add_argument("--max-target-unlabeled", type=int, default=8000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--summary-prefix", default="few_shot_neural_adaptation")
    return parser.parse_args()


def get_r2(metrics):
    return metrics[[key for key in metrics if key not in {"MSE", "RMSE", "MAE"}][0]]


def choose_calibration_indices(n_samples, n_calibration, seed):
    n_calibration = min(int(n_calibration), n_samples - 1)
    if n_calibration <= 1:
        raise ValueError("n_calibration must leave calibration and evaluation samples.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, size=n_calibration, replace=False))


def split_calibration(calibration_indices, valid_ratio, seed):
    rng = np.random.default_rng(seed)
    shuffled = np.array(calibration_indices, copy=True)
    rng.shuffle(shuffled)
    n_valid = max(1, int(round(len(shuffled) * valid_ratio)))
    valid_idx = np.sort(shuffled[:n_valid])
    train_idx = np.sort(shuffled[n_valid:])
    if len(train_idx) == 0:
        train_idx, valid_idx = valid_idx, train_idx
    return train_idx, valid_idx


def train_mlp(model, X_train, y_train, X_valid, y_valid, args, device, learning_rate, epochs, seed):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(to_float_tensor(X_train), to_float_tensor(y_train)),
        batch_size=min(args.batch_size, len(X_train)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    X_valid_tensor = to_float_tensor(X_valid).to(device)
    y_valid_tensor = to_float_tensor(y_valid).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_valid = float("inf")
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = criterion(model(X_valid_tensor), y_valid_tensor).item()
        if valid_loss < best_valid - 1e-8:
            best_valid = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        if wait >= args.patience:
            break

    model.load_state_dict(best_state)
    return model, best_valid


def predict_scaled(model, X, device):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(to_float_tensor(X)), batch_size=2048, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return np.vstack(preds)


def aggregate_summary(summary):
    rows = []
    group_cols = ["mode", "n_calibration"]
    for keys, group in summary.groupby(group_cols, sort=True):
        mode, n_calibration = keys
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


def main():
    args = parse_args()
    config = ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for target in args.targets:
        data = load_sequence_cross_domain_data(
            config=config,
            target_condition=target,
            seq_len=1,
            max_source_per_condition=args.max_source_per_condition,
            max_target_unlabeled=args.max_target_unlabeled,
            source_valid_ratio=args.source_valid_ratio,
            source_condition_names=None,
            x_scaler_mode="source",
        )
        X_source_train = data.X_source_train[:, -1, :]
        X_source_valid = data.X_source_valid[:, -1, :]
        X_target = data.X_target_eval[:, -1, :]
        y_target_raw = data.y_target_eval_raw.reshape(-1, 1)

        pretrained_state = None
        if "source_finetune" in args.modes:
            source_model = MLPRegressor(
                input_dim=X_source_train.shape[1],
                hidden_sizes=tuple(args.hidden_sizes),
                activation_name="relu",
                dropout_rate=args.dropout,
            )
            source_model, _ = train_mlp(
                source_model,
                X_source_train,
                data.y_source_train,
                X_source_valid,
                data.y_source_valid,
                args,
                device,
                args.learning_rate,
                args.pretrain_epochs,
                config.random_seed,
            )
            pretrained_state = copy.deepcopy(source_model.state_dict())

        for n_calibration in args.n_calibration_list:
            for trial in range(args.n_trials):
                seed = config.random_seed + 1009 * trial + 17 * int(n_calibration)
                cal_idx = choose_calibration_indices(len(X_target), n_calibration, seed)
                ft_train_idx, ft_valid_idx = split_calibration(cal_idx, args.valid_ratio, seed + 1)
                eval_mask = np.ones(len(X_target), dtype=bool)
                eval_mask[cal_idx] = False

                target_y_scaler = StandardScaler()
                y_ft_train = target_y_scaler.fit_transform(y_target_raw[ft_train_idx]).astype(np.float32)
                y_ft_valid = target_y_scaler.transform(y_target_raw[ft_valid_idx]).astype(np.float32)

                for mode in args.modes:
                    print(f"target={target} | mode={mode} | n_cal={n_calibration} | trial={trial + 1}/{args.n_trials}")
                    if mode == "target_only":
                        target_x_scaler = StandardScaler()
                        X_train_mode = target_x_scaler.fit_transform(X_target[ft_train_idx]).astype(np.float32)
                        X_valid_mode = target_x_scaler.transform(X_target[ft_valid_idx]).astype(np.float32)
                        X_eval_mode = target_x_scaler.transform(X_target[eval_mask]).astype(np.float32)
                    else:
                        X_train_mode = X_target[ft_train_idx]
                        X_valid_mode = X_target[ft_valid_idx]
                        X_eval_mode = X_target[eval_mask]

                    model = MLPRegressor(
                        input_dim=X_target.shape[1],
                        hidden_sizes=tuple(args.hidden_sizes),
                        activation_name="relu",
                        dropout_rate=args.dropout,
                    )
                    if mode == "source_finetune":
                        model.load_state_dict(pretrained_state)
                    model, best_valid = train_mlp(
                        model,
                        X_train_mode,
                        y_ft_train,
                        X_valid_mode,
                        y_ft_valid,
                        args,
                        device,
                        args.finetune_learning_rate if mode == "source_finetune" else args.learning_rate,
                        args.finetune_epochs,
                        seed,
                    )
                    pred_scaled = predict_scaled(model, X_eval_mode, device)
                    pred = inverse_y(target_y_scaler, pred_scaled)
                    metrics = evaluate_regression(y_target_raw[eval_mask], pred)
                    target_r2 = get_r2(metrics)
                    rows.append(
                        {
                            "method": "few_shot_neural_adaptation",
                            "mode": mode,
                            "target_condition": target,
                            "n_calibration": int(n_calibration),
                            "trial": int(trial),
                            "best_valid_loss_scaled": best_valid,
                            "Target RMSE": metrics["RMSE"],
                            "Target MAE": metrics["MAE"],
                            "Target R2": target_r2,
                            "rmse_below_mlp": bool(metrics["RMSE"] < MLP_BASELINE_RMSE),
                            "r2_above_mlp": bool(target_r2 > MLP_BASELINE_R2),
                            "both_above_mlp": bool(metrics["RMSE"] < MLP_BASELINE_RMSE and target_r2 > MLP_BASELINE_R2),
                        }
                    )

    output_dir = PROJECT_ROOT / "results" / "cross_domain"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values(["mode", "n_calibration", "target_condition", "trial"])
    aggregate = aggregate_summary(summary)
    per_condition = condition_summary(summary)
    summary_path = output_dir / f"{args.summary_prefix}_summary.csv"
    aggregate_path = output_dir / f"{args.summary_prefix}_aggregate.csv"
    condition_path = output_dir / f"{args.summary_prefix}_by_condition.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    per_condition.to_csv(condition_path, index=False, encoding="utf-8-sig")

    print("\nSaved few-shot neural adaptation summary to:", summary_path)
    print("\nSaved aggregate to:", aggregate_path)
    print(aggregate.to_string(index=False))
    print("\nSaved per-condition summary to:", condition_path)
    print(per_condition.to_string(index=False))


if __name__ == "__main__":
    main()
