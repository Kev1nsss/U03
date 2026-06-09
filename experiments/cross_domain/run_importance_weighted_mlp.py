import argparse
import copy
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.cross_domain.common import (
    DEFAULT_TARGET_CONDITION,
    DomainMLPRegressor,
    load_cross_domain_data,
    plot_prediction_curve,
    plot_residuals,
    plot_true_pred,
    save_condition_overview,
)
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import set_random_seed
from unit03_soft_sensor.evaluation import evaluate_regression, inverse_y
from unit03_soft_sensor.plotting import plot_loss_curve
from unit03_soft_sensor.train import to_float_tensor


def build_parser():
    parser = argparse.ArgumentParser(description="Run importance-weighted MLP cross-domain experiment.")
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--max-source-per-condition", type=int, default=3000)
    parser.add_argument("--max-target-unlabeled", type=int, default=6000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--weight-clip", type=float, default=10.0)
    parser.add_argument("--weight-power", type=float, default=1.0)
    return parser


def make_output_dirs(target_condition, weight_clip, weight_power, max_source_per_condition):
    clip_tag = str(weight_clip).replace(".", "p")
    power_tag = str(weight_power).replace(".", "p")
    root = (
        PROJECT_ROOT
        / "results"
        / "cross_domain"
        / "importance_weighted_mlp"
        / target_condition
        / f"clip_{clip_tag}_power_{power_tag}_maxsrc_{max_source_per_condition}"
    )
    dirs = {"root": root, "metrics": root / "metrics", "figures": root / "figures", "models": root / "models"}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def estimate_importance_weights(data, seed, clip_value, power):
    rng = np.random.default_rng(seed)
    n_target = len(data.X_target_unlabeled)
    n_source = len(data.X_source_train)
    source_for_domain = data.X_source_train
    if n_source > n_target:
        source_idx = rng.choice(n_source, size=n_target, replace=False)
        source_for_domain = data.X_source_train[source_idx]

    X_domain = np.vstack([source_for_domain, data.X_target_unlabeled])
    y_domain = np.concatenate([np.zeros(len(source_for_domain)), np.ones(len(data.X_target_unlabeled))])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    clf.fit(X_domain, y_domain)

    def predict_weights(X):
        proba_target = clf.predict_proba(X)[:, 1]
        eps = 1e-4
        odds = proba_target / np.maximum(1.0 - proba_target, eps)
        weights = np.power(odds, power)
        weights = np.clip(weights, 1.0 / clip_value, clip_value)
        weights = weights / np.mean(weights)
        return weights.astype(np.float32)

    return clf, predict_weights(data.X_source_train), predict_weights(data.X_source_valid)


def train_weighted_mlp(model, data, train_weights, valid_weights, args, config, device, model_path):
    model = model.to(device)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=config.supervised_weight_decay)

    train_dataset = TensorDataset(
        to_float_tensor(data.X_source_train),
        to_float_tensor(data.y_source_train),
        to_float_tensor(train_weights.reshape(-1, 1)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.random_seed),
    )
    X_valid = to_float_tensor(data.X_source_valid).to(device)
    y_valid = to_float_tensor(data.y_source_valid).to(device)
    w_valid = to_float_tensor(valid_weights.reshape(-1, 1)).to(device)

    history = {"train_loss": [], "valid_loss": []}
    best_state = copy.deepcopy(model.state_dict())
    best_valid_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, wb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = (criterion(pred, yb) * wb).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            valid_loss = (criterion(model(X_valid), y_valid) * w_valid).mean().item()

        history["train_loss"].append(float(np.mean(train_losses)))
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid_loss - 1e-8:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | weighted_train_loss={history['train_loss'][-1]:.6f} | weighted_valid_loss={valid_loss:.6f}")
        if wait >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch = {best_epoch}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, model_path)
    return model, history, best_epoch, best_valid_loss


def predict_raw(model, X, y_scaler, device):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(to_float_tensor(X)), batch_size=2048, shuffle=False)
    with torch.no_grad():
        for (xb,) in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    return inverse_y(y_scaler, np.vstack(preds))


def update_summary(row):
    path = PROJECT_ROOT / "results" / "cross_domain" / "importance_weighted_mlp_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        keep = ~(
            (old["target_condition"] == row["target_condition"])
            & (old["weight_clip"].astype(float) == float(row["weight_clip"]))
            & (old["weight_power"].astype(float) == float(row["weight_power"]))
            & (old["max_source_per_condition"].astype(int) == int(row["max_source_per_condition"]))
        )
        combined = pd.concat([old[keep], new_row], ignore_index=True)
    else:
        combined = new_row
    combined = combined.sort_values(["Target RMSE", "target_condition"])
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main():
    args = build_parser().parse_args()
    config = ExperimentConfig()
    set_random_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dirs = make_output_dirs(args.target_condition, args.weight_clip, args.weight_power, args.max_source_per_condition)

    data = load_cross_domain_data(
        config=config,
        target_condition=args.target_condition,
        max_source_per_condition=args.max_source_per_condition,
        max_target_unlabeled=args.max_target_unlabeled,
        source_valid_ratio=args.source_valid_ratio,
    )
    data.condition_table.to_csv(dirs["metrics"] / "condition_split.csv", index=False, encoding="utf-8-sig")
    save_condition_overview(config, data.target_condition, dirs)

    domain_clf, train_weights, valid_weights = estimate_importance_weights(
        data=data,
        seed=config.random_seed,
        clip_value=args.weight_clip,
        power=args.weight_power,
    )
    with open(dirs["models"] / "domain_classifier.pkl", "wb") as f:
        pickle.dump(domain_clf, f)
    pd.DataFrame({"train_weight": train_weights}).describe().to_csv(dirs["metrics"] / "train_weight_stats.csv", encoding="utf-8-sig")

    print(f"Using device: {device}")
    print(f"Target condition: {data.target_condition}")
    print(f"Source train: {data.X_source_train.shape}, source valid: {data.X_source_valid.shape}, target eval: {data.X_target_eval.shape}")
    print(f"Importance weights: min={train_weights.min():.4f}, mean={train_weights.mean():.4f}, max={train_weights.max():.4f}")

    model = DomainMLPRegressor(
        input_dim=data.X_source_train.shape[1],
        hidden_sizes=config.mlp_hidden_sizes,
        activation_name=config.mlp_activation_name,
        dropout_rate=config.mlp_dropout_rate,
    )
    model, history, best_epoch, best_valid_loss = train_weighted_mlp(
        model=model,
        data=data,
        train_weights=train_weights,
        valid_weights=valid_weights,
        args=args,
        config=config,
        device=device,
        model_path=dirs["models"] / "best_importance_weighted_mlp.pt",
    )

    source_valid_pred = predict_raw(model, data.X_source_valid, data.y_scaler, device)
    target_pred = predict_raw(model, data.X_target_eval, data.y_scaler, device)
    source_valid_metrics = evaluate_regression(data.y_source_valid_raw, source_valid_pred)
    target_metrics = evaluate_regression(data.y_target_eval_raw, target_pred)

    row = {
        "method": "importance_weighted_mlp",
        "target_condition": data.target_condition,
        "source_conditions": ";".join(data.source_conditions),
        "max_source_per_condition": args.max_source_per_condition,
        "weight_clip": args.weight_clip,
        "weight_power": args.weight_power,
        "best_epoch": best_epoch,
        "best_weighted_source_valid_loss_scaled": best_valid_loss,
        "Source Valid RMSE": source_valid_metrics["RMSE"],
        "Source Valid MAE": source_valid_metrics["MAE"],
        "Source Valid R2": source_valid_metrics["R²"],
        "Target RMSE": target_metrics["RMSE"],
        "Target MAE": target_metrics["MAE"],
        "Target R2": target_metrics["R²"],
    }
    pd.DataFrame([row]).to_csv(dirs["metrics"] / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(dirs["metrics"] / "loss_history.csv", index_label="epoch", encoding="utf-8-sig")

    plot_loss_curve(history, "importance weighted MLP train / valid loss", dirs["figures"] / "loss_curve.png", best_epoch=best_epoch)
    plot_prediction_curve(data.target_eval_indices, data.y_target_eval_raw, target_pred, "importance weighted MLP target prediction curve", dirs["figures"] / "target_prediction_curve.png")
    plot_true_pred(data.y_target_eval_raw, target_pred, "importance weighted MLP target true vs predicted", dirs["figures"] / "target_true_pred_scatter.png")
    plot_residuals(data.y_target_eval_raw, target_pred, "importance weighted MLP target residual distribution", dirs["figures"] / "target_residual.png")

    summary_path = update_summary(row)
    print("\nTarget metrics:")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"\nSaved importance-weighted summary to: {dirs['metrics'] / 'summary.csv'}")
    print(f"Updated importance-weighted comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
