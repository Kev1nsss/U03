import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.cross_domain.common import (
    DEFAULT_TARGET_CONDITION,
    load_cross_domain_data,
    plot_prediction_curve,
    plot_residuals,
    plot_true_pred,
    save_condition_overview,
)
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import set_random_seed
from unit03_soft_sensor.evaluation import evaluate_regression


def build_parser():
    parser = argparse.ArgumentParser(description="Run KNN source-only cross-domain baseline.")
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--max-source-per-condition", type=int, default=0)
    parser.add_argument("--max-target-unlabeled", type=int, default=6000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--weights", choices=["uniform", "distance"], default="distance")
    return parser


def make_output_dirs(target_condition, n_neighbors, weights, max_source_per_condition):
    root = (
        PROJECT_ROOT
        / "results"
        / "cross_domain"
        / "knn_source_only"
        / target_condition
        / f"k_{n_neighbors}_weights_{weights}_maxsrc_{max_source_per_condition}"
    )
    dirs = {"root": root, "metrics": root / "metrics", "figures": root / "figures", "models": root / "models"}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def update_summary(row):
    path = PROJECT_ROOT / "results" / "cross_domain" / "knn_comparison_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        keep = ~(
            (old["target_condition"] == row["target_condition"])
            & (old["n_neighbors"].astype(int) == int(row["n_neighbors"]))
            & (old["weights"] == row["weights"])
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
    dirs = make_output_dirs(args.target_condition, args.n_neighbors, args.weights, args.max_source_per_condition)

    data = load_cross_domain_data(
        config=config,
        target_condition=args.target_condition,
        max_source_per_condition=args.max_source_per_condition,
        max_target_unlabeled=args.max_target_unlabeled,
        source_valid_ratio=args.source_valid_ratio,
    )
    data.condition_table.to_csv(dirs["metrics"] / "condition_split.csv", index=False, encoding="utf-8-sig")
    save_condition_overview(config, data.target_condition, dirs)

    model = KNeighborsRegressor(n_neighbors=args.n_neighbors, weights=args.weights, algorithm="auto", n_jobs=-1)
    print(f"Target condition: {data.target_condition}")
    print(f"KNN: k={args.n_neighbors}, weights={args.weights}")
    print(f"Source train: {data.X_source_train.shape}, source valid: {data.X_source_valid.shape}, target eval: {data.X_target_eval.shape}")
    model.fit(data.X_source_train, data.y_source_train_raw.reshape(-1))

    valid_pred = np.asarray(model.predict(data.X_source_valid)).reshape(-1, 1)
    target_pred = np.asarray(model.predict(data.X_target_eval)).reshape(-1, 1)
    valid_metrics = evaluate_regression(data.y_source_valid_raw, valid_pred)
    target_metrics = evaluate_regression(data.y_target_eval_raw, target_pred)

    row = {
        "method": "knn_source_only",
        "target_condition": data.target_condition,
        "source_conditions": ";".join(data.source_conditions),
        "max_source_per_condition": args.max_source_per_condition,
        "n_neighbors": args.n_neighbors,
        "weights": args.weights,
        "Source Valid RMSE": valid_metrics["RMSE"],
        "Source Valid MAE": valid_metrics["MAE"],
        "Source Valid R2": valid_metrics["R²"],
        "Target RMSE": target_metrics["RMSE"],
        "Target MAE": target_metrics["MAE"],
        "Target R2": target_metrics["R²"],
    }
    pd.DataFrame([row]).to_csv(dirs["metrics"] / "summary.csv", index=False, encoding="utf-8-sig")
    with open(dirs["models"] / "knn_source_only.pkl", "wb") as f:
        pickle.dump(model, f)

    plot_prediction_curve(data.target_eval_indices, data.y_target_eval_raw, target_pred, "KNN target prediction curve", dirs["figures"] / "target_prediction_curve.png")
    plot_true_pred(data.y_target_eval_raw, target_pred, "KNN target true vs predicted", dirs["figures"] / "target_true_pred_scatter.png")
    plot_residuals(data.y_target_eval_raw, target_pred, "KNN target residual distribution", dirs["figures"] / "target_residual.png")

    summary_path = update_summary(row)
    print("\nTarget metrics:")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"\nSaved KNN summary to: {dirs['metrics'] / 'summary.csv'}")
    print(f"Updated KNN comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
