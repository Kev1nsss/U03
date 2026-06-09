import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

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
    parser = argparse.ArgumentParser(description="Run tree-based source-only cross-domain baseline.")
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--model", choices=["extra_trees", "random_forest", "hist_gb"], default="extra_trees")
    parser.add_argument("--max-source-per-condition", type=int, default=0)
    parser.add_argument("--max-target-unlabeled", type=int, default=6000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=600)
    return parser


def make_model(args, seed):
    if args.model == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    if args.model == "random_forest":
        return RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    return HistGradientBoostingRegressor(
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=31,
        l2_regularization=1e-4,
        random_state=seed,
    )


def make_output_dirs(model_name, target_condition):
    root = PROJECT_ROOT / "results" / "cross_domain" / "tree_source_only" / target_condition / model_name
    dirs = {"root": root, "metrics": root / "metrics", "figures": root / "figures", "models": root / "models"}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def update_tree_summary(row):
    path = PROJECT_ROOT / "results" / "cross_domain" / "tree_comparison_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        keep = ~((old["model"] == row["model"]) & (old["target_condition"] == row["target_condition"]))
        combined = pd.concat([old[keep], new_row], ignore_index=True)
    else:
        combined = new_row
    combined = combined.sort_values(["Target RMSE", "target_condition", "model"])
    combined.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main():
    args = build_parser().parse_args()
    config = ExperimentConfig()
    set_random_seed(config.random_seed)
    dirs = make_output_dirs(args.model, args.target_condition)

    data = load_cross_domain_data(
        config=config,
        target_condition=args.target_condition,
        max_source_per_condition=args.max_source_per_condition,
        max_target_unlabeled=args.max_target_unlabeled,
        source_valid_ratio=args.source_valid_ratio,
    )
    data.condition_table.to_csv(dirs["metrics"] / "condition_split.csv", index=False, encoding="utf-8-sig")
    save_condition_overview(config, data.target_condition, dirs)

    X_train = data.X_source_train_raw
    y_train = data.y_source_train_raw.reshape(-1)
    X_valid = data.X_source_valid_raw
    y_valid = data.y_source_valid_raw
    X_target = data.X_target_eval_raw
    y_target = data.y_target_eval_raw

    model = make_model(args, config.random_seed)
    print(f"Model: {args.model}")
    print(f"Target condition: {data.target_condition}")
    print(f"Source conditions: {', '.join(data.source_conditions)}")
    print(f"Source train: {X_train.shape}, source valid: {X_valid.shape}, target eval: {X_target.shape}")
    model.fit(X_train, y_train)

    valid_pred = np.asarray(model.predict(X_valid)).reshape(-1, 1)
    target_pred = np.asarray(model.predict(X_target)).reshape(-1, 1)
    valid_metrics = evaluate_regression(y_valid, valid_pred)
    target_metrics = evaluate_regression(y_target, target_pred)

    row = {
        "method": "tree_source_only",
        "model": args.model,
        "target_condition": data.target_condition,
        "source_conditions": ";".join(data.source_conditions),
        "max_source_per_condition": args.max_source_per_condition,
        "n_estimators": args.n_estimators if hasattr(model, "n_estimators") else None,
        "max_depth": args.max_depth,
        "Source Valid RMSE": valid_metrics["RMSE"],
        "Source Valid MAE": valid_metrics["MAE"],
        "Source Valid R2": valid_metrics["R²"],
        "Target RMSE": target_metrics["RMSE"],
        "Target MAE": target_metrics["MAE"],
        "Target R2": target_metrics["R²"],
    }
    pd.DataFrame([row]).to_csv(dirs["metrics"] / "summary.csv", index=False, encoding="utf-8-sig")
    with open(dirs["models"] / f"best_{args.model}.pkl", "wb") as f:
        pickle.dump(model, f)

    plot_prediction_curve(data.target_eval_indices, y_target, target_pred, f"{args.model} target prediction curve", dirs["figures"] / "target_prediction_curve.png")
    plot_true_pred(y_target, target_pred, f"{args.model} target true vs predicted", dirs["figures"] / "target_true_pred_scatter.png")
    plot_residuals(y_target, target_pred, f"{args.model} target residual distribution", dirs["figures"] / "target_residual.png")

    summary_path = update_tree_summary(row)
    print("\nTarget metrics:")
    print(pd.DataFrame([row]).to_string(index=False))
    print(f"\nSaved tree summary to: {dirs['metrics'] / 'summary.csv'}")
    print(f"Updated tree comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
