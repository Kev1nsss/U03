import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.aecl_cross_condition.aecl_common import (
    MLP_BASELINE_R2,
    MLP_BASELINE_RMSE,
    aggregate_summary,
    condition_summary,
    evaluate_target,
    get_metric_r2,
    load_aecl_condition_data,
    make_run_dirs,
    train_aecl,
    train_mlp_calibration,
)
from experiments.cross_domain.common import condition_names
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run AECL leave-one-condition-out adaptation.")
    parser.add_argument("--targets", nargs="+", default=condition_names())
    parser.add_argument("--modes", nargs="+", default=["mlp", "ae_mlp", "aecl"], choices=["mlp", "ae_mlp", "aecl"])
    parser.add_argument("--n-calibration-list", nargs="+", type=int, default=[1000])
    parser.add_argument("--n-trials", type=int, default=3)

    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--encoder-hidden-sizes", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--regressor-hidden-sizes", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--mlp-hidden-sizes", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--mlp-dropout", type=float, default=0.05)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--valid-ratio", type=float, default=0.2)

    parser.add_argument("--lambda-rec", type=float, default=0.2)
    parser.add_argument("--lambda-sup", type=float, default=1.0)
    parser.add_argument("--lambda-con", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--contrast-labeled-only", action="store_true")
    parser.add_argument("--target-supervision-weight", type=float, default=10.0)
    parser.add_argument("--source-supervision-weight", type=float, default=0.1)

    parser.add_argument("--max-source-per-condition", type=int, default=3000)
    parser.add_argument("--max-target-unlabeled", type=int, default=6000)
    parser.add_argument(
        "--x-scaler-mode",
        choices=["target", "source", "target_unlabeled", "source_target"],
        default="target_unlabeled",
    )
    parser.add_argument("--ae-mlp-x-scaler-mode", choices=["target", "target_unlabeled"], default="target_unlabeled")
    parser.add_argument("--mlp-x-scaler-mode", choices=["target", "target_unlabeled"], default="target")
    parser.add_argument("--y-scaler-mode", choices=["target", "labeled"], default="target")

    parser.add_argument("--summary-prefix", default="aecl_leave_one")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def mode_settings(mode, args):
    if mode == "mlp":
        return {
            "include_source": False,
            "x_scaler_mode": args.mlp_x_scaler_mode,
            "lambda_con": 0.0,
        }
    if mode == "ae_mlp":
        return {
            "include_source": False,
            "x_scaler_mode": args.ae_mlp_x_scaler_mode,
            "lambda_con": 0.0,
        }
    if mode == "aecl":
        return {
            "include_source": True,
            "x_scaler_mode": args.x_scaler_mode,
            "lambda_con": args.lambda_con,
        }
    raise ValueError(f"Unknown mode={mode}")


def main():
    args = parse_args()
    config = ExperimentConfig()
    set_random_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = PROJECT_ROOT / "results" / "aecl_cross_condition"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"Using device: {device}")
    print(f"Targets: {', '.join(args.targets)}")
    print(f"Modes: {', '.join(args.modes)}")
    print(f"Reference MLP baseline: RMSE={MLP_BASELINE_RMSE}, R2={MLP_BASELINE_R2}")

    for target_idx, target in enumerate(args.targets):
        for n_calibration in args.n_calibration_list:
            for trial in range(args.n_trials):
                base_seed = config.random_seed + 1009 * trial + 17 * int(n_calibration) + 131 * target_idx
                for mode_idx, mode in enumerate(args.modes):
                    settings = mode_settings(mode, args)
                    seed = base_seed + 31 * mode_idx
                    print(
                        f"\n[target={target}] mode={mode} | n_cal={n_calibration} "
                        f"| trial={trial + 1}/{args.n_trials}"
                    )
                    data = load_aecl_condition_data(
                        config=config,
                        target_condition=target,
                        n_calibration=n_calibration,
                        seed=seed,
                        valid_ratio=args.valid_ratio,
                        max_source_per_condition=args.max_source_per_condition,
                        max_target_unlabeled=args.max_target_unlabeled,
                        x_scaler_mode=settings["x_scaler_mode"],
                        y_scaler_mode=args.y_scaler_mode,
                        include_source=settings["include_source"],
                        target_supervision_weight=args.target_supervision_weight,
                        source_supervision_weight=args.source_supervision_weight,
                    )
                    dirs = make_run_dirs(PROJECT_ROOT, mode, target, n_calibration, trial, args)
                    model_path = dirs["models"] / f"best_{mode}.pt" if args.save_models else None

                    if mode == "mlp":
                        model, history, best_epoch, best_valid = train_mlp_calibration(
                            data=data,
                            args=args,
                            device=device,
                            seed=seed,
                            model_path=model_path,
                        )
                    else:
                        model, history, best_epoch, best_valid = train_aecl(
                            data=data,
                            args=args,
                            device=device,
                            seed=seed,
                            lambda_con=settings["lambda_con"],
                            model_path=model_path,
                        )

                    pred, metrics = evaluate_target(model, data, device)
                    target_r2 = get_metric_r2(metrics)
                    row = {
                        "method": "aecl_cross_condition",
                        "mode": mode,
                        "target_condition": target,
                        "n_calibration": int(n_calibration),
                        "trial": int(trial),
                        "best_epoch": int(best_epoch),
                        "best_valid_loss_scaled": float(best_valid),
                        "x_scaler_mode": settings["x_scaler_mode"],
                        "y_scaler_mode": args.y_scaler_mode,
                        "include_source": bool(settings["include_source"]),
                        "lambda_rec": float(args.lambda_rec),
                        "lambda_sup": float(args.lambda_sup),
                        "lambda_con": float(settings["lambda_con"]),
                        "Target RMSE": metrics["RMSE"],
                        "Target MAE": metrics["MAE"],
                        "Target R2": target_r2,
                        "rmse_below_mlp": bool(metrics["RMSE"] < MLP_BASELINE_RMSE),
                        "r2_above_mlp": bool(target_r2 > MLP_BASELINE_R2),
                        "both_above_mlp": bool(metrics["RMSE"] < MLP_BASELINE_RMSE and target_r2 > MLP_BASELINE_R2),
                    }
                    rows.append(row)

                    pd.DataFrame([row]).to_csv(dirs["metrics"] / "summary.csv", index=False, encoding="utf-8-sig")
                    pd.DataFrame(history).to_csv(dirs["metrics"] / "loss_history.csv", index_label="epoch", encoding="utf-8-sig")
                    data.condition_table.to_csv(dirs["metrics"] / "condition_split.csv", index=False, encoding="utf-8-sig")
                    if args.save_predictions:
                        pd.DataFrame(
                            {
                                "target_eval_index": data.target_eval_indices,
                                "y_true": data.y_target_eval_raw.reshape(-1),
                                "y_pred": pred.reshape(-1),
                            }
                        ).to_csv(dirs["metrics"] / "target_predictions.csv", index=False, encoding="utf-8-sig")

                    print(
                        f"Target RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
                        f"R2={target_r2:.4f}, best_epoch={best_epoch}"
                    )

    summary = pd.DataFrame(rows).sort_values(["mode", "n_calibration", "target_condition", "trial"])
    aggregate = aggregate_summary(summary)
    by_condition = condition_summary(summary)
    summary_path = output_root / f"{args.summary_prefix}_summary.csv"
    aggregate_path = output_root / f"{args.summary_prefix}_aggregate.csv"
    condition_path = output_root / f"{args.summary_prefix}_by_condition.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    by_condition.to_csv(condition_path, index=False, encoding="utf-8-sig")

    print("\nSaved summary to:", summary_path)
    print("Saved aggregate to:", aggregate_path)
    print(aggregate.to_string(index=False))
    print("\nSaved per-condition summary to:", condition_path)
    print(by_condition.to_string(index=False))


if __name__ == "__main__":
    main()
