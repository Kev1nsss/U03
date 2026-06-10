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

from experiments.cross_domain.common import condition_names
from experiments.cross_domain.sequence_common import load_sequence_cross_domain_data
from experiments.dvpf_inspired.dvpf_common import (
    GRUVAEFlowRegressor,
    MLP_BASELINE_R2,
    MLP_BASELINE_RMSE,
    aggregate_summary,
    condition_summary,
    evaluate_dvpf_model,
    get_metric_r2,
    make_run_dirs,
    train_dvpf_model,
)
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run DVPF-inspired GRU-VAE-flow leave-one-condition-out experiment.")
    parser.add_argument("--targets", nargs="+", default=condition_names())
    parser.add_argument("--seq-len", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--flow-steps", type=int, default=1)
    parser.add_argument("--flow-hidden-dim", type=int, default=64)
    parser.add_argument("--flow-step-size", type=float, default=0.2)
    parser.add_argument("--alignment", choices=["none", "coral", "mmd"], default="none")
    parser.add_argument("--alignment-weight", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.00001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--lambda-source-rec", type=float, default=0.05)
    parser.add_argument("--lambda-target-rec", type=float, default=0.2)
    parser.add_argument("--lambda-kl", type=float, default=0.001)

    parser.add_argument("--max-source-per-condition", type=int, default=3000)
    parser.add_argument("--max-target-unlabeled", type=int, default=6000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--x-scaler-mode", choices=["source", "source_target"], default="source_target")

    parser.add_argument("--summary-prefix", default="gru_vae_flow_leave_one")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = ExperimentConfig()
    set_random_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = PROJECT_ROOT / "results" / "dvpf_inspired"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    method = "gru_vae_flow"

    print(f"Using device: {device}")
    print(f"Targets: {', '.join(args.targets)}")
    print(f"Reference MLP baseline: RMSE={MLP_BASELINE_RMSE}, R2={MLP_BASELINE_R2}")
    print("Target labels are not used for training; target y is used only for evaluation.")

    for target in args.targets:
        print(f"\n[target={target}] method={method}")
        data = load_sequence_cross_domain_data(
            config=config,
            target_condition=target,
            seq_len=args.seq_len,
            max_source_per_condition=args.max_source_per_condition,
            max_target_unlabeled=args.max_target_unlabeled,
            source_valid_ratio=args.source_valid_ratio,
            source_condition_names=None,
            x_scaler_mode=args.x_scaler_mode,
        )
        dirs = make_run_dirs(PROJECT_ROOT, method, target, args)
        model_path = dirs["models"] / f"best_{method}.pt" if args.save_models else None
        model = GRUVAEFlowRegressor(
            input_dim=data.X_source_train.shape[-1],
            seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            num_layers=args.num_layers,
            dropout_rate=args.dropout,
            flow_steps=args.flow_steps,
            flow_hidden_dim=args.flow_hidden_dim,
            flow_step_size=args.flow_step_size,
        )

        result = train_dvpf_model(
            model=model,
            data=data,
            args=args,
            device=device,
            seed=config.random_seed,
            model_path=model_path,
        )

        _, source_valid_metrics = evaluate_dvpf_model(
            result.model,
            data.X_source_valid,
            data.y_source_valid_raw,
            data.y_scaler,
            device,
        )
        target_pred, target_metrics = evaluate_dvpf_model(
            result.model,
            data.X_target_eval,
            data.y_target_eval_raw,
            data.y_scaler,
            device,
        )
        target_r2 = get_metric_r2(target_metrics)
        source_r2 = get_metric_r2(source_valid_metrics)
        row = {
            "method": method,
            "target_condition": target,
            "source_conditions": ";".join(data.source_conditions),
            "seq_len": args.seq_len,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "flow_steps": args.flow_steps,
            "alignment": args.alignment,
            "alignment_weight": args.alignment_weight,
            "lambda_source_rec": args.lambda_source_rec,
            "lambda_target_rec": args.lambda_target_rec,
            "lambda_kl": args.lambda_kl,
            "best_epoch": int(result.best_epoch),
            "best_source_valid_loss_scaled": float(result.best_valid_loss),
            "Source Valid RMSE": source_valid_metrics["RMSE"],
            "Source Valid MAE": source_valid_metrics["MAE"],
            "Source Valid R2": source_r2,
            "Target RMSE": target_metrics["RMSE"],
            "Target MAE": target_metrics["MAE"],
            "Target R2": target_r2,
            "rmse_below_mlp": bool(target_metrics["RMSE"] < MLP_BASELINE_RMSE),
            "r2_above_mlp": bool(target_r2 > MLP_BASELINE_R2),
        }
        rows.append(row)

        pd.DataFrame([row]).to_csv(dirs["metrics"] / "summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(result.history).to_csv(dirs["metrics"] / "loss_history.csv", index_label="epoch", encoding="utf-8-sig")
        data.condition_table.to_csv(dirs["metrics"] / "condition_split.csv", index=False, encoding="utf-8-sig")
        if args.save_predictions:
            pd.DataFrame(
                {
                    "target_eval_index": data.target_eval_indices,
                    "y_true": data.y_target_eval_raw.reshape(-1),
                    "y_pred": target_pred.reshape(-1),
                }
            ).to_csv(dirs["metrics"] / "target_predictions.csv", index=False, encoding="utf-8-sig")

        print(
            f"Target RMSE={target_metrics['RMSE']:.4f}, MAE={target_metrics['MAE']:.4f}, "
            f"R2={target_r2:.4f}, best_epoch={result.best_epoch}"
        )

    summary = pd.DataFrame(rows).sort_values(["method", "target_condition"])
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
