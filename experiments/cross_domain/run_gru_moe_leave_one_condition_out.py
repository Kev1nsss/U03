import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.cross_domain.common import condition_names
from experiments.cross_domain.sequence_common import run_gru_moe_cross_domain_experiment


DEFAULT_METHODS = ["gru_moe", "gru_moe_coral"]
DEFAULT_WEIGHTS = {"gru_moe": 0.0, "gru_moe_coral": 0.1}


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRU-MoE leave-one-condition-out experiments.")
    parser.add_argument("--targets", nargs="+", default=condition_names())
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--alignment-weight", type=float, default=None)
    parser.add_argument("--expert-loss-weight", type=float, default=0.5)
    parser.add_argument("--gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--x-scaler-mode", choices=["source", "source_target"], default="source")
    parser.add_argument("--max-source-per-condition", type=int, default=4000)
    parser.add_argument("--max-target-unlabeled", type=int, default=8000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--tsne-max-per-domain", type=int, default=500)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--summary-name", default="gru_moe_leave_one_condition_summary.csv")
    return parser.parse_args()


def build_aggregate(summary):
    rows = []
    for method, group in summary.groupby("method", sort=False):
        worst_rmse_idx = group["Target RMSE"].idxmax()
        worst_mae_idx = group["Target MAE"].idxmax()
        worst_r2_idx = group["Target R2"].idxmin()
        rows.append(
            {
                "method": method,
                "n_targets": len(group),
                "avg_target_rmse": float(group["Target RMSE"].mean()),
                "avg_target_mae": float(group["Target MAE"].mean()),
                "avg_target_r2": float(group["Target R2"].mean()),
                "worst_rmse_condition": group.loc[worst_rmse_idx, "target_condition"],
                "worst_target_rmse": float(group.loc[worst_rmse_idx, "Target RMSE"]),
                "worst_mae_condition": group.loc[worst_mae_idx, "target_condition"],
                "worst_target_mae": float(group.loc[worst_mae_idx, "Target MAE"]),
                "worst_r2_condition": group.loc[worst_r2_idx, "target_condition"],
                "worst_target_r2": float(group.loc[worst_r2_idx, "Target R2"]),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_target_rmse")


def main():
    args = parse_args()
    valid_targets = set(condition_names())
    rows = []

    for target in args.targets:
        if target not in valid_targets:
            raise ValueError(f"Unknown target condition: {target}. Choose from: {sorted(valid_targets)}")
        for method in args.methods:
            alignment_weight = DEFAULT_WEIGHTS[method] if args.alignment_weight is None else args.alignment_weight
            print("\n" + "#" * 90)
            print(f"GRU-MoE leave-one-condition-out | target={target} | method={method}")
            print("#" * 90)
            method_args = SimpleNamespace(
                target_condition=target,
                source_conditions=None,
                x_scaler_mode=args.x_scaler_mode,
                seq_len=args.seq_len,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                alignment_weight=alignment_weight,
                expert_loss_weight=args.expert_loss_weight,
                gate_loss_weight=args.gate_loss_weight,
                gate_temperature=args.gate_temperature,
                max_source_per_condition=args.max_source_per_condition,
                max_target_unlabeled=args.max_target_unlabeled,
                source_valid_ratio=args.source_valid_ratio,
                tsne_max_per_domain=args.tsne_max_per_domain,
                make_plots=args.make_plots,
            )
            rows.append(run_gru_moe_cross_domain_experiment(PROJECT_ROOT, method, method_args))

    summary = pd.DataFrame(rows).sort_values(["method", "target_condition"])
    output_dir = PROJECT_ROOT / "results" / "cross_domain"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / args.summary_name
    aggregate_path = output_dir / args.summary_name.replace("_summary.csv", "_aggregate.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    aggregate = build_aggregate(summary)
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")

    print("\nSaved GRU-MoE leave-one-condition summary to:", summary_path)
    print(summary.to_string(index=False))
    print("\nSaved GRU-MoE aggregate summary to:", aggregate_path)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
