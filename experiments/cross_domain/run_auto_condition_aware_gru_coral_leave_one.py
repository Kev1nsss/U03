import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from experiments.cross_domain.common import CONDITION_INTERVALS, condition_names, load_raw_dataframe
from experiments.cross_domain.sequence_common import run_gru_cross_domain_experiment
from unit03_soft_sensor.config import ExperimentConfig


MLP_BASELINE_RMSE = 1.4514
MLP_BASELINE_R2 = 0.9116


def parse_args():
    parser = argparse.ArgumentParser(description="Automatic condition-aware GRU+CORAL leave-one-condition-out.")
    parser.add_argument("--targets", nargs="+", default=condition_names())
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[1, 2, 3, 6])
    parser.add_argument("--seq-len", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--alignment-weight", type=float, default=0.1)
    parser.add_argument("--x-scaler-mode", choices=["source", "source_target"], default="source")
    parser.add_argument("--max-source-per-condition", type=int, default=4000)
    parser.add_argument("--max-target-unlabeled", type=int, default=8000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--tsne-max-per-domain", type=int, default=400)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--summary-prefix", default="auto_condition_aware_gru_coral")
    return parser.parse_args()


def clip_interval(interval, n_rows):
    start = max(0, min(int(interval["start"]), n_rows))
    end = max(start, min(int(interval["end"]), n_rows))
    return start, end


def build_condition_profiles(config):
    df = load_raw_dataframe(config)
    X = df.iloc[:, :-1].values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)

    profiles = {}
    for interval in CONDITION_INTERVALS:
        start, end = clip_interval(interval, len(df))
        values = X_scaled[start:end]
        profiles[interval["name"]] = {
            "mean": values.mean(axis=0),
            "std": values.std(axis=0) + 1e-6,
            "n_samples": len(values),
        }
    return profiles


def condition_distance(target_profile, source_profile):
    mean_dist = np.linalg.norm(target_profile["mean"] - source_profile["mean"])
    std_dist = np.linalg.norm(np.log(target_profile["std"]) - np.log(source_profile["std"]))
    return float(mean_dist + 0.5 * std_dist)


def rank_sources_by_unlabeled_x(profiles, target_condition):
    rows = []
    for source_condition, source_profile in profiles.items():
        if source_condition == target_condition:
            continue
        rows.append(
            {
                "target_condition": target_condition,
                "source_condition": source_condition,
                "x_distribution_distance": condition_distance(profiles[target_condition], source_profile),
            }
        )
    return pd.DataFrame(rows).sort_values("x_distribution_distance")


def aggregate_summary(summary):
    rows = []
    for top_k, group in summary.groupby("top_k", sort=True):
        worst_rmse_idx = group["Target RMSE"].idxmax()
        worst_r2_idx = group["Target R2"].idxmin()
        rows.append(
            {
                "top_k": int(top_k),
                "n_targets": len(group),
                "avg_target_rmse": float(group["Target RMSE"].mean()),
                "avg_target_mae": float(group["Target MAE"].mean()),
                "avg_target_r2": float(group["Target R2"].mean()),
                "worst_rmse_condition": group.loc[worst_rmse_idx, "target_condition"],
                "worst_target_rmse": float(group.loc[worst_rmse_idx, "Target RMSE"]),
                "worst_r2_condition": group.loc[worst_r2_idx, "target_condition"],
                "worst_target_r2": float(group.loc[worst_r2_idx, "Target R2"]),
                "count_rmse_below_mlp": int((group["Target RMSE"] < MLP_BASELINE_RMSE).sum()),
                "count_r2_above_mlp": int((group["Target R2"] > MLP_BASELINE_R2).sum()),
                "count_both_above_mlp": int(
                    ((group["Target RMSE"] < MLP_BASELINE_RMSE) & (group["Target R2"] > MLP_BASELINE_R2)).sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["count_rmse_below_mlp", "avg_target_rmse"], ascending=[False, True])


def main():
    args = parse_args()
    config = ExperimentConfig()
    profiles = build_condition_profiles(config)
    valid_targets = set(condition_names())
    rows = []
    distance_rows = []

    for target in args.targets:
        if target not in valid_targets:
            raise ValueError(f"Unknown target condition: {target}. Choose from: {sorted(valid_targets)}")
        ranked = rank_sources_by_unlabeled_x(profiles, target)
        distance_rows.append(ranked)
        for requested_top_k in args.top_k_values:
            top_k = min(int(requested_top_k), len(ranked))
            sources = ranked.head(top_k)["source_condition"].tolist()
            method_name = f"auto_ca_top{top_k}_gru_coral"
            print("\n" + "#" * 90)
            print(f"Auto condition-aware GRU+CORAL | target={target} | top_k={top_k} | sources={','.join(sources)}")
            print("#" * 90)
            method_args = SimpleNamespace(
                target_condition=target,
                source_conditions=sources,
                x_scaler_mode=args.x_scaler_mode,
                seq_len=args.seq_len,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                alignment_weight=args.alignment_weight,
                max_source_per_condition=args.max_source_per_condition,
                max_target_unlabeled=args.max_target_unlabeled,
                source_valid_ratio=args.source_valid_ratio,
                tsne_max_per_domain=args.tsne_max_per_domain,
                make_plots=args.make_plots,
            )
            row = run_gru_cross_domain_experiment(PROJECT_ROOT, method_name, method_args)
            row["top_k"] = top_k
            row["selected_source_conditions"] = ";".join(sources)
            row["rmse_below_mlp"] = row["Target RMSE"] < MLP_BASELINE_RMSE
            row["r2_above_mlp"] = row["Target R2"] > MLP_BASELINE_R2
            row["both_above_mlp"] = row["rmse_below_mlp"] and row["r2_above_mlp"]
            rows.append(row)

    output_dir = PROJECT_ROOT / "results" / "cross_domain"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows).sort_values(["top_k", "target_condition"])
    aggregate = aggregate_summary(summary)
    distance_table = pd.concat(distance_rows, ignore_index=True)

    summary_path = output_dir / f"{args.summary_prefix}_summary.csv"
    aggregate_path = output_dir / f"{args.summary_prefix}_aggregate.csv"
    distance_path = output_dir / f"{args.summary_prefix}_source_distances.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    distance_table.to_csv(distance_path, index=False, encoding="utf-8-sig")

    print("\nSaved automatic condition-aware summary to:", summary_path)
    print(summary.to_string(index=False))
    print("\nSaved automatic condition-aware aggregate to:", aggregate_path)
    print(aggregate.to_string(index=False))
    print("\nSaved source-distance table to:", distance_path)


if __name__ == "__main__":
    main()
