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

from experiments.cross_domain.common import condition_names, run_cross_domain_experiment


DEFAULT_TARGETS = [
    "startup_ramp",
    "early_stable",
    "restart_transition",
    "long_stable",
    "late_disturbance",
    "late_stable",
]
DEFAULT_METHODS = ["source_only", "coral"]
DEFAULT_WEIGHTS = {"source_only": 0.0, "coral": 1.0, "mmd": 0.05}


def parse_args():
    parser = argparse.ArgumentParser(description="Run leave-one-condition-out cross-domain experiments.")
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=["source_only", "coral", "mmd"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-source-per-condition", type=int, default=3000)
    parser.add_argument("--max-target-unlabeled", type=int, default=6000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--tsne-max-per-domain", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    valid_targets = set(condition_names())
    rows = []

    for target in args.targets:
        if target not in valid_targets:
            raise ValueError(f"Unknown target condition: {target}. Choose from: {sorted(valid_targets)}")
        for method in args.methods:
            print("\n" + "#" * 90)
            print(f"Leave-one-condition-out | target={target} | method={method}")
            print("#" * 90)
            method_args = SimpleNamespace(
                target_condition=target,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                patience=args.patience,
                alignment_weight=DEFAULT_WEIGHTS[method],
                max_source_per_condition=args.max_source_per_condition,
                max_target_unlabeled=args.max_target_unlabeled,
                source_valid_ratio=args.source_valid_ratio,
                tsne_max_per_domain=args.tsne_max_per_domain,
            )
            rows.append(run_cross_domain_experiment(PROJECT_ROOT, method, method_args))

    summary = pd.DataFrame(rows).sort_values(["target_condition", "Target RMSE", "method"])
    output_path = PROJECT_ROOT / "results" / "cross_domain" / "leave_one_condition_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\nSaved leave-one-condition summary to:", output_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
