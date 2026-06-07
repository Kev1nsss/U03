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

from experiments.cross_domain.common import DEFAULT_TARGET_CONDITION, run_cross_domain_experiment


DEFAULT_WEIGHTS = {
    "coral": [0.1, 1.0, 10.0, 100.0],
    "mmd": [0.001, 0.01, 0.05, 0.1],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Search CORAL/MMD alignment weights on one target condition.")
    parser.add_argument("--target-condition", default=DEFAULT_TARGET_CONDITION)
    parser.add_argument("--method", choices=["coral", "mmd"], default="coral")
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--max-source-per-condition", type=int, default=3000)
    parser.add_argument("--max-target-unlabeled", type=int, default=6000)
    parser.add_argument("--source-valid-ratio", type=float, default=0.2)
    parser.add_argument("--tsne-max-per-domain", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights if args.weights is not None else DEFAULT_WEIGHTS[args.method]
    rows = []

    for weight in weights:
        print("\n" + "#" * 90)
        print(f"Alignment weight search | target={args.target_condition} | method={args.method} | weight={weight}")
        print("#" * 90)
        method_args = SimpleNamespace(
            target_condition=args.target_condition,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            alignment_weight=weight,
            max_source_per_condition=args.max_source_per_condition,
            max_target_unlabeled=args.max_target_unlabeled,
            source_valid_ratio=args.source_valid_ratio,
            tsne_max_per_domain=args.tsne_max_per_domain,
        )
        rows.append(run_cross_domain_experiment(PROJECT_ROOT, args.method, method_args))

    summary = pd.DataFrame(rows).sort_values("Target RMSE")
    output_dir = PROJECT_ROOT / "results" / "cross_domain" / "weight_search" / args.method / args.target_condition
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.csv"
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\nSaved weight-search summary to:", output_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
