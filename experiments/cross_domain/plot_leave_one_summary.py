import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def main():
    summary_path = PROJECT_ROOT / "results" / "cross_domain" / "leave_one_condition_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    df = pd.read_csv(summary_path)
    order = [
        "startup_ramp",
        "early_stable",
        "restart_transition",
        "long_stable",
        "late_disturbance",
        "late_stable",
    ]
    methods = [method for method in ["source_only", "coral", "mmd"] if method in set(df["method"])]
    colors = {"source_only": "#64748b", "coral": "#0ea5e9", "mmd": "#f59e0b"}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    width = 0.8 / max(1, len(methods))
    x = list(range(len(order)))

    for i, method in enumerate(methods):
        method_df = df[df["method"] == method].set_index("target_condition")
        offsets = [v + (i - (len(methods) - 1) / 2) * width for v in x]
        rmse = [method_df.loc[target, "Target RMSE"] if target in method_df.index else None for target in order]
        r2 = [method_df.loc[target, "Target R2"] if target in method_df.index else None for target in order]
        axes[0].bar(offsets, rmse, width=width, label=method, color=colors.get(method, None), alpha=0.85)
        axes[1].bar(offsets, r2, width=width, label=method, color=colors.get(method, None), alpha=0.85)

    axes[0].axhline(1.4514, color="crimson", linestyle="--", linewidth=1.2, label="same-split MLP RMSE 1.4514")
    axes[0].set_ylabel("Target RMSE")
    axes[0].set_title("Leave-one-condition-out Target RMSE")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=3)

    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Target R2")
    axes[1].set_title("Leave-one-condition-out Target R2")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=20, ha="right")
    axes[1].legend(ncol=3)

    output_dir = PROJECT_ROOT / "results" / "cross_domain" / "summary_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "leave_one_condition_summary.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()
    print("Saved summary figure to:", output_path)


if __name__ == "__main__":
    main()
