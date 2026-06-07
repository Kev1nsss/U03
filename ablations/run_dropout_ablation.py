import os
import sys
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import torch

from ablations.common import ensure_ablation_dirs, run_aemlp_trial
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import prepare_data, set_random_seed
from unit03_soft_sensor.train import make_labeled_mask


def format_dropout(dropout_rate):
    return str(dropout_rate).replace(".", "p")


def main():
    """AE+MLP regression head 的 Dropout 消融实验。

    目的：
    比较 dropout = 0 / 0.05 / 0.1 / 0.2 对 AE+MLP 的影响。

    为什么要做：
    Dropout 可以降低过拟合，但在少标签回归中太大可能导致欠拟合。
    """
    base_config = ExperimentConfig()
    set_random_seed(base_config.random_seed)

    output_root = PROJECT_ROOT / "results" / "ablations" / "dropout"
    dirs = ensure_ablation_dirs(output_root)

    data = prepare_data(base_config)
    labeled_mask = make_labeled_mask(len(data.X_train), base_config.labeled_ratio, base_config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for dropout_rate in [0.0, 0.05, 0.1, 0.2]:
        trial_config = replace(base_config, aemlp_dropout_rate=dropout_rate)
        trial_name = f"dropout_{format_dropout(dropout_rate)}"
        rows.append(run_aemlp_trial(trial_config, data, labeled_mask, device, output_root, trial_name))

    summary = pd.DataFrame(rows).sort_values("Valid RMSE")
    summary_path = dirs["metrics"] / "dropout_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nDropout ablation summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()

