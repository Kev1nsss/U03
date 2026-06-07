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
    """latent_dim + dropout 二维组合消融实验。

    前两个消融分别只改变一个参数：
    - latent_dim 消融：固定 dropout=0.1
    - dropout 消融：固定 latent_dim=6

    这个脚本会同时搜索两个参数：
    latent_dim = [6, 8, 10, 12]
    dropout = [0.0, 0.05, 0.1, 0.2]

    最终根据验证集 RMSE 选择更可靠的组合。
    """
    base_config = ExperimentConfig()
    set_random_seed(base_config.random_seed)

    output_root = PROJECT_ROOT / "results" / "ablations" / "latent_dropout_grid"
    dirs = ensure_ablation_dirs(output_root)

    data = prepare_data(base_config)
    labeled_mask = make_labeled_mask(len(data.X_train), base_config.labeled_ratio, base_config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for latent_dim in [6, 8, 10, 12]:
        for dropout_rate in [0.0, 0.05, 0.1, 0.2]:
            # 每个 trial 前重置随机种子，让组合之间更可复现。
            set_random_seed(base_config.random_seed)

            trial_config = replace(
                base_config,
                latent_dim=latent_dim,
                aemlp_dropout_rate=dropout_rate,
            )
            trial_name = f"latent_dim_{latent_dim}_dropout_{format_dropout(dropout_rate)}"
            rows.append(run_aemlp_trial(trial_config, data, labeled_mask, device, output_root, trial_name))

    summary = pd.DataFrame(rows).sort_values("Valid RMSE")
    summary_path = dirs["metrics"] / "latent_dropout_grid_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nLatent dim + dropout grid summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()

