import os
import sys
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import torch

from experiments.aemlp_ablations.common import ensure_ablation_dirs, run_aemlp_trial
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import prepare_data, set_random_seed
from unit03_soft_sensor.train import make_labeled_mask


def main():
    """AE latent_dim 消融实验。

    目的：
    比较 latent_dim = 6 / 8 / 10 / 12 时，AE+MLP 的验证集和测试集表现。

    为什么要做：
    latent_dim 太小可能丢失对 y 有用的信息；latent_dim 太大则可能压缩效果不明显，
    所以需要用验证集 RMSE 选择更合适的隐变量维度。
    """
    base_config = ExperimentConfig()
    set_random_seed(base_config.random_seed)

    output_root = PROJECT_ROOT / "results" / "aemlp_ablations" / "latent_dim"
    dirs = ensure_ablation_dirs(output_root)

    data = prepare_data(base_config)
    labeled_mask = make_labeled_mask(len(data.X_train), base_config.labeled_ratio, base_config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for latent_dim in [6, 8, 10, 12]:
        trial_config = replace(base_config, latent_dim=latent_dim)
        trial_name = f"latent_dim_{latent_dim}"
        rows.append(run_aemlp_trial(trial_config, data, labeled_mask, device, output_root, trial_name))

    summary = pd.DataFrame(rows).sort_values("Valid RMSE")
    summary_path = dirs["metrics"] / "latent_dim_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nLatent dim ablation summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()




