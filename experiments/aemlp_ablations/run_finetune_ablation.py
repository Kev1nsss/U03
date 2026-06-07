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

from experiments.aemlp_ablations.common import ensure_ablation_dirs, run_finetune_trial
from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import prepare_data, set_random_seed
from unit03_soft_sensor.train import make_labeled_mask


def main():
    """AE 预训练 + Encoder 微调消融实验。

    目的：
    检查“冻结 Encoder”是否限制了 AE+MLP 的效果。

    做法：
    1. AE 先用 100% X_train 做重构预训练；
    2. 然后用 20% 标签样本训练 Encoder + MLP 回归头；
    3. 用验证集 RMSE 选择较好的 latent_dim。
    """
    base_config = ExperimentConfig()
    set_random_seed(base_config.random_seed)

    output_root = PROJECT_ROOT / "results" / "aemlp_ablations" / "finetune"
    dirs = ensure_ablation_dirs(output_root)

    data = prepare_data(base_config)
    labeled_mask = make_labeled_mask(len(data.X_train), base_config.labeled_ratio, base_config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for latent_dim in [6, 8, 10, 12]:
        # dropout 先用 0.0：少标签回归头在 grid 中显示不一定需要 dropout。
        set_random_seed(base_config.random_seed)
        trial_config = replace(base_config, latent_dim=latent_dim, aemlp_dropout_rate=0.0)
        trial_name = f"finetune_latent_dim_{latent_dim}_dropout_0p0"
        rows.append(run_finetune_trial(trial_config, data, labeled_mask, device, output_root, trial_name))

    summary = pd.DataFrame(rows).sort_values("Valid RMSE")
    summary_path = dirs["metrics"] / "finetune_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nFine-tune ablation summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {summary_path}")


if __name__ == "__main__":
    main()


