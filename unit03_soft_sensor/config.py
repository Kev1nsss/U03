from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ExperimentConfig:
    """Shared configuration for the Unit03 baseline experiments."""

    csv_path: str = os.path.join("data", "Unit03_1_5_select.csv")
    fallback_csv_path: str = "Unit03_1_5_select.csv"

    # Baseline outputs are kept under results/baseline so other experiment
    # families can save their own outputs without mixing files together.
    results_dir: str = os.path.join("results", "baseline")
    metrics_dir: str = os.path.join("results", "baseline", "metrics")
    models_dir: str = os.path.join("results", "baseline", "models")
    mlp_figures_dir: str = os.path.join("results", "baseline", "figures", "mlp")
    gmm_figures_dir: str = os.path.join("results", "baseline", "figures", "gmm")
    aemlp_figures_dir: str = os.path.join("results", "baseline", "figures", "ae_mlp")

    n_sample: int = 12000
    random_seed: int = 42

    # 4:3:3 means each cycle of 10 samples is split into train/valid/test.
    # All baseline models share the same split so the comparison is fair.
    split_pattern: tuple = (4, 3, 3)

    # Supervised regressor training settings.
    epochs: int = 80
    batch_size: int = 64
    learning_rate: float = 0.001
    patience: int = 20

    # Low-label setting: only 20% of training labels are used for supervised loss.
    labeled_ratio: float = 0.2

    # MLP baseline architecture and regularization settings.
    mlp_hidden_sizes: tuple = (128, 64)
    mlp_activation_name: str = "relu"
    mlp_dropout_rate: float = 0.1
    supervised_weight_decay: float = 1e-5

    # GMM candidates. Validation RMSE selects the final n_components.
    n_components_list: tuple = (2, 3, 5, 8, 10)

    # AE+MLP settings. AE learns z from X, then the regressor learns z -> y.
    latent_dim: int = 6
    aemlp_activation_name: str = "relu"
    aemlp_dropout_rate: float = 0.1

    ae_epochs: int = 80
    ae_batch_size: int = 64
    ae_learning_rate: float = 0.001
    ae_patience: int = 20

    def ensure_dirs(self):
        """Create output directories before saving metrics, figures, or models."""
        for path in [
            self.results_dir,
            self.metrics_dir,
            self.models_dir,
            self.mlp_figures_dir,
            self.gmm_figures_dir,
            self.aemlp_figures_dir,
        ]:
            os.makedirs(path, exist_ok=True)
