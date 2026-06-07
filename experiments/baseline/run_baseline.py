import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import torch

from unit03_soft_sensor.config import ExperimentConfig
from unit03_soft_sensor.data import prepare_data, set_random_seed
from unit03_soft_sensor.evaluation import evaluate_regression, inverse_y, print_metrics, print_metrics_table
from unit03_soft_sensor.models import AEMLPRegressor, AutoEncoder, GMMMeanRegressor, MLPRegressor
from unit03_soft_sensor.plotting import plot_loss_curve, plot_residuals, plot_true_pred
from unit03_soft_sensor.train import (
    extract_encoder_features,
    make_labeled_mask,
    predict_torch_model,
    train_autoencoder,
    train_supervised_model,
)


def configure_stdout():
    """Use UTF-8 output on Windows so Chinese text and R² display correctly."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def print_data_report(data):
    """Print split shapes and sampled index range for a quick sanity check."""
    print(f"Data after dropna: {data.df_shape}")
    print(f"Train: X={data.X_train.shape}, y={data.y_train.shape}")
    print(f"Valid: X={data.X_valid.shape}, y={data.y_valid.shape}")
    print(f"Test : X={data.X_test.shape}, y={data.y_test.shape}")
    print(f"Sampled original index range: {data.sampled_indices.min()} - {data.sampled_indices.max()}")


def evaluate_torch_regressor_on_original_scale(model, X, y_raw, y_scaler, device):
    """Predict with a PyTorch model and evaluate on the original y scale."""
    pred_scaled = predict_torch_model(model, X, device)
    pred = inverse_y(y_scaler, pred_scaled)
    metrics = evaluate_regression(y_raw, pred)
    return pred, metrics


def save_split_metrics(config, file_name, split_metrics):
    """Save Train / Valid / Test metrics for one model."""
    split_metrics_df = pd.DataFrame.from_dict(split_metrics, orient="index")
    split_metrics_df.index.name = "Split"
    split_metrics_df.to_csv(
        os.path.join(config.metrics_dir, file_name),
        encoding="utf-8-sig",
    )


def run_mlp_experiment(config, data, labeled_mask, device):
    print("\n" + "=" * 70)
    print("MLP baseline")
    print("=" * 70)
    print(f"Hidden sizes : {list(config.mlp_hidden_sizes)}")
    print(f"Activation   : {config.mlp_activation_name}")
    print(f"Dropout rate : {config.mlp_dropout_rate}")
    print(f"Weight decay : {config.supervised_weight_decay}")

    model = MLPRegressor(
        input_dim=data.X_train.shape[1],
        hidden_sizes=config.mlp_hidden_sizes,
        output_dim=1,
        activation_name=config.mlp_activation_name,
        dropout_rate=config.mlp_dropout_rate,
    )

    model, history, best_epoch, _ = train_supervised_model(
        model=model,
        X_train=data.X_train,
        y_train=data.y_train,
        X_valid=data.X_valid,
        y_valid=data.y_valid,
        labeled_mask=labeled_mask,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        patience=config.patience,
        device=device,
        seed=config.random_seed,
        weight_decay=config.supervised_weight_decay,
        model_path=os.path.join(config.models_dir, "best_mlp.pt"),
    )

    # Evaluate all splits to check whether the model is overfitting.
    train_pred, train_metrics = evaluate_torch_regressor_on_original_scale(
        model, data.X_train, data.y_train_raw, data.y_scaler, device
    )
    valid_pred, valid_metrics = evaluate_torch_regressor_on_original_scale(
        model, data.X_valid, data.y_valid_raw, data.y_scaler, device
    )
    pred, metrics = evaluate_torch_regressor_on_original_scale(
        model, data.X_test, data.y_test_raw, data.y_scaler, device
    )

    mlp_split_metrics = {
        "Train": train_metrics,
        "Valid": valid_metrics,
        "Test": metrics,
    }
    print_metrics_table("MLP metrics on original y scale", mlp_split_metrics)
    print_metrics("MLP", metrics)

    save_split_metrics(config, "mlp_split_metrics.csv", mlp_split_metrics)

    plot_loss_curve(
        history,
        "MLP Train / Valid Loss",
        os.path.join(config.mlp_figures_dir, "mlp_loss.png"),
        best_epoch=best_epoch,
    )
    plot_true_pred(
        data.y_test_raw,
        pred,
        "MLP True vs Predicted",
        os.path.join(config.mlp_figures_dir, "mlp_pred.png"),
    )
    plot_residuals(
        data.y_test_raw,
        pred,
        "MLP Residual Distribution",
        os.path.join(config.mlp_figures_dir, "mlp_residual.png"),
    )

    return {
        "Model": "MLP",
        **metrics,
        "Extra Info": (
            f"hidden sizes = {list(config.mlp_hidden_sizes)}; "
            f"activation = {config.mlp_activation_name}; "
            f"dropout = {config.mlp_dropout_rate}; "
            f"weight_decay = {config.supervised_weight_decay}; "
            f"best epoch = {best_epoch}"
        ),
    }


def run_gmm_experiment(config, data, labeled_mask):
    print("\n" + "=" * 70)
    print("GMM-based regression baseline")
    print("=" * 70)

    candidates = []
    for n_components in config.n_components_list:
        model = GMMMeanRegressor(n_components=n_components, random_state=config.random_seed)
        model.fit(data.X_train, data.y_train, labeled_mask)

        valid_pred_scaled = model.predict(data.X_valid)
        valid_pred = inverse_y(data.y_scaler, valid_pred_scaled)
        valid_metrics = evaluate_regression(data.y_valid_raw, valid_pred)

        candidates.append(
            {
                "n_components": n_components,
                "validation_rmse": valid_metrics["RMSE"],
                "model": model,
            }
        )
        print(f"GMM n_components={n_components:>2} | valid RMSE={valid_metrics['RMSE']:.6f}")

    # Save the validation search process so the selected n_components is traceable.
    pd.DataFrame(
        [
            {
                "n_components": item["n_components"],
                "Validation RMSE": item["validation_rmse"],
            }
            for item in candidates
        ]
    ).to_csv(
        os.path.join(config.metrics_dir, "gmm_validation_search.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    best_info = min(candidates, key=lambda item: item["validation_rmse"])
    best_model = best_info["model"]

    train_pred = inverse_y(data.y_scaler, best_model.predict(data.X_train))
    valid_pred = inverse_y(data.y_scaler, best_model.predict(data.X_valid))
    pred = inverse_y(data.y_scaler, best_model.predict(data.X_test))

    train_metrics = evaluate_regression(data.y_train_raw, train_pred)
    valid_metrics = evaluate_regression(data.y_valid_raw, valid_pred)
    metrics = evaluate_regression(data.y_test_raw, pred)

    gmm_split_metrics = {
        "Train": train_metrics,
        "Valid": valid_metrics,
        "Test": metrics,
    }

    print(f"Best n_components: {best_info['n_components']}")
    print(f"Best validation RMSE: {best_info['validation_rmse']:.6f}")
    print_metrics_table("GMM metrics on original y scale", gmm_split_metrics)
    print_metrics("GMM", metrics)
    save_split_metrics(config, "gmm_split_metrics.csv", gmm_split_metrics)

    plot_true_pred(
        data.y_test_raw,
        pred,
        "GMM True vs Predicted",
        os.path.join(config.gmm_figures_dir, "gmm_pred.png"),
    )
    plot_residuals(
        data.y_test_raw,
        pred,
        "GMM Residual Distribution",
        os.path.join(config.gmm_figures_dir, "gmm_residual.png"),
    )

    return {
        "Model": "GMM",
        **metrics,
        "Extra Info": (
            f"best n_components = {best_info['n_components']}; "
            f"validation RMSE = {best_info['validation_rmse']:.6f}"
        ),
    }


def run_aemlp_experiment(config, data, labeled_mask, device):
    print("\n" + "=" * 70)
    print("AE feature extractor + MLP regression")
    print("=" * 70)
    print(f"Latent dim          : {config.latent_dim}")
    print(f"Regressor activation: {config.aemlp_activation_name}")
    print(f"Regressor dropout   : {config.aemlp_dropout_rate}")
    print(f"Regressor weight decay: {config.supervised_weight_decay}")

    autoencoder = AutoEncoder(input_dim=data.X_train.shape[1], latent_dim=config.latent_dim)
    autoencoder, ae_history, ae_best_epoch, _ = train_autoencoder(
        autoencoder=autoencoder,
        X_train=data.X_train,
        X_valid=data.X_valid,
        epochs=config.ae_epochs,
        batch_size=config.ae_batch_size,
        learning_rate=config.ae_learning_rate,
        patience=config.ae_patience,
        device=device,
        seed=config.random_seed,
        model_path=os.path.join(config.models_dir, "best_ae.pt"),
    )

    plot_loss_curve(
        ae_history,
        "AE Reconstruction Loss",
        os.path.join(config.aemlp_figures_dir, "ae_loss.png"),
        best_epoch=ae_best_epoch,
        y_label="Reconstruction MSE Loss",
    )

    # Extract latent features, then train the regressor on z instead of raw X.
    Z_train = extract_encoder_features(autoencoder, data.X_train, device)
    Z_valid = extract_encoder_features(autoencoder, data.X_valid, device)
    Z_test = extract_encoder_features(autoencoder, data.X_test, device)
    print(f"Latent feature shapes: train={Z_train.shape}, valid={Z_valid.shape}, test={Z_test.shape}")

    regressor = AEMLPRegressor(
        latent_dim=config.latent_dim,
        activation_name=config.aemlp_activation_name,
        dropout_rate=config.aemlp_dropout_rate,
    )
    regressor, history, best_epoch, _ = train_supervised_model(
        model=regressor,
        X_train=Z_train,
        y_train=data.y_train,
        X_valid=Z_valid,
        y_valid=data.y_valid,
        labeled_mask=labeled_mask,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        patience=config.patience,
        device=device,
        seed=config.random_seed,
        weight_decay=config.supervised_weight_decay,
        model_path=os.path.join(config.models_dir, "best_aemlp.pt"),
    )

    pred_scaled = predict_torch_model(regressor, Z_test, device)
    pred = inverse_y(data.y_scaler, pred_scaled)
    metrics = evaluate_regression(data.y_test_raw, pred)

    train_pred, train_metrics = evaluate_torch_regressor_on_original_scale(
        regressor, Z_train, data.y_train_raw, data.y_scaler, device
    )
    valid_pred, valid_metrics = evaluate_torch_regressor_on_original_scale(
        regressor, Z_valid, data.y_valid_raw, data.y_scaler, device
    )
    aemlp_split_metrics = {
        "Train": train_metrics,
        "Valid": valid_metrics,
        "Test": metrics,
    }
    print_metrics_table("AE+MLP metrics on original y scale", aemlp_split_metrics)
    print_metrics("AE+MLP", metrics)
    save_split_metrics(config, "aemlp_split_metrics.csv", aemlp_split_metrics)

    plot_loss_curve(
        history,
        "AE+MLP Train / Valid Regression Loss",
        os.path.join(config.aemlp_figures_dir, "aemlp_loss.png"),
        best_epoch=best_epoch,
    )
    plot_true_pred(
        data.y_test_raw,
        pred,
        "AE+MLP True vs Predicted",
        os.path.join(config.aemlp_figures_dir, "aemlp_pred.png"),
    )
    plot_residuals(
        data.y_test_raw,
        pred,
        "AE+MLP Residual Distribution",
        os.path.join(config.aemlp_figures_dir, "aemlp_residual.png"),
    )

    return {
        "Model": "AE+MLP",
        **metrics,
        "Extra Info": (
            f"latent_dim = {config.latent_dim}; "
            f"activation = {config.aemlp_activation_name}; "
            f"dropout = {config.aemlp_dropout_rate}; "
            f"weight_decay = {config.supervised_weight_decay}; "
            f"AE best epoch = {ae_best_epoch}; MLP best epoch = {best_epoch}"
        ),
    }


def main():
    configure_stdout()

    config = ExperimentConfig()
    config.ensure_dirs()
    set_random_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = prepare_data(config)
    print_data_report(data)

    labeled_mask = make_labeled_mask(len(data.X_train), config.labeled_ratio, config.random_seed)
    print(f"Labeled train samples: {labeled_mask.sum()} / {len(labeled_mask)}")

    results_rows = [
        run_mlp_experiment(config, data, labeled_mask, device),
        run_gmm_experiment(config, data, labeled_mask),
        run_aemlp_experiment(config, data, labeled_mask, device),
    ]

    metrics_df = pd.DataFrame(results_rows, columns=["Model", "MSE", "RMSE", "MAE", "R²", "Extra Info"])
    summary_path = os.path.join(config.metrics_dir, "metrics_summary.csv")
    metrics_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nSaved metrics summary to:", summary_path)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
