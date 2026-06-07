from pathlib import Path

import pandas as pd
import torch

from unit03_soft_sensor.evaluation import evaluate_regression, inverse_y
from unit03_soft_sensor.models import AEMLPFineTuner, AEMLPRegressor, AutoEncoder
from unit03_soft_sensor.plotting import plot_loss_curve, plot_residuals, plot_true_pred
from unit03_soft_sensor.train import (
    extract_encoder_features,
    predict_torch_model,
    train_autoencoder,
    train_supervised_model,
)


def ensure_ablation_dirs(output_root):
    """为某个消融实验创建 metrics / figures / models 三类输出目录。"""
    output_root = Path(output_root)
    dirs = {
        "metrics": output_root / "metrics",
        "figures": output_root / "figures",
        "models": output_root / "models",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def evaluate_regressor_split(model, X, y_raw, y_scaler, device):
    """在原始 y 尺度上评价 PyTorch 回归器。"""
    pred_scaled = predict_torch_model(model, X, device)
    pred = inverse_y(y_scaler, pred_scaled)
    metrics = evaluate_regression(y_raw, pred)
    return pred, metrics


def save_split_metrics(path, split_metrics):
    split_metrics_df = pd.DataFrame.from_dict(split_metrics, orient="index")
    split_metrics_df.index.name = "Split"
    split_metrics_df.to_csv(path, encoding="utf-8-sig")


def run_aemlp_trial(config, data, labeled_mask, device, output_root, trial_name):
    """运行一次 AE+MLP 消融实验。

    一个 trial 对应一组超参数，例如 latent_dim=8 或 dropout=0.05。
    这里保存三类结果：
    1. metrics: Train / Valid / Test 指标
    2. figures: AE loss、AE+MLP loss、预测图、残差图
    3. models: 当前 trial 的 best AE 和 best AEMLP 参数
    """
    dirs = ensure_ablation_dirs(output_root)

    print("\n" + "=" * 80)
    print(f"AE+MLP ablation trial: {trial_name}")
    print(f"latent_dim={config.latent_dim}, dropout={config.aemlp_dropout_rate}")
    print("=" * 80)

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
        model_path=dirs["models"] / f"best_ae_{trial_name}.pt",
    )

    plot_loss_curve(
        ae_history,
        f"AE Reconstruction Loss ({trial_name})",
        dirs["figures"] / f"ae_loss_{trial_name}.png",
        best_epoch=ae_best_epoch,
        y_label="Reconstruction MSE Loss",
    )

    Z_train = extract_encoder_features(autoencoder, data.X_train, device)
    Z_valid = extract_encoder_features(autoencoder, data.X_valid, device)
    Z_test = extract_encoder_features(autoencoder, data.X_test, device)

    regressor = AEMLPRegressor(
        latent_dim=config.latent_dim,
        activation_name=config.aemlp_activation_name,
        dropout_rate=config.aemlp_dropout_rate,
    )
    regressor, history, reg_best_epoch, _ = train_supervised_model(
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
        model_path=dirs["models"] / f"best_aemlp_{trial_name}.pt",
    )

    train_pred, train_metrics = evaluate_regressor_split(
        regressor, Z_train, data.y_train_raw, data.y_scaler, device
    )
    valid_pred, valid_metrics = evaluate_regressor_split(
        regressor, Z_valid, data.y_valid_raw, data.y_scaler, device
    )
    test_pred, test_metrics = evaluate_regressor_split(
        regressor, Z_test, data.y_test_raw, data.y_scaler, device
    )

    split_metrics = {
        "Train": train_metrics,
        "Valid": valid_metrics,
        "Test": test_metrics,
    }
    save_split_metrics(dirs["metrics"] / f"{trial_name}_split_metrics.csv", split_metrics)

    plot_loss_curve(
        history,
        f"AE+MLP Regression Loss ({trial_name})",
        dirs["figures"] / f"aemlp_loss_{trial_name}.png",
        best_epoch=reg_best_epoch,
    )
    plot_true_pred(
        data.y_test_raw,
        test_pred,
        f"AE+MLP True vs Predicted ({trial_name})",
        dirs["figures"] / f"aemlp_pred_{trial_name}.png",
    )
    plot_residuals(
        data.y_test_raw,
        test_pred,
        f"AE+MLP Residual Distribution ({trial_name})",
        dirs["figures"] / f"aemlp_residual_{trial_name}.png",
    )

    return {
        "trial": trial_name,
        "latent_dim": config.latent_dim,
        "dropout": config.aemlp_dropout_rate,
        "Valid RMSE": valid_metrics["RMSE"],
        "Valid R²": valid_metrics["R²"],
        "Test RMSE": test_metrics["RMSE"],
        "Test MAE": test_metrics["MAE"],
        "Test R²": test_metrics["R²"],
        "AE best epoch": ae_best_epoch,
        "AEMLP best epoch": reg_best_epoch,
    }



def run_finetune_trial(config, data, labeled_mask, device, output_root, trial_name):
    """运行一次 AE pretrain + Encoder fine-tune 实验。

    这个 trial 和普通 AE+MLP 的最大区别：
    - AE 仍然先用 100% X_train 做无监督预训练；
    - 之后不再冻结 Encoder；
    - 使用 20% 标签样本训练 Encoder + MLP 回归头，让 latent feature 更贴近 y 预测任务。
    """
    dirs = ensure_ablation_dirs(output_root)

    print("\n" + "=" * 80)
    print(f"AE pretrain + fine-tune trial: {trial_name}")
    print(f"latent_dim={config.latent_dim}, dropout={config.aemlp_dropout_rate}")
    print("=" * 80)

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
        model_path=dirs["models"] / f"pretrained_ae_{trial_name}.pt",
    )

    plot_loss_curve(
        ae_history,
        f"AE Pretrain Reconstruction Loss ({trial_name})",
        dirs["figures"] / f"ae_pretrain_loss_{trial_name}.png",
        best_epoch=ae_best_epoch,
        y_label="Reconstruction MSE Loss",
    )

    finetune_model = AEMLPFineTuner(
        encoder=autoencoder.encoder,
        latent_dim=config.latent_dim,
        activation_name=config.aemlp_activation_name,
        dropout_rate=config.aemlp_dropout_rate,
    )

    finetune_model, history, finetune_best_epoch, _ = train_supervised_model(
        model=finetune_model,
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
        model_path=dirs["models"] / f"best_finetune_aemlp_{trial_name}.pt",
    )

    train_pred, train_metrics = evaluate_regressor_split(
        finetune_model, data.X_train, data.y_train_raw, data.y_scaler, device
    )
    valid_pred, valid_metrics = evaluate_regressor_split(
        finetune_model, data.X_valid, data.y_valid_raw, data.y_scaler, device
    )
    test_pred, test_metrics = evaluate_regressor_split(
        finetune_model, data.X_test, data.y_test_raw, data.y_scaler, device
    )

    split_metrics = {
        "Train": train_metrics,
        "Valid": valid_metrics,
        "Test": test_metrics,
    }
    save_split_metrics(dirs["metrics"] / f"{trial_name}_split_metrics.csv", split_metrics)

    plot_loss_curve(
        history,
        f"Fine-tuned AE+MLP Regression Loss ({trial_name})",
        dirs["figures"] / f"finetune_loss_{trial_name}.png",
        best_epoch=finetune_best_epoch,
    )
    plot_true_pred(
        data.y_test_raw,
        test_pred,
        f"Fine-tuned AE+MLP True vs Predicted ({trial_name})",
        dirs["figures"] / f"finetune_pred_{trial_name}.png",
    )
    plot_residuals(
        data.y_test_raw,
        test_pred,
        f"Fine-tuned AE+MLP Residual Distribution ({trial_name})",
        dirs["figures"] / f"finetune_residual_{trial_name}.png",
    )

    return {
        "trial": trial_name,
        "latent_dim": config.latent_dim,
        "dropout": config.aemlp_dropout_rate,
        "Valid RMSE": valid_metrics["RMSE"],
        "Valid R²": valid_metrics["R²"],
        "Test RMSE": test_metrics["RMSE"],
        "Test MAE": test_metrics["MAE"],
        "Test R²": test_metrics["R²"],
        "AE best epoch": ae_best_epoch,
        "Fine-tune best epoch": finetune_best_epoch,
    }

