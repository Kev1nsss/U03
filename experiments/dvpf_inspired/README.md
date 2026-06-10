# DVPF-Inspired Cross-Condition Experiments

This folder contains a simplified implementation inspired by the DVPF paper:

> A Deep Probabilistic Flow-Based Framework for Unsupervised Cross-Domain Soft Sensing

The goal is not to reproduce the full paper exactly. The full DVPF method uses sequential variational Bayes and a potential-flow posterior update. This implementation keeps the part that is most useful for the Unit03 project:

- leave-one-condition-out cross-condition evaluation;
- source conditions have labels;
- the held-out target condition uses only unlabeled `X` during training;
- target `y` is used only for final testing;
- GRU sequence encoder extracts dynamic features;
- VAE-style latent variables and reconstruction use target unlabeled data;
- an optional residual latent flow approximates the paper's flow idea;
- optional CORAL/MMD aligns source and target latent features.

Compared with the earlier AECL calibration experiment, this setting is stricter:

- AECL calibration: each target condition provides 1000 labeled calibration samples.
- DVPF-inspired: target condition provides no labels during training.

So poor results here are expected unless the unlabeled target distribution contains enough information to identify the label shift.

## Commands

Smoke test:

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\dvpf_inspired\run_gru_vae_flow_leave_one.py --targets late_stable --seq-len 10 --epochs 2 --patience 2 --batch-size 64 --max-source-per-condition 200 --max-target-unlabeled 200 --summary-prefix smoke_gru_vae_flow
```

All-condition pilot:

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\dvpf_inspired\run_gru_vae_flow_leave_one.py --seq-len 20 --epochs 40 --patience 10 --batch-size 128 --max-source-per-condition 1000 --max-target-unlabeled 2000 --summary-prefix pilot_gru_vae_flow_all_conditions
```

CORAL variant:

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\dvpf_inspired\run_gru_vae_flow_leave_one.py --alignment coral --alignment-weight 0.1 --seq-len 20 --epochs 40 --patience 10 --batch-size 128 --max-source-per-condition 1000 --max-target-unlabeled 2000 --summary-prefix pilot_gru_vae_flow_coral_all_conditions
```

Calibrated GRU-VAE-flow variant:

```powershell
D:\miniconda3\envs\d2l\python.exe -B experiments\dvpf_inspired\run_calibrated_gru_vae_flow_leave_one.py --include-source --n-calibration-list 1000 --n-trials 3 --seq-len 20 --epochs 100 --patience 25 --batch-size 256 --max-source-per-condition 3000 --max-target-unlabeled 6000 --summary-prefix formal_calibrated_gru_vae_flow_1000_include_source
```

Results are written to `results/dvpf_inspired/`.

## Pilot Results

Strict target-unlabeled setting:

```text
GRU-VAE-flow        avg RMSE = 2.2260, avg MAE = 1.7447, avg R2 = -11.4577
GRU-VAE-flow+CORAL  avg RMSE = 2.2329, avg MAE = 1.7516, avg R2 = -11.5212
GRU-VAE-flow+MMD    avg RMSE = 2.2873, avg MAE = 1.8089, avg R2 = -9.7773
```

This setting does not beat the original MLP baseline. It shows that reconstruction and latent alignment alone are not enough to transfer the `X -> y` mapping across these operating conditions.

Calibrated setting with 1000 target labels:

```text
calibrated GRU-VAE-flow, include-source, n_calibration = 1000, n_trials = 3
avg RMSE = 0.8350
avg MAE  = 0.6047
avg R2   = 0.6445
worst condition = long_stable, RMSE = 1.3664
7/7 conditions have RMSE below the original MLP baseline RMSE = 1.4514
```

This is the useful DVPF-inspired direction: sequence probabilistic representation plus few-shot target-condition calibration. It is stable below the original MLP RMSE baseline, but it does not beat the current AE-MLP formal average RMSE of 0.8201.
