# MJD/evaluate_stage2.py

"""
Evaluate the Stage 2 posterior update network for the MJD model.
Compares prior vs true and posterior vs true for [mu, sigma, lambda, mu_J, sigma_J].
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Project root on PYTHONPATH so we can import common ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from common.networks_stage2 import get_stage2_network

# --- Config ---
DATA_PATH      = r"C:\Users\david\BathUni\MA50290_24\model_calibration\data\synthetic_data2\MJD\grouped_stage2_mjd.npz"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT     = os.path.join(CHECKPOINT_DIR, "stage2_mjd_model.pth")
PLOTS_DIR      = os.path.join(os.path.dirname(__file__), "plots")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_NAMES = ["mu", "sigma", "lambda", "mu_J", "sigma_J"]

def main():
    # --- Load grouped Stage 2 data ---
    data = np.load(DATA_PATH)
    priors        = data["priors"]           # shape: (N, 5)
    option_feats  = data["option_features"]  # shape: (N, M, F)
    option_prices = data["option_prices"]    # shape: (N, M)
    true_params   = data["true_params"]      # shape: (N, 5)

    N, M, F = option_feats.shape
    P = priors.shape[1]  # 5 for MJD

    # --- Prepare model ---
    input_dim = M * F + M + P
    model = get_stage2_network(input_dim, P, arch="mlp").to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # --- Flatten inputs for prediction ---
    feats_flat  = option_feats.reshape(N, M * F)
    prices_flat = option_prices.reshape(N, M)
    X_flat = np.hstack([feats_flat, prices_flat, priors])

    # --- Model predictions ---
    with torch.no_grad():
        X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(DEVICE)
        pred_params = model(X_tensor).cpu().numpy()

    # --- Compute MSEs ---
    prior_mse = np.mean((priors - true_params) ** 2, axis=0)
    post_mse  = np.mean((pred_params - true_params) ** 2, axis=0)

    print("Parameter      Prior MSE    Posterior MSE")
    for i, name in enumerate(PARAM_NAMES):
        print(f"{name:>8}       {prior_mse[i]:.6f}       {post_mse[i]:.6f}")

    # --- Plotting ---
    os.makedirs(PLOTS_DIR, exist_ok=True)
    for i, name in enumerate(PARAM_NAMES):
        plt.figure()
        plt.scatter(true_params[:, i], priors[:, i], alpha=0.4, label="Prior")
        plt.scatter(true_params[:, i], pred_params[:, i], alpha=0.4, label="Posterior")
        mn = min(true_params[:, i].min(), priors[:, i].min(), pred_params[:, i].min())
        mx = max(true_params[:, i].max(), priors[:, i].max(), pred_params[:, i].max())
        plt.plot([mn, mx], [mn, mx], 'r--', label="Ideal")
        plt.xlabel(f"True {name}")
        plt.ylabel("Estimate")
        plt.title(f"MJD Stage 2: {name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"{name}_stage2_mjd.png")
        plt.savefig(path)
        print(f"Saved plot: {path}")
        plt.close()

if __name__ == "__main__":
    main()
