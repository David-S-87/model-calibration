# GBM/evaluate_stage2.py

"""
Evaluate the Stage 2 posterior update network for the GBM model.
Compares prior vs true and posterior vs true for [mu, sigma].
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
DATA_PATH      = r"C:\Users\david\BathUni\MA50290_24\model_calibration\data\synthetic_data2\GBM\grouped_stage2_gbm.npz"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CHECKPOINT     = os.path.join(CHECKPOINT_DIR, "stage2_gbm_model.pth")
PLOTS_DIR      = os.path.join(os.path.dirname(__file__), "plots")
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PARAM_NAMES = ["mu", "sigma"]

def main():
    # --- Load grouped Stage 2 data ---
    data = np.load(DATA_PATH)
    priors         = data["priors"]           # shape: (N, 2)
    option_feats   = data["option_features"]  # shape: (N, M, F)
    option_prices  = data["option_prices"]    # shape: (N, M)
    true_params    = data["true_params"]      # shape: (N, 2)

    N, M, F = option_feats.shape
    P = priors.shape[1]  # 2 for GBM

    # --- Prepare model ---
    input_dim = M * F + M + P
    model = get_stage2_network(input_dim, P, arch="mlp").to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    # --- Flatten inputs for prediction ---
    # features: flatten each (M, F) to (M*F), prices: (M), priors: (P)
    feats_flat  = option_feats.reshape(N, M * F)
    prices_flat = option_prices.reshape(N, M)
    X_flat = np.hstack([feats_flat, prices_flat, priors])  # shape: (N, input_dim)

    # --- Model predictions ---
    with torch.no_grad():
        X_tensor = torch.tensor(X_flat, dtype=torch.float32).to(DEVICE)
        pred_params = model(X_tensor).cpu().numpy()  # shape: (N, 2)

    # --- Compute MSEs ---
    prior_mse = np.mean((priors - true_params) ** 2, axis=0)
    post_mse  = np.mean((pred_params - true_params) ** 2, axis=0)

    print("Parameter\tPrior MSE\tPosterior MSE")
    for i, name in enumerate(PARAM_NAMES):
        print(f"{name:>6}\t{prior_mse[i]:.6f}\t{post_mse[i]:.6f}")

    # --- Ensure plots directory exists ---
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- Scatter plots ---
    for i, name in enumerate(PARAM_NAMES):
        plt.figure()
        # Prior vs True
        plt.scatter(true_params[:, i], priors[:, i], alpha=0.4, label="Prior")
        # Posterior vs True
        plt.scatter(true_params[:, i], pred_params[:, i], alpha=0.4, label="Posterior")
        # 45-degree line
        mn = min(true_params[:, i].min(), priors[:, i].min(), pred_params[:, i].min())
        mx = max(true_params[:, i].max(), priors[:, i].max(), pred_params[:, i].max())
        plt.plot([mn, mx], [mn, mx], 'r--', label="Ideal")
        plt.xlabel(f"True {name}")
        plt.ylabel(f"Estimate")
        plt.title(f"GBM Stage 2: {name} (Prior vs Posterior)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f"{name}_stage2_gbm.png")
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
        plt.close()

if __name__ == "__main__":
    main()
