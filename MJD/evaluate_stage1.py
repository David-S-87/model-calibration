# evaluate_stage1.py

"""
Evaluate the Stage 1 MJD model on the training set.
Compares predicted parameters vs. true ones using MSE and scatter plots.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Setup paths and device ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from common import get_stage1_network

DATA_PATH = r"C:\Users\david\BathUni\MA50290_24\model_calibration\data\synthetic_data1\MJD\stage1_mjd_dataset.npz"
CHECKPOINT_PATH = r"C:\Users\david\BathUni\MA50290_24\model_calibration\MJD\checkpoints\stage1_mjd_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # --- Load data ---
    data = np.load(DATA_PATH)
    X = data["summary_stats"]
    y_true = data["true_params"]  # shape (N, 5)

    # --- Load model ---
    input_dim = X.shape[1]
    output_dim = y_true.shape[1]
    model = get_stage1_network(input_dim, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # --- Predict ---
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        y_pred = model(X_tensor).cpu().numpy()

    # --- Compute metrics ---
    mse = np.mean((y_pred - y_true) ** 2)
    print(f"Stage 1 MSE on full dataset: {mse:.6f}")

    # --- Plotting ---
    labels = ["mu", "sigma", "lambda", "mu_J", "sigma_J"]
    for i, label in enumerate(labels):
        plt.figure()
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        plt.xlabel(f"True {label}")
        plt.ylabel(f"Predicted {label}")
        plt.title(f"{label}: True vs. Predicted")
        plt.plot([y_true[:, i].min(), y_true[:, i].max()],
                 [y_true[:, i].min(), y_true[:, i].max()], 'r--')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"{label}_true_vs_pred.png")
        print(f"Saved plot: {label}_true_vs_pred.png")

if __name__ == "__main__":
    main()
