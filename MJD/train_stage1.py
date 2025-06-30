# MJD/train_stage1.py

"""
Train Stage 1 prior network for the MJD model:
maps summary statistics → [mu, sigma, lambda, mu_J, sigma_J].

Loads synthetic data from data/synthetic_data1/MJD, trains an MLP, and
saves the checkpoint in MJD/checkpoints.
"""

import os
import sys

# Ensure project root is on PYTHONPATH so we can import 'common'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from common import get_stage1_network, stage1_mse

# Configuration
DATA_PATH = r"C:\Users\david\BathUni\MA50290_24\model_calibration\data\synthetic_data1\MJD\stage1_mjd_dataset.npz"
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
SAVE_PATH = os.path.join(CHECKPOINT_DIR, "stage1_mjd_model.pth")

BATCH_SIZE    = 64
EPOCHS        = 50
LEARNING_RATE = 1e-3
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- STAT WEIGHTS based on highest |corr| per parameter ---
# Boost stat_0, stat_1, stat_8 (indices 0,1,8)
STAT_WEIGHTS = torch.tensor(
    [3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0],
    dtype=torch.float32
)

def main():
    # --- Load synthetic data ---
    data = np.load(DATA_PATH)
    X = data["summary_stats"]    # shape: (N, 12)
    y = data["true_params"]      # shape: (N, 5)

    # Convert and weight
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X_tensor *= STAT_WEIGHTS     # apply per-stat weighting
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- Build model ---
    input_dim  = X.shape[1]
    output_dim = y.shape[1]
    model = get_stage1_network(input_dim, output_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Training loop ---
    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(xb)
            loss  = stage1_mse(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)

        avg_loss = epoch_loss / len(loader.dataset)
        print(f"Epoch {epoch:3d}/{EPOCHS} — Loss: {avg_loss:.6f}")

    # --- Save ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Trained model saved to:\n   {SAVE_PATH}")

if __name__ == "__main__":
    main()
