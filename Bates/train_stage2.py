# Bates/train_stage2.py

"""
Train Stage 2 posterior update network for the Bates model.
Inputs: grouped option_features, option_prices, and priors.
Target: true Bates parameters [mu, v0, theta, kappa, sigma_v, rho, lambda, mu_J, sigma_J].
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Repository root for ``common`` imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from common.networks_stage2 import get_stage2_network
from common.loss_functions  import stage2_mse_with_prior_reg

DATA_PATH      = project_root / "data" / "synthetic_data2" / "Bates" / "grouped_stage2_bates.npz"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
SAVE_PATH      = CHECKPOINT_DIR / "stage2_bates_model.pth"

BATCH_SIZE      = 32
EPOCHS          = 100
LEARNING_RATE   = 1e-3
PRIOR_REG_ALPHA = 0.1
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stage2BatesDataset(Dataset):
    def __init__(self, data_path):
        arr = np.load(data_path)
        self.priors  = torch.tensor(arr["priors"], dtype=torch.float32)
        self.feats   = torch.tensor(arr["option_features"], dtype=torch.float32)
        self.prices  = torch.tensor(arr["option_prices"], dtype=torch.float32)
        self.targets = torch.tensor(arr["true_params"], dtype=torch.float32)
        self.N, self.M, self.F = self.feats.shape
        self.P = self.priors.shape[1]  # 9 for Bates

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        feat_flat  = self.feats[idx].view(-1)
        price_flat = self.prices[idx].view(-1)
        prior      = self.priors[idx]
        x = torch.cat([feat_flat, price_flat, prior], dim=0)
        y = self.targets[idx]
        return x, prior, y


def main():
    ds     = Stage2BatesDataset(DATA_PATH)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    input_dim  = ds.M * ds.F + ds.M + ds.P
    output_dim = ds.P

    model     = get_stage2_network(input_dim, output_dim, arch="mlp").to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for x, prior, y in loader:
            x, prior, y = x.to(DEVICE), prior.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = stage2_mse_with_prior_reg(pred, y, prior, alpha=PRIOR_REG_ALPHA)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)

        avg_loss = epoch_loss / len(ds)
        print(f"Epoch {epoch:3d}/{EPOCHS} — Loss: {avg_loss:.6f}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Stage 2 Bates model saved to:\n   {SAVE_PATH}")


if __name__ == "__main__":
    main()
