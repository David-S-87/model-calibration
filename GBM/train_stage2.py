# GBM/evaluate_stage2.py

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path

# ─ make sure common/ is importable ──────────────────────────────────────────
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# ─ your exact MLP definition ───────────────────────────────────────────────
class Stage2MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256,256]):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def main():
    # paths
    DATA_PATH = Path(__file__).parent / "../data/synthetic_data2/GBM/grouped_stage2_gbm.npz"
    CKPT      = Path(__file__).parent / "checkpoints/stage2_gbm_model.pth"
    PLOTS     = Path(__file__).parent / "plots"
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load data
    arr         = np.load(str(DATA_PATH))
    priors      = arr["priors"]            # (N, P)
    feats, pr   = arr["option_features"], arr["option_prices"]
    true_params = arr["true_params"]       # (N, P)
    N, M, F     = feats.shape
    P           = priors.shape[1]

    # 2) Flatten inputs
    x = np.concatenate(
        [
            feats.reshape(N, M * F),
            pr.reshape(N, M),
            priors,
        ],
        axis=1,
    )

    # 3) Load checkpoint **first**, infer the right input_dim
    state = torch.load(str(CKPT), map_location=DEVICE)
    # "net.0.weight" is the first Linear layer's weight: shape [H, input_dim]
    ckpt_input_dim = state["net.0.weight"].shape[1]
    print(f"✔ Found checkpoint with input_dim = {ckpt_input_dim}")

    # 4) If your data has more features, *truncate* to match the checkpoint:
    if X.shape[1] > ckpt_input_dim:
        print(f"⚠️  Truncating data from {X.shape[1]} → {ckpt_input_dim} features")
        X = X[:, :ckpt_input_dim]
    elif X.shape[1] < ckpt_input_dim:
        raise RuntimeError(f"Data has fewer features ({X.shape[1]}) than the checkpoint expects ({ckpt_input_dim}).")

    # 5) Build & load the model
    model = Stage2MLP(ckpt_input_dim, P).to(DEVICE)
    model.load_state_dict(state)
    model.eval()

    # 6) Predict
    with torch.no_grad():
        X_t   = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        post  = model(X_t).cpu().numpy()

    # 7) Report MSE
    prior_mse = np.mean((priors - true_params)**2, axis=0)
    post_mse  = np.mean((post   - true_params)**2, axis=0)
    print("\nParam  Prior MSE  Posterior MSE")
    for name, pm, qm in zip(["mu","sigma"], prior_mse, post_mse):
        print(f"{name:>5}  {pm:8.4f}    {qm:8.4f}")

    # 8) Plot
    PLOTS.mkdir(exist_ok=True)
    for i, name in enumerate(["mu","sigma"]):
        plt.figure()
        plt.scatter(true_params[:,i], priors[:,i], alpha=0.4, label="Prior")
        plt.scatter(true_params[:,i], post[:,i],   alpha=0.4, label="Posterior")
        mn, mx = true_params[:,i].min(), true_params[:,i].max()
        plt.plot([mn,mx],[mn,mx],"r--")
        plt.xlabel(f"True {name}")
        plt.ylabel("Estimate")
        plt.legend(); plt.tight_layout()
        out = PLOTS/f"{name}_stage2.png"
        plt.savefig(str(out))
        print("Saved plot:", out)
        plt.close()

if __name__=="__main__":
    main()
