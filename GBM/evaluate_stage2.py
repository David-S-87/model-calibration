# GBM/evaluate_stage2.py

import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path

# ensure common/ on path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

class Stage2MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256,        256), nn.ReLU(),
            nn.Linear(256,        output_dim)
        )
    def forward(self, x):
        return self.net(x)

def main():
    DATA_PATH = Path(__file__).parent / "../data/synthetic_data2/GBM/grouped_stage2_gbm.npz"
    CKPT      = Path(__file__).parent / "checkpoints/stage2_gbm_model.pth"
    PLOTS     = Path(__file__).parent / "plots"
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arr          = np.load(str(DATA_PATH))
    priors       = arr["priors"]
    feats, pr, _ = arr["option_features"], arr["option_prices"], arr["true_params"]
    true_params  = arr["true_params"]
    N, M, F      = feats.shape; P = priors.shape[1]
    input_dim    = M*F + M + P

    print(f"🔎 Evaluating with input_dim={input_dim}, output_dim={P}")
    model = Stage2MLP(input_dim, P).to(DEVICE)

    state = torch.load(str(CKPT), map_location=DEVICE)
    model.load_state_dict(state)  # now matches

    model.eval()
    X_flat  = np.hstack([feats.reshape(N, M*F), pr.reshape(N, M), priors])
    with torch.no_grad():
        post = model(torch.tensor(X_flat, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    # Report & plot
    prior_mse = np.mean((priors - true_params)**2, axis=0)
    post_mse  = np.mean((post   - true_params)**2, axis=0)
    print("\nParam  Prior MSE  Post MSE")
    for name, pm, qm in zip(["mu","sigma"], prior_mse, post_mse):
        print(f"{name:>5}  {pm:>8.4f}  {qm:>8.4f}")

    PLOTS.mkdir(exist_ok=True)
    for i, name in enumerate(["mu","sigma"]):
        plt.figure()
        plt.scatter(true_params[:,i], priors[:,i], alpha=0.4, label="Prior")
        plt.scatter(true_params[:,i], post[:,i],   alpha=0.4, label="Posterior")
        mn, mx = true_params[:,i].min(), true_params[:,i].max()
        plt.plot([mn,mx],[mn,mx],"r--")
        plt.xlabel(f"True {name}"); plt.ylabel("Estimate")
        plt.legend(); plt.tight_layout()
        plt.savefig(str(PLOTS/f"{name}_stage2.png"))
        plt.close()

if __name__=="__main__":
    main()
