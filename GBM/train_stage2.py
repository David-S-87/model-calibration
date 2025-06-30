# GBM/evaluate_stage2.py

import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --- Project root on PYTHONPATH so we can import common ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from common.networks_stage2   import get_stage2_network
from common.loss_functions    import stage2_mse_with_prior_reg

# --- Config ---
DATA_PATH       = r"C:\Users\david\BathUni\MA50290_24\model_calibration\data\synthetic_data2\GBM\grouped_stage2_gbm.npz"
CHECKPOINT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")
SAVE_PATH       = os.path.join(CHECKPOINT_DIR, "stage2_gbm_model.pth")

BATCH_SIZE      = 32
EPOCHS          = 100
LEARNING_RATE   = 1e-3
PRIOR_REG_ALPHA = 0.1
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Stage2GBMDataset(Dataset):
    """
    Dataset for Stage 2 GBM:
    - priors:        (N, 2)
    - option_feats:  (N, M, F)
    - option_prices: (N, M)
    - true_params:   (N, 2)
    We flatten [option_feats, option_prices, priors] into one input vector.
    """
    def __init__(self, data_path):
        arr = np.load(data_path)
        self.priors = torch.tensor(arr["priors"], dtype=torch.float32)
        self.feats  = torch.tensor(arr["option_features"], dtype=torch.float32)
        self.prices = torch.tensor(arr["option_prices"], dtype=torch.float32)
        self.targets= torch.tensor(arr["true_params"], dtype=torch.float32)

        # Dimensions
        self.N, self.M, self.F = self.feats.shape   # N sets, M options, F features per option
        self.P = self.priors.shape[1]               # number of parameters (2 for GBM)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # Flatten option features and prices
        feat_flat  = self.feats[idx].view(-1)       # shape: (M*F,)
        price_flat = self.prices[idx].view(-1)      # shape: (M,)
        prior      = self.priors[idx]               # shape: (P,)
        # Concatenate into one input vector
        x = torch.cat([feat_flat, price_flat, prior], dim=0)
        y = self.targets[idx]                       # shape: (P,)
        return x, prior, y


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
