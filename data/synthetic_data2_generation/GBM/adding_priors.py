# adding_priors.py

"""
Add priors to Stage 2 dataset for GBM model using trained Stage 1 network.
Outputs grouped dataset with priors per parameter set.
"""

import os
import sys
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from common import get_stage1_network

# --- Paths ---
MODEL_NAME = "GBM"
CHECKPOINT_PATH = project_root / "GBM" / "checkpoints" / "stage1_gbm_model.pth"
STAGE1_DATA_PATH = project_root / "data" / "synthetic_data1" / "GBM" / "stage1_gbm_dataset.npz"
STAGE2_DATA_PATH = project_root / "data" / "synthetic_data2" / "GBM" / "stage2_gbm_dataset.npz"
OUTPUT_PATH = project_root / "data" / "synthetic_data2" / "GBM" / "grouped_stage2_gbm.npz"

DEVICE = torch.device("cpu")
N_PARAMS = 2

# --- Load Trained Model ---
def load_stage1_model():
    model = get_stage1_network(input_dim=12, output_dim=N_PARAMS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model

# --- Construct option features from strikes, maturities, and S0 ---
def construct_option_features(strikes, maturities, S0):
    num_options, num_sets = strikes.shape
    features = []
    for i in range(num_sets):
        strike_i = strikes[:, i]
        maturity_i = maturities[:, i]
        S0_i = S0[i] if S0.ndim > 0 else S0
        S0_vec = np.full_like(strike_i, S0_i)
        # You can add more components here if needed
        feat = np.stack([strike_i, maturity_i, S0_vec], axis=1)  # shape: (num_options, 3)
        features.append(feat)
    return features

# --- Group Stage 2 Data by Set ID ---
def group_stage2_data(stage2_data):
    features_list = construct_option_features(stage2_data["strikes"],
                                              stage2_data["maturities"],
                                              stage2_data["S0"])
    grouped = defaultdict(lambda: {"features": [], "prices": [], "true_params": None})
    for set_id, (feat, price, param) in enumerate(zip(features_list,
                                                      stage2_data["option_prices"],
                                                      stage2_data["true_params"])):
        grouped[set_id]["features"] = feat
        grouped[set_id]["prices"] = price
        grouped[set_id]["true_params"] = param
    return grouped

# --- Load Summary Statistics ---
def load_summary_stats():
    data = np.load(STAGE1_DATA_PATH)
    return dict(enumerate(data["summary_stats"]))  # map set_id → stats

# --- Main Logic ---
def main():
    print(f"🔧 Generating priors for {MODEL_NAME} Stage 2...")

    # Load files
    stage2_data = np.load(STAGE2_DATA_PATH)
    summary_stats_map = load_summary_stats()
    model = load_stage1_model()

    # Group option data
    grouped = group_stage2_data(stage2_data)

    # Output containers
    priors, features, prices, targets = [], [], [], []

    for set_id, group in grouped.items():
        stats = summary_stats_map[set_id]
        stats_tensor = torch.tensor(stats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prior = model(stats_tensor).squeeze(0).numpy()

        priors.append(prior)
        features.append(np.array(group["features"]))
        prices.append(np.array(group["prices"]))
        targets.append(group["true_params"])

    # Save grouped Stage 2 dataset
    np.savez(OUTPUT_PATH,
             priors=np.array(priors),
             option_features=np.array(features),
             option_prices=np.array(prices),
             true_params=np.array(targets))

    print(f"✅ Saved with priors to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
