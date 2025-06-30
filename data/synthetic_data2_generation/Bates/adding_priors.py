# adding_priors.py

"""
Add priors to Stage 2 dataset for Bates model using trained Stage 1 network.
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

MODEL_NAME = "Bates"
CHECKPOINT_PATH = project_root / "Bates" / "checkpoints" / "stage1_bates_model.pth"
STAGE1_DATA_PATH = project_root / "data" / "synthetic_data1" / "Bates" / "stage1_bates_dataset.npz"
STAGE2_DATA_PATH = project_root / "data" / "synthetic_data2" / "Bates" / "stage2_bates_dataset.npz"
OUTPUT_PATH = project_root / "data" / "synthetic_data2" / "Bates" / "grouped_stage2_bates.npz"

DEVICE = torch.device("cpu")
N_PARAMS = 9

def load_stage1_model():
    model = get_stage1_network(input_dim=12, output_dim=N_PARAMS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model

def construct_option_features(strikes, maturities, S0):
    num_options, num_sets = strikes.shape
    features = []
    for i in range(num_sets):
        feat = np.stack([strikes[:, i], maturities[:, i], np.full_like(strikes[:, i], S0[i])], axis=1)
        features.append(feat)
    return features

def group_stage2_data(data):
    features_list = construct_option_features(data["strikes"], data["maturities"], data["S0"])
    grouped = defaultdict(lambda: {"features": [], "prices": [], "true_params": None})
    for set_id, (feat, price, param) in enumerate(zip(features_list, data["option_prices"], data["true_params"])):
        grouped[set_id]["features"] = feat
        grouped[set_id]["prices"] = price
        grouped[set_id]["true_params"] = param[:N_PARAMS]
    return grouped

def load_summary_stats():
    arr = np.load(STAGE1_DATA_PATH)
    return dict(enumerate(arr["summary_stats"]))

def main():
    print(f"🔧 Generating priors for {MODEL_NAME} Stage 2...")
    stage2_data = np.load(STAGE2_DATA_PATH)
    summary_map = load_summary_stats()
    model = load_stage1_model()
    grouped = group_stage2_data(stage2_data)

    priors, feats, pris, trues = [], [], [], []
    for sid, grp in grouped.items():
        stats = summary_map[sid]
        with torch.no_grad():
            prior = model(torch.tensor(stats, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        priors.append(prior)
        feats.append(np.array(grp["features"]))
        pris.append(np.array(grp["prices"]))
        trues.append(grp["true_params"])

    np.savez(OUTPUT_PATH, priors=np.array(priors), option_features=np.array(feats),
             option_prices=np.array(pris), true_params=np.array(trues))
    print(f"✅ Saved with priors to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
