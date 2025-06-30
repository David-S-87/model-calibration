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

# Add repository root to the path so ``common`` can be imported
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from common import get_stage1_network

# --- Paths & Constants ---
MODEL_NAME       = "Bates"
N_PARAMS         = 9
CHECKPOINT_PATH  = project_root / "Bates" / "checkpoints" / "stage1_bates_model.pth"
STAGE1_DATA_PATH = project_root / "data" / "synthetic_data1" / "Bates" / "stage1_bates_dataset.npz"
STAGE2_DATA_PATH = project_root / "data" / "synthetic_data2" / "Bates" / "stage2_bates_dataset.npz"
OUTPUT_PATH      = project_root / "data" / "synthetic_data2" / "Bates" / "grouped_stage2_bates.npz"

DEVICE = torch.device("cpu")

def load_stage1_model():
    model = get_stage1_network(input_dim=12, output_dim=N_PARAMS)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model

def load_summary_stats():
    arr = np.load(STAGE1_DATA_PATH)
    return dict(enumerate(arr["summary_stats"]))

def group_stage2_data(data):
    grouped = defaultdict(lambda: {"features": [], "prices": [], "true_params": None})
    for feat, price, true_p, sid in zip(data["option_features"],
                                        data["option_prices"],
                                        data["true_params"],
                                        data["set_ids"]):
        grouped[sid]["features"].append(feat)
        grouped[sid]["prices"].append(price)
        grouped[sid]["true_params"] = true_p[:N_PARAMS]
    return grouped

def main():
    print(f"Processing {MODEL_NAME} Stage 2…")
    stage2 = np.load(STAGE2_DATA_PATH)
    summary_map = load_summary_stats()
    model = load_stage1_model()

    grouped = group_stage2_data(stage2)

    priors, feats, pris, trues = [], [], [], []
    for sid, grp in grouped.items():
        stats = summary_map[sid]
        with torch.no_grad():
            prior = model(torch.tensor(stats, dtype=torch.float32).unsqueeze(0))\
                        .squeeze(0).numpy()
        priors.append(prior)
        feats.append(np.array(grp["features"]))
        pris.append(np.array(grp["prices"]))
        trues.append(grp["true_params"])

    np.savez(OUTPUT_PATH,
             priors=   np.array(priors),
             option_features=np.array(feats),
             option_prices=  np.array(pris),
             true_params=    np.array(trues))
    print(f"Saved grouped Stage 2 with priors to:\n  {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
