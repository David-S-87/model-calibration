# generate.py

import sys
import os

# Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from common.synthetic_data2 import generate_stage2_dataset

if __name__ == "__main__":
    generate_stage2_dataset(
        model_type="gbm",
        n_param_sets=1000,
        n_options=100,
        n_mc_paths=5000,
        save_path=r"C:\Users\david\BathUni\MA50290_24\model_calibration\data\synthetic_data2\GBM\stage2_gbm_dataset.npz",
        seed=42
    )

