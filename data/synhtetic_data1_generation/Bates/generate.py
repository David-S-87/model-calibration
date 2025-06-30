# generate.py

"""
Generate Stage 1 synthetic dataset for Bates model.
Saves file to:
C:/Users/david/BathUni/MA50290_24/model_calibration/data/synthetic_data1/Bates/
"""

import sys
import os

# Dynamically add model_calibration/ to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from common.synthetic_data1 import generate_stage1_dataset

if __name__ == "__main__":
    generate_stage1_dataset(
        model_type="bates",
        n_param_sets=5000,
        n_paths_per_set=100,
        save_path=r"C:\Users\david\BathUni\MA50290_24\model_calibration\data\synthetic_data1\Bates\stage1_bates_dataset.npz",
        seed=42
    )
