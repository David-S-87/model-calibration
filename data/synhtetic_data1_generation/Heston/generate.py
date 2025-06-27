# generate.py

"""
Generate Stage 1 synthetic dataset for Heston model.
Saves file to:
C:/Users/david/BathUni/MA50290_24/model_calibration/data/synthetic_data1/Heston/
"""

import sys
import os

# Dynamically add model_calibration/ to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from common.synthetic_data1 import generate_stage1_dataset

if __name__ == "__main__":
    generate_stage1_dataset(
        model_type="heston",
        n_param_sets=1000,
        n_paths_per_set=10,
        save_path=r"C:\Users\david\BathUni\MA50290_24\model_calibration\data\synthetic_data1\Heston\stage1_heston_dataset.npz",
        seed=42
    )
