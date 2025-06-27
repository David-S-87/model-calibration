# synthetic_data1.py

import os
import numpy as np
from tqdm import tqdm

from config_template import sample_param_set, get_param_names
from price_paths import simulate_gbm, simulate_mjd, simulate_heston, simulate_bates
from summary_stats import compute_summary_stats

# Map model types to simulation functions
_SIM_FUNCS = {
    'gbm': simulate_gbm,
    'mjd': simulate_mjd,
    'heston': simulate_heston,
    'bates': simulate_bates
}

# Define the fixed order of summary stats keys (as in compute_summary_stats)
_SUMMARY_KEYS = [
    'mean_return', 'realized_volatility', 'skewness', 'kurtosis',
    'max_drawdown', 'jump_count', 'mean_jump_size', 'autocorr_lag1',
    'max_abs_return', 'autocorr_sq_lag1', 'realized_vol_of_vol', 'jump_std_ratio'
]


def generate_stage1_dataset(
    model_type: str,
    n_param_sets: int,
    n_paths_per_set: int,
    save_path: str,
    seed: int = None
) -> None:
    """
    Generate and save Stage 1 dataset for a given model.

    Parameters:
    - model_type: 'gbm', 'mjd', 'heston', or 'bates'
    - n_param_sets: number of distinct parameter sets to sample
    - n_paths_per_set: number of simulated paths per parameter set
    - save_path: file path for .npz output
    - seed: random seed for reproducibility

    Outputs (.npz):
    - summary_stats: shape (N, n_stats)
    - true_params: shape (N, n_params)
    - param_names: list of parameter names
    """
    rng = np.random.default_rng(seed)
    model_type = model_type.lower()
    if model_type not in _SIM_FUNCS:
        raise ValueError(f"Unknown model_type: {model_type}")

    sim_func = _SIM_FUNCS[model_type]
    param_names = get_param_names(model_type)
    n_stats = len(_SUMMARY_KEYS)
    n_params = len(param_names)

    total_samples = n_param_sets * n_paths_per_set
    X = np.zeros((total_samples, n_stats), dtype=float)
    y = np.zeros((total_samples, n_params), dtype=float)

    idx = 0
    # Generate data
    for i in tqdm(range(n_param_sets), desc=f"Generating Stage1 {model_type}"):
        cfg = sample_param_set(model_type, seed=rng.integers(1e9))
        S0 = cfg['S0']
        T = cfg['T']
        r = cfg['r']  # not used in Stage1, but sampled
        params = cfg['model_params']
        n_days = int(T * 252)

        # Simulate multiple paths for this parameter set
        # Paths shape: (n_paths_per_set, n_days+1)
        paths = sim_func(
            n_paths_per_set,
            n_days + 1,
            S0,
            params,
            seed=rng.integers(1e9)
        )

        # Compute stats and fill arrays
        for p in range(n_paths_per_set):
            stats = compute_summary_stats(paths[p, :])
            X[idx, :] = [stats[key] for key in _SUMMARY_KEYS]
            y[idx, :] = [params[name] for name in param_names]
            idx += 1

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save dataset
    np.savez(
        save_path,
        summary_stats=X,
        true_params=y,
        param_names=np.array(param_names)
    )

