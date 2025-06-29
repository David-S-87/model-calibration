# synthetic_data2.py

import os
import numpy as np
from tqdm import tqdm
from .config_template import sample_param_set, get_param_names
from .options import price_option_batch

def generate_stage2_dataset(
    model_type: str,
    n_param_sets: int,
    n_options: int,
    n_mc_paths: int,
    save_path: str,
    seed: int = None
) -> None:
    """
    Generate and save Stage 2 dataset for a given model.

    Parameters:
    - model_type: 'gbm', 'mjd', 'heston', or 'bates'
    - n_param_sets: number of distinct parameter sets to sample
    - n_options: total number of (strike, maturity) option points per set
    - n_mc_paths: number of Monte Carlo paths for pricing
    - save_path: file path for .npz output
    - seed: random seed for reproducibility

    Outputs (.npz):
    - option_prices: shape (n_param_sets, n_options)
    - true_params: shape (n_param_sets, n_params)
    - param_names: list of parameter names
    - strikes: shape (n_param_sets, n_options)
    - maturities: shape (n_param_sets, n_options)
    - S0: shape (n_param_sets,)
    """
    rng = np.random.default_rng(seed)
    model_type = model_type.lower()
    param_names = get_param_names(model_type)
    n_params = len(param_names)

    # determine grid dimensions
    n_strikes = int(np.sqrt(n_options))
    n_maturities = int(np.ceil(n_options / n_strikes))
    total_options = n_strikes * n_maturities

    # pre-allocate arrays
    option_prices = np.zeros((n_param_sets, total_options), dtype=float)
    true_params = np.zeros((n_param_sets, n_params), dtype=float)
    strikes_store = np.zeros((n_param_sets, total_options), dtype=float)
    maturities_store = np.zeros((n_param_sets, total_options), dtype=float)
    S0_store = np.zeros(n_param_sets, dtype=float)

    for i in tqdm(range(n_param_sets), desc=f"Generating Stage2 {model_type}"):
        # 1. sample parameters
        cfg = sample_param_set(model_type, seed=rng.integers(1e9))
        S0 = cfg['S0']
        r = cfg['r']
        params = cfg['model_params']

        # 2. build strike & maturity grids
        strikes = np.linspace(0.8 * S0, 1.2 * S0, n_strikes)
        maturities = np.linspace(0.1, 2.0, n_maturities)
        max_T = float(np.max(maturities))
        n_steps = int(max_T * 252) + 1

        # 3. price options in a batch
        price_mat = price_option_batch(
            model_type=model_type,
            option_type='call',
            S0=S0,
            K=strikes,
            T=maturities,
            r=r,
            params=params,
            n_paths=n_mc_paths,
            n_steps=n_steps,
            antithetic=True,
            seed=int(rng.integers(1e9))
        )  # shape (n_strikes, n_maturities)

        # 4. flatten results
        flat_prices = price_mat.flatten()[:total_options]
        option_prices[i, :] = flat_prices

        # 5. record true parameters
        true_params[i, :] = [params[name] for name in param_names]

        # 6. record grids per sample
        K_grid, T_grid = np.meshgrid(strikes, maturities, indexing='ij')
        strikes_store[i, :] = K_grid.flatten()
        maturities_store[i, :] = T_grid.flatten()

        # 7. record S0
        S0_store[i] = S0

    # ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # save to .npz
    np.savez(
        save_path,
        option_prices=option_prices,
        true_params=true_params,
        param_names=np.array(param_names),
        strikes=strikes_store,
        maturities=maturities_store,
        S0=S0_store
    )
