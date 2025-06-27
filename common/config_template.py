# config.py

import numpy as np

def sample_gbm_params(seed=None):
    """
    Sample parameters for GBM model.
    Returns dict with:
      - model_params: {'mu', 'sigma'}
      - S0: initial price
      - T: time horizon (years)
      - r: risk-free rate
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.15)
    sigma = rng.uniform(0.1, 0.5)
    S0 = rng.uniform(50.0, 200.0)
    T = rng.uniform(0.5, 2.0)
    r = rng.uniform(0.01, 0.05)
    return {
        'model_params': {'mu': mu, 'sigma': sigma},
        'S0': S0,
        'T': T,
        'r': r
    }

def sample_mjd_params(seed=None):
    """
    Sample parameters for Merton Jump-Diffusion model.
    Returns dict with:
      - model_params: {'mu', 'sigma', 'lambda', 'mu_J', 'sigma_J'}
      - S0, T, r as above
    """
    rng = np.random.default_rng(seed)
    base = sample_gbm_params(seed)
    lam = rng.uniform(0.05, 0.3)
    mu_J = rng.uniform(-0.05, 0.0)
    sigma_J = rng.uniform(0.01, 0.1)
    base['model_params'].update({'lambda': lam, 'mu_J': mu_J, 'sigma_J': sigma_J})
    return base

def sample_heston_params(seed=None):
    """
    Sample parameters for Heston model.
    Returns dict with:
      - model_params: {'mu', 'v0', 'theta', 'kappa', 'sigma_v', 'rho'}
      - S0, T, r
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.02, 0.15)
    v0 = rng.uniform(0.01, 0.09)
    theta = rng.uniform(0.01, 0.09)
    kappa = rng.uniform(0.5, 3.0)
    sigma_v = rng.uniform(0.1, 0.6)
    rho = rng.uniform(-0.9, -0.1)
    S0 = rng.uniform(50.0, 200.0)
    T = rng.uniform(0.5, 2.0)
    r = rng.uniform(0.01, 0.05)
    return {
        'model_params': {
            'mu': mu,
            'v0': v0,
            'theta': theta,
            'kappa': kappa,
            'sigma_v': sigma_v,
            'rho': rho
        },
        'S0': S0,
        'T': T,
        'r': r
    }

def sample_bates_params(seed=None):
    """
    Sample parameters for Bates model (Heston + jumps).
    Returns dict with:
      - model_params: Heston params + {'lambda', 'mu_J', 'sigma_J'}
      - S0, T, r
    """
    rng = np.random.default_rng(seed)
    base = sample_heston_params(seed)
    lam = rng.uniform(0.05, 0.3)
    mu_J = rng.uniform(-0.05, 0.0)
    sigma_J = rng.uniform(0.01, 0.1)
    base['model_params'].update({'lambda': lam, 'mu_J': mu_J, 'sigma_J': sigma_J})
    return base

def sample_param_set(model_type: str, seed: int = None) -> dict:
    """
    Wrapper to sample a parameter set for the given model.
    model_type: 'gbm', 'mjd', 'heston', or 'bates'
    """
    model_type = model_type.lower()
    if model_type == "gbm":
        return sample_gbm_params(seed)
    elif model_type == "mjd":
        return sample_mjd_params(seed)
    elif model_type == "heston":
        return sample_heston_params(seed)
    elif model_type == "bates":
        return sample_bates_params(seed)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

def get_param_names(model_type: str):
    """
    Return ordered list of parameter names for label vectors.
    """
    model_type = model_type.lower()
    if model_type == "gbm":
        return ['mu', 'sigma']
    elif model_type == "mjd":
        return ['mu', 'sigma', 'lambda', 'mu_J', 'sigma_J']
    elif model_type == "heston":
        return ['mu', 'v0', 'theta', 'kappa', 'sigma_v', 'rho']
    elif model_type == "bates":
        return ['mu', 'v0', 'theta', 'kappa', 'sigma_v', 'rho', 'lambda', 'mu_J', 'sigma_J']
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
