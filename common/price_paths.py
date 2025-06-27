# price_paths.py

import numpy as np

def simulate_gbm(n_paths: int, n_days: int, S0: float, params: dict, seed: int = None) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion paths.
    
    dS_t = mu * S_t dt + sigma * S_t dW_t
    
    Parameters:
    - n_paths: number of simulated paths
    - n_days: number of time steps per path
    - S0: initial price
    - params: {'mu': float, 'sigma': float}
    - seed: random seed
    
    Returns:
    - prices: ndarray of shape (n_paths, n_days)
    """
    mu = params['mu']
    sigma = params['sigma']
    dt = 1.0 / n_days
    
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_days - 1))
    
    # log-price increments
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    # build log-price array
    logS = np.zeros((n_paths, n_days))
    logS[:, 0] = np.log(S0)
    logS[:, 1:] = logS[:, [0]] + np.cumsum(increments, axis=1)
    
    return np.exp(logS)


def simulate_mjd(n_paths: int, n_days: int, S0: float, params: dict, seed: int = None) -> np.ndarray:
    """
    Simulate Merton Jump-Diffusion paths.
    
    dS_t = mu * S_t dt + sigma * S_t dW_t + S_{t-}(e^J - 1)dN_t
    
    Parameters:
    - params: {'mu', 'sigma', 'lambda', 'mu_J', 'sigma_J'}
    """
    mu = params['mu']
    sigma = params['sigma']
    lam = params['lambda']
    mu_j = params['mu_J']
    sigma_j = params['sigma_J']
    dt = 1.0 / n_days
    
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_paths, n_days - 1))
    N = rng.poisson(lam * dt, size=(n_paths, n_days - 1))
    Zj = rng.standard_normal((n_paths, n_days - 1))
    
    # Jump sum via compound Poisson: sum of N normals ~ Normal(N*mu_j, N*sigma_j^2)
    Jsum = N * mu_j + np.sqrt(N) * sigma_j * Zj
    jump_factor = np.exp(Jsum)  # =1 when N=0
    
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    total_inc = increments + np.log(jump_factor)
    
    logS = np.zeros((n_paths, n_days))
    logS[:, 0] = np.log(S0)
    logS[:, 1:] = logS[:, [0]] + np.cumsum(total_inc, axis=1)
    
    return np.exp(logS)


def simulate_heston(n_paths: int, n_days: int, S0: float, params: dict, seed: int = None) -> np.ndarray:
    """
    Simulate Heston stochastic volatility paths.
    
    dS_t = mu * S_t dt + sqrt(v_t) * S_t dW^S_t
    dv_t = kappa(theta - v_t)dt + sigma_v sqrt(v_t) dW^v_t
    corr(dW^S, dW^v) = rho dt
    """
    mu = params['mu']
    v0 = params['v0']
    kappa = params['kappa']
    theta = params['theta']
    sigma_v = params['sigma_v']
    rho = params['rho']
    dt = 1.0 / n_days
    
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_days - 1))
    Z2 = rng.standard_normal((n_paths, n_days - 1))
    
    # correlated Brownian increments
    dW_v = np.sqrt(dt) * Z1
    dW_s = rho * dW_v + np.sqrt(dt * (1 - rho**2)) * Z2
    
    v = np.zeros((n_paths, n_days))
    v[:, 0] = v0
    
    for t in range(1, n_days):
        vt_prev = np.maximum(v[:, t - 1], 0)
        v[:, t] = vt_prev + kappa * (theta - vt_prev) * dt + sigma_v * np.sqrt(vt_prev) * Z1[:, t - 1] * np.sqrt(dt)
        v[:, t] = np.maximum(v[:, t], 1e-8)
    
    # simulate log-prices
    logS = np.zeros((n_paths, n_days))
    logS[:, 0] = np.log(S0)
    sqrt_v = np.sqrt(v[:, :-1])
    increments = (mu - 0.5 * v[:, :-1]) * dt + sqrt_v * dW_s
    
    logS[:, 1:] = logS[:, [0]] + np.cumsum(increments, axis=1)
    
    return np.exp(logS)


def simulate_bates(n_paths: int, n_days: int, S0: float, params: dict, seed: int = None) -> np.ndarray:
    """
    Simulate Bates model (Heston + jumps).
    """
    # Extract Heston + jump params
    mu = params['mu']
    v0 = params['v0']
    kappa = params['kappa']
    theta = params['theta']
    sigma_v = params['sigma_v']
    rho = params['rho']
    lam = params['lambda']
    mu_j = params['mu_J']
    sigma_j = params['sigma_J']
    dt = 1.0 / n_days
    
    rng = np.random.default_rng(seed)
    Z1 = rng.standard_normal((n_paths, n_days - 1))
    Z2 = rng.standard_normal((n_paths, n_days - 1))
    N = rng.poisson(lam * dt, size=(n_paths, n_days - 1))
    Zj = rng.standard_normal((n_paths, n_days - 1))
    
    # correlated Brownian increments
    dW_v = np.sqrt(dt) * Z1
    dW_s = rho * dW_v + np.sqrt(dt * (1 - rho**2)) * Z2
    
    # jump increments
    Jsum = N * mu_j + np.sqrt(N) * sigma_j * Zj
    jump_factor = np.exp(Jsum)
    
    # initialize
    v = np.zeros((n_paths, n_days))
    v[:, 0] = v0
    logS = np.zeros((n_paths, n_days))
    logS[:, 0] = np.log(S0)
    
    # simulate
    for t in range(1, n_days):
        vt_prev = np.maximum(v[:, t - 1], 0)
        # volatility update
        v[:, t] = vt_prev + kappa * (theta - vt_prev) * dt + sigma_v * np.sqrt(vt_prev) * Z1[:, t - 1] * np.sqrt(dt)
        v[:, t] = np.maximum(v[:, t], 1e-8)
        # price increment: Heston part + jump
        inc_heston = (mu - 0.5 * vt_prev) * dt + np.sqrt(vt_prev) * dW_s[:, t - 1]
        logS[:, t] = logS[:, t - 1] + inc_heston + np.log(jump_factor[:, t - 1])
    
    return np.exp(logS)