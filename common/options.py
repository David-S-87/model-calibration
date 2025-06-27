# options.py

import numpy as np
from scipy.stats import norm

# Import simulation functions
from .price_paths import (
    simulate_gbm,
    simulate_mjd,
    simulate_heston,
    simulate_bates,
)


def price_option(
    model_type: str,
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r: float,
    params: dict,
    n_paths: int = 10000,
    n_steps: int = None,
    antithetic: bool = True,
    seed: int = None
) -> float:
    """
    Price a single vanilla European option under the specified model.
    Dispatches to Black-Scholes (GBM) or Monte Carlo (others).
    """
    model_type = model_type.lower()
    option_type = option_type.lower()
    if n_steps is None:
        n_steps = int(T * 252)

    if model_type == "gbm":
        sigma = params.get('sigma')
        return black_scholes_price(S0, K, T, r, sigma, option_type)
    else:
        return monte_carlo_price(
            model_type, option_type, S0, K, T, r, params,
            n_paths, n_steps, antithetic, seed
        )


def price_option_batch(
    model_type: str,
    option_type: str,
    S0: float,
    K: np.ndarray,
    T: np.ndarray,
    r: float,
    params: dict,
    n_paths: int = 10000,
    antithetic: bool = True,
    seed: int = None
) -> np.ndarray:
    """
    Price a grid of vanilla European options for arrays of strikes K and maturities T.
    Returns an array of shape (len(K), len(T)).
    """
    model_type = model_type.lower()
    option_type = option_type.lower()

    # For GBM, vectorize BS formula
    if model_type == "gbm":
        sigma = params.get('sigma')
        # Broadcast sigma, S0, r across grid
        price_grid = np.zeros((len(K), len(T)))
        for i, Ki in enumerate(K):
            for j, Tj in enumerate(T):
                price_grid[i, j] = black_scholes_price(
                    S0, Ki, Tj, r, sigma, option_type
                )
        return price_grid

    # Monte Carlo for other models
    # Determine max steps for longest maturity
    steps = (T * 252).astype(int)
    max_steps = int(np.max(steps))

    # Simulate once up to max_steps
    prices = {
        'mjd': simulate_mjd,
        'heston': simulate_heston,
        'bates': simulate_bates
    }.get(model_type, None)

    if prices is None:
        raise ValueError(f"Unsupported model_type for batch pricing: {model_type}")

    paths = prices(n_paths // (2 if antithetic else 1), max_steps + 1, S0, params, seed=seed)
    if antithetic:
        # assume simulate_ returns symmetric paths when half-sample; append negated log paths
        # simplistic: concatenate same paths for demonstration; for true antithetic, implement in price_paths
        paths = np.vstack([paths, paths])

    # Compute discounted payoffs for each K, T
    price_grid = np.zeros((len(K), len(T)))
    for j, tj in enumerate(steps):
        ST = paths[:, tj]
        for i, Ki in enumerate(K):
            pay = payoff(ST, Ki, option_type)
            price_grid[i, j] = np.exp(-r * T[j]) * np.mean(pay)
    return price_grid


def black_scholes_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str
) -> float:
    """
    Closed-form Black-Scholes price for European options.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def monte_carlo_price(
    model_type: str,
    option_type: str,
    S0: float,
    K: float,
    T: float,
    r: float,
    params: dict,
    n_paths: int,
    n_steps: int,
    antithetic: bool,
    seed: int
) -> float:
    """
    Monte Carlo price for a single vanilla option.
    """
    # Choose simulation function
    sim_func = {
        'mjd': simulate_mjd,
        'heston': simulate_heston,
        'bates': simulate_bates
    }.get(model_type)
    if sim_func is None:
        raise ValueError(f"Unsupported model_type: {model_type}")

    paths = sim_func(n_paths // (2 if antithetic else 1), n_steps + 1, S0, params, seed=seed)
    if antithetic:
        paths = np.vstack([paths, paths])  # placeholder for true antithetic

    ST = paths[:, -1]
    pay = payoff(ST, K, option_type)
    return float(np.exp(-r * T) * np.mean(pay))


def payoff(S_T: np.ndarray, K: float, option_type: str) -> np.ndarray:
    """
    Vectorized payoff for European call/put given terminal prices.
    """
    if option_type == 'call':
        return np.maximum(S_T - K, 0)
    else:
        return np.maximum(K - S_T, 0)
