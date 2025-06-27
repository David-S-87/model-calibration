# summary_stats.py

import numpy as np
from scipy.stats import skew, kurtosis


def _compute_returns(price_series: np.ndarray) -> np.ndarray:
    """
    Compute log-returns from price series.
    """
    return np.diff(np.log(price_series))


def _compute_max_drawdown(price_series: np.ndarray) -> float:
    """
    Compute the maximum drawdown of a price series.
    """
    cum_max = np.maximum.accumulate(price_series)
    drawdowns = (cum_max - price_series) / cum_max
    return float(np.max(drawdowns))


def _autocorr(x: np.ndarray, lag: int = 1) -> float:
    """
    Compute lag-n autocorrelation of a sequence.
    """
    x = np.asarray(x)
    n = len(x)
    if n <= lag:
        return 0.0
    mean = x.mean()
    num = np.sum((x[:-lag] - mean) * (x[lag:] - mean))
    denom = np.sum((x - mean) ** 2)
    return float(num / denom) if denom != 0 else 0.0


def compute_summary_stats(
    price_series: np.ndarray,
    jump_thresh: float = 3.0,
    vol_of_vol_window: int = 20
) -> dict:
    """
    Compute a set of summary statistics from a 1D price series.

    Parameters:
    - price_series: 1D array of prices (length n).
    - jump_thresh: threshold in STD units for jump detection.
    - vol_of_vol_window: window size for realized vol-of-vol.

    Returns:
    - stats: dict with keys:
      'mean_return', 'realized_volatility', 'skewness', 'kurtosis',
      'max_drawdown', 'jump_count', 'mean_jump_size', 'autocorr_lag1',
      'max_abs_return', 'autocorr_sq_lag1', 'realized_vol_of_vol', 'jump_std_ratio'
    """
    # Compute returns
    returns = _compute_returns(price_series)
    n = len(returns)
    if n == 0:
        raise ValueError("Price series must have at least two points.")

    # Basic moments
    mean_return = float(np.mean(returns))
    realized_volatility = float(np.std(returns, ddof=1))
    skewness = float(skew(returns, bias=False)) if n > 2 else 0.0
    kurt = float(kurtosis(returns, fisher=False, bias=False)) if n > 3 else 0.0

    # Drawdown
    max_drawdown = _compute_max_drawdown(price_series)

    # Jump statistics
    jump_mask = np.abs(returns) > jump_thresh * realized_volatility
    jump_count = int(np.sum(jump_mask))
    mean_jump_size = float(np.mean(np.abs(returns[jump_mask]))) if jump_count > 0 else 0.0

    # Autocorrelations
    autocorr_lag1 = _autocorr(returns, lag=1)
    autocorr_sq_lag1 = _autocorr(returns**2, lag=1)

    # Extreme returns
    max_abs_return = float(np.max(np.abs(returns)))

    # Realized vol-of-vol
    if n >= vol_of_vol_window:
        rolling_vol = [
            np.std(returns[i : i + vol_of_vol_window], ddof=1)
            for i in range(n - vol_of_vol_window + 1)
        ]
        realized_vol_of_vol = float(np.std(rolling_vol, ddof=1))
    else:
        realized_vol_of_vol = 0.0

    # Jump vs non-jump volatility ratio
    std_jumps = np.std(returns[jump_mask], ddof=1) if jump_count > 1 else 0.0
    std_non_jumps = np.std(returns[~jump_mask], ddof=1) if (n - jump_count) > 1 else 0.0
    jump_std_ratio = float(std_jumps / std_non_jumps) if std_non_jumps > 0 else 0.0

    # Aggregate stats
    stats = {
        "mean_return": mean_return,
        "realized_volatility": realized_volatility,
        "skewness": skewness,
        "kurtosis": kurt,
        "max_drawdown": max_drawdown,
        "jump_count": jump_count,
        "mean_jump_size": mean_jump_size,
        "autocorr_lag1": autocorr_lag1,
        "max_abs_return": max_abs_return,
        "autocorr_sq_lag1": autocorr_sq_lag1,
        "realized_vol_of_vol": realized_vol_of_vol,
        "jump_std_ratio": jump_std_ratio,
    }

    return stats