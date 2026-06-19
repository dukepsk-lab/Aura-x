"""Validation metrics — cost-adjusted, baseline-relative, multiple-testing-honest.

Every metric is reported *against* a baseline (majority-class / Buy-and-Hold /
random-entry) and *after* realistic costs. The Deflated Sharpe Ratio discounts
for the number of configurations tried — the source PPO "edge" looked far weaker
once read against its own Buy-and-Hold baseline, so honesty is wired in here.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

_EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int | None = None) -> float:
    """Sharpe of per-observation returns; annualised if ``periods_per_year`` given."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    return sr * np.sqrt(periods_per_year) if periods_per_year else sr


def sortino_ratio(
    returns: np.ndarray, periods_per_year: int | None = None, target: float = 0.0
) -> float:
    """Sortino — like Sharpe but penalising only downside deviation below ``target``."""
    r = np.asarray(returns, dtype=float)
    if r.size < 2:
        return float("nan")
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd == 0:
        return float("nan")
    sr = (r.mean() - target) / dd
    return sr * np.sqrt(periods_per_year) if periods_per_year else sr


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of the additive equity curve (≤ 0)."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return 0.0
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def cost_adjusted_returns(
    gross: np.ndarray,
    *,
    spread_cost: np.ndarray | float = 0.0,
    commission: np.ndarray | float = 0.0,
    slippage: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Subtract modeled spread + commission + slippage. An edge that only exists
    gross is not an edge."""
    return np.asarray(gross, dtype=float) - spread_cost - commission - slippage


def probabilistic_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    benchmark: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR — probability the true (per-obs) Sharpe exceeds ``benchmark`` (AFML)."""
    if n_obs < 2:
        return float("nan")
    denom = np.sqrt(1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2)
    if denom == 0:
        return float("nan")
    return float(norm.cdf((sharpe - benchmark) * np.sqrt(n_obs - 1) / denom))


def expected_max_sharpe(sharpe_std: float, n_trials: int) -> float:
    """Expected maximum Sharpe under ``n_trials`` independent configurations.

    With a single trial there is no multiple-testing inflation, so the expected
    maximum is 0 (and ``norm.ppf(0)`` is avoided).
    """
    if n_trials <= 1 or sharpe_std <= 0:
        return 0.0
    g = _EULER_MASCHERONI
    e = np.e
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * e))
    return float(sharpe_std * ((1.0 - g) * z1 + g * z2))


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    n_trials: int,
    sharpe_std: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR — PSR against the expected-max Sharpe from ``n_trials`` (AFML).

    ``sharpe``/``sharpe_std`` are per-observation; ``sharpe_std`` is the spread
    of Sharpe ratios across the configurations tried. Clearing DSR > 0.95 is the
    multiple-testing-honest bar for "this edge is real".
    """
    benchmark = expected_max_sharpe(sharpe_std, n_trials)
    return probabilistic_sharpe_ratio(
        sharpe, n_obs=n_obs, benchmark=benchmark, skew=skew, kurtosis=kurtosis
    )


def baseline_relative(strategy: float, baseline: float) -> float:
    """Excess of a strategy metric over its baseline (never report in isolation)."""
    return strategy - baseline
