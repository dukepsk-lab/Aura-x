"""Trend/memory block — Hurst exponent, Kaufman Efficiency Ratio, DFA.

These distinguish *trending* vs. *mean-reverting* structure (the FX translation
of "rhythmic vs. chaotic signal") and drive the Layer 2 regime router:
e.g. ``Hurst > 0.55 & high KER → trend regime``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlock, ensure_ohlcv


def kaufman_efficiency_ratio(close: pd.Series, window: int = 14) -> pd.Series:
    """KER ∈ [0, 1] = |net change| / Σ|bar-to-bar change| over ``window``.

    ≈1 → efficient/trending move; ≈0 → choppy/mean-reverting.
    """
    direction = close.diff(window).abs()
    volatility = close.diff().abs().rolling(window).sum()
    return direction / volatility.replace(0.0, np.nan)


def hurst_exponent(series: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """Generalized Hurst via the structure-function method on a (log-price) path.

    For a self-affine path the RMS of lag-``τ`` increments scales as ``τ^H``, so
    ``H`` is the slope of ``log(RMS) vs log(τ)``. ``H>0.5`` trending, ``H<0.5``
    mean-reverting, ``H≈0.5`` random walk. Returns ``nan`` if under-determined.
    """
    x = np.asarray(series, dtype=float)
    if x.size <= max_lag:
        return float("nan")
    lags = np.arange(min_lag, max_lag + 1)
    tau = np.array([np.sqrt(np.mean((x[lag:] - x[:-lag]) ** 2)) for lag in lags])
    mask = tau > 0
    if mask.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0])


def rolling_hurst(
    close: pd.Series, window: int = 100, min_lag: int = 2, max_lag: int = 20
) -> pd.Series:
    """Trailing Hurst over ``window`` bars, computed on log price."""
    log_price = np.log(close)
    return log_price.rolling(window).apply(
        lambda arr: hurst_exponent(arr, min_lag, max_lag), raw=True
    )


def dfa_hurst(series: np.ndarray, order: int = 1) -> float:
    """Detrended Fluctuation Analysis exponent (monofractal ≈ Hurst).

    Heavier than the structure-function estimate; used as the DFA member of the
    multifractal block. Disabled by default in v1 (cost vs. signal).
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 16:
        return float("nan")
    profile = np.cumsum(x - x.mean())
    scales = np.unique(np.floor(np.logspace(np.log10(8), np.log10(n // 4), 12)).astype(int))
    fluct, used = [], []
    for s in scales:
        if s < order + 2 or s > n // 2:
            continue
        n_seg = n // s
        t = np.arange(s)
        rms = []
        for i in range(n_seg):
            seg = profile[i * s : (i + 1) * s]
            trend = np.polyval(np.polyfit(t, seg, order), t)
            rms.append(np.mean((seg - trend) ** 2))
        fluct.append(np.sqrt(np.mean(rms)))
        used.append(s)
    if len(used) < 2:
        return float("nan")
    return float(np.polyfit(np.log(used), np.log(fluct), 1)[0])


class TrendMemoryBlock(FeatureBlock):
    """Hurst + KER (+ optional DFA) — the regime-character features."""

    name = "tm"

    def __init__(
        self,
        hurst_window: int = 100,
        hurst_min_lag: int = 2,
        hurst_max_lag: int = 20,
        ker_window: int = 14,
        mfdfa_window: int = 256,
        mfdfa_enabled: bool = False,
    ) -> None:
        self.hurst_window = hurst_window
        self.hurst_min_lag = hurst_min_lag
        self.hurst_max_lag = hurst_max_lag
        self.ker_window = ker_window
        self.mfdfa_window = mfdfa_window
        self.mfdfa_enabled = mfdfa_enabled

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        ensure_ohlcv(bars)
        out = pd.DataFrame(index=bars.index)
        out["ker"] = kaufman_efficiency_ratio(bars["close"], self.ker_window)
        out["hurst"] = rolling_hurst(
            bars["close"], self.hurst_window, self.hurst_min_lag, self.hurst_max_lag
        )
        if self.mfdfa_enabled:
            out["dfa"] = (
                np.log(bars["close"]).rolling(self.mfdfa_window).apply(dfa_hurst, raw=True)
            )
        return self._prefixed(out)
