"""Volatility block — ATR (multi-lookback), Yang-Zhang, realized vol, ATR-pct.

These are the backbone of the ATR pillar: they feed barrier scaling (L4),
position sizing (L6) and the shock-regime gate (L2). All trailing / point-in-time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import (
    FeatureBlock,
    ensure_ohlcv,
    log_returns,
    rolling_percentile,
    true_range,
    wilder_smooth,
)


def atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder-smoothed). Shared by L4 barrier scaling & L6."""
    return wilder_smooth(true_range(bars["high"], bars["low"], bars["close"]), window)


def realized_volatility(
    close: pd.Series, window: int = 24, periods_per_year: int | None = None
) -> pd.Series:
    """Rolling std of log returns, optionally annualised by ``sqrt(periods)``."""
    vol = log_returns(close).rolling(window).std(ddof=1)
    if periods_per_year:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def yang_zhang(bars: pd.DataFrame, window: int = 24) -> pd.Series:
    """Yang-Zhang volatility — drift-independent, handles overnight gaps.

    YZ² = σ_overnight² + k·σ_open-close² + (1−k)·σ_Rogers-Satchell²,
    with k = 0.34 / (1.34 + (n+1)/(n−1)).
    """
    log_o = np.log(bars["open"])
    log_h = np.log(bars["high"])
    log_l = np.log(bars["low"])
    log_c = np.log(bars["close"])

    overnight = log_o - log_c.shift(1)          # close(t-1) → open(t)
    open_close = log_c - log_o                  # open(t) → close(t)
    u = log_h - log_o
    d = log_l - log_o
    rogers_satchell = u * (u - open_close) + d * (d - open_close)

    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    sigma_o2 = overnight.rolling(n).var(ddof=1)
    sigma_c2 = open_close.rolling(n).var(ddof=1)
    sigma_rs2 = rogers_satchell.rolling(n).mean()
    return np.sqrt(sigma_o2 + k * sigma_c2 + (1.0 - k) * sigma_rs2)


class VolatilityBlock(FeatureBlock):
    """ATR (several lookbacks) + ATR-percentile + Yang-Zhang + realized vol."""

    name = "vol"

    def __init__(
        self,
        atr_lookbacks: list[int] | None = None,
        atr_percentile_window: int = 252,
        yang_zhang_window: int = 24,
        realized_vol_window: int = 24,
        realized_vol_annualization: int | None = 1512,
    ) -> None:
        self.atr_lookbacks = atr_lookbacks or [14, 24, 48]
        self.atr_percentile_window = atr_percentile_window
        self.yang_zhang_window = yang_zhang_window
        self.realized_vol_window = realized_vol_window
        self.realized_vol_annualization = realized_vol_annualization

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        ensure_ohlcv(bars)
        out = pd.DataFrame(index=bars.index)

        for lb in self.atr_lookbacks:
            a = atr(bars, lb)
            out[f"atr_{lb}"] = a
            # ATR normalised by price → unit-free, comparable across regimes.
            out[f"atr_{lb}_pct_price"] = a / bars["close"]

        primary = self.atr_lookbacks[0]
        out["atr_pct"] = rolling_percentile(out[f"atr_{primary}"], self.atr_percentile_window)
        out["yang_zhang"] = yang_zhang(bars, self.yang_zhang_window)
        out["realized_vol"] = realized_volatility(
            bars["close"], self.realized_vol_window, self.realized_vol_annualization
        )
        return self._prefixed(out)
