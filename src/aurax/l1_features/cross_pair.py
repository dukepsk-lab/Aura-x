"""Cross-pair block — the cross-variate insight applied to two correlated majors.

EURUSD and GBPUSD share USD risk; their rolling correlation, log-spread
mean-reversion and lead-lag carry information neither series has alone. Feeds the
Layer 6 correlation cap as well as the primary model.

Because a block sees only its own ``bars``, the partner's close is injected at
construction and aligned by exact timestamp join (no forward-fill → no leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlock, ensure_ohlcv, log_returns, rolling_zscore


def rolling_correlation(ret_a: pd.Series, ret_b: pd.Series, window: int) -> pd.Series:
    """Trailing Pearson correlation of two return series."""
    return ret_a.rolling(window).corr(ret_b)


def log_spread(close_a: pd.Series, close_b: pd.Series) -> pd.Series:
    """log(A) − log(B): the (near-)stationary spread of two cointegrated majors."""
    return np.log(close_a) - np.log(close_b)


class CrossPairBlock(FeatureBlock):
    """Rolling correlation + spread z-score + 1-bar lead-lag vs. a partner leg."""

    name = "xpair"

    def __init__(
        self,
        partner_close: pd.Series,
        corr_window: int = 48,
        ratio_zscore_window: int = 96,
        lead_lag_max_shift: int = 6,
    ) -> None:
        self.partner_close = partner_close
        self.corr_window = corr_window
        self.ratio_zscore_window = ratio_zscore_window
        self.lead_lag_max_shift = lead_lag_max_shift

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        ensure_ohlcv(bars)
        out = pd.DataFrame(index=bars.index)

        # Exact-timestamp align (same H4 grid); missing partner bars stay NaN.
        partner = self.partner_close.reindex(bars.index)
        ret_a = log_returns(bars["close"])
        ret_b = log_returns(partner)

        out["corr"] = rolling_correlation(ret_a, ret_b, self.corr_window)
        out["spread_z"] = rolling_zscore(
            log_spread(bars["close"], partner), self.ratio_zscore_window
        )
        # Lead-lag: does the partner's *previous* return inform this bar?
        # (shift(+k) uses past partner data only → point-in-time safe).
        out["lead_lag_corr"] = rolling_correlation(ret_a, ret_b.shift(1), self.corr_window)

        # Strongest past lead-lag magnitude over k = 1..max_shift (regime context).
        best = None
        for k in range(1, self.lead_lag_max_shift + 1):
            c = rolling_correlation(ret_a, ret_b.shift(k), self.corr_window)
            best = c if best is None else best.where(best.abs() >= c.abs(), c)
        out["lead_lag_best"] = best
        return self._prefixed(out)
