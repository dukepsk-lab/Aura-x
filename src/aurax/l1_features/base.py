"""Feature-engineering primitives and the ``FeatureBlock`` interface.

**Point-in-time discipline is the contract of this layer.** Every feature at bar
``t`` is computed only from bars with index ``<= t`` (data available at *that*
bar's close). Concretely that means:

* rolling windows are trailing (pandas ``.rolling`` is left-closed/trailing);
* we never call ``.shift(-k)`` or any forward-looking transform here — that is
  the exclusive job of Layer 4 (labeling);
* the True Range / overnight terms use the *previous* close (``.shift(1)``),
  which is known at ``t``.

A :class:`FeatureBlock` maps an OHLCV frame to a frame of named features sharing
the same index. Blocks are pure and stateless so they compose freely in the
:class:`~aurax.l1_features.pipeline.FeaturePipeline`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

OHLCV = ("open", "high", "low", "close", "volume")


class FeatureError(ValueError):
    """Raised when a feature block receives malformed input."""


def ensure_ohlcv(bars: pd.DataFrame, *, require_volume: bool = False) -> None:
    """Validate that ``bars`` is a sorted, OHLC(V) frame with a DatetimeIndex."""
    needed = OHLCV if require_volume else OHLCV[:4]
    missing = [c for c in needed if c not in bars.columns]
    if missing:
        raise FeatureError(f"bars frame missing columns: {missing}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise FeatureError("bars must be indexed by a DatetimeIndex (bar timestamps)")
    if not bars.index.is_monotonic_increasing:
        raise FeatureError("bars index must be sorted ascending (point-in-time order)")


def log_returns(close: pd.Series) -> pd.Series:
    """Close-to-close log returns (trailing; first value is NaN)."""
    return np.log(close).diff()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder True Range = max(H−L, |H−C_prev|, |L−C_prev|).

    Uses the *previous* close (``shift(1)``) → known at the current bar's close.
    """
    prev_close = close.shift(1)
    ranges = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing (RMA): EMA with alpha = 1/window."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Trailing z-score: (x − rolling_mean) / rolling_std."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=1)
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Trailing percentile rank in [0, 1] of the latest value within the window.

    Fraction of the trailing ``window`` observations ``<=`` the current value —
    used for ATR-percentile regime context.
    """

    def _rank(x: np.ndarray) -> float:
        return float((x <= x[-1]).mean())

    return series.rolling(window).apply(_rank, raw=True)


class FeatureBlock(ABC):
    """A pure, stateless mapping: OHLCV frame → named feature frame."""

    #: short prefix applied to every output column, e.g. ``atr_14``.
    name: str = "feature"

    @abstractmethod
    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Return a feature frame sharing ``bars``' index."""

    def _prefixed(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Prefix columns with the block name (idempotent if already prefixed)."""
        return frame.rename(
            columns={c: (c if c.startswith(f"{self.name}_") else f"{self.name}_{c}") for c in frame.columns}
        )
