"""Session block — Asian/London/NY dummies + session-relative volatility.

H4 bars behave very differently by session, so the model needs to know *when*
a bar prints and how active it is *relative to its own session's* typical
volatility. Membership is derived from the bar's open hour (the index); ranges
may overlap (London/NY), so dummies are non-exclusive while the relative-vol
baseline uses a single priority session (NY > London > Asian).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..enums import Session
from .base import FeatureBlock, ensure_ohlcv, log_returns

# Default UTC hour windows [start, end). Overridable via config.
DEFAULT_WINDOWS: dict[Session, tuple[int, int]] = {
    Session.ASIAN: (0, 8),
    Session.LONDON: (7, 16),
    Session.NY: (12, 21),
}


def _in_window(hour: pd.Index, start: int, end: int) -> np.ndarray:
    return np.asarray((hour >= start) & (hour < end))


def session_membership(
    index: pd.DatetimeIndex, windows: dict[Session, tuple[int, int]] | None = None
) -> pd.DataFrame:
    """Non-exclusive session dummies (a bar can be in London *and* NY)."""
    windows = windows or DEFAULT_WINDOWS
    hour = index.hour
    cols = {
        f"is_{sess.value}": _in_window(hour, start, end).astype(float)
        for sess, (start, end) in windows.items()
    }
    return pd.DataFrame(cols, index=index)


def primary_session(
    index: pd.DatetimeIndex, windows: dict[Session, tuple[int, int]] | None = None
) -> pd.Series:
    """Single session per bar by priority NY > London > Asian, else OFF."""
    windows = windows or DEFAULT_WINDOWS
    hour = index.hour
    result = np.full(len(index), Session.OFF.value, dtype=object)
    for sess in (Session.ASIAN, Session.LONDON, Session.NY):  # later wins → NY top priority
        start, end = windows[sess]
        result[_in_window(hour, start, end)] = sess.value
    return pd.Series(result, index=index, name="session")


class SessionBlock(FeatureBlock):
    """Session dummies + |return| relative to the session's rolling baseline."""

    name = "sess"

    def __init__(
        self,
        windows: dict[Session, tuple[int, int]] | None = None,
        session_relative_vol_window: int = 120,
    ) -> None:
        self.windows = windows or DEFAULT_WINDOWS
        self.session_relative_vol_window = session_relative_vol_window

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        ensure_ohlcv(bars)
        idx = bars.index
        out = session_membership(idx, self.windows)

        abs_ret = log_returns(bars["close"]).abs()
        sess = primary_session(idx, self.windows)
        # Baseline = mean |return| over PRIOR same-session bars (shift(1) → no leak).
        win = self.session_relative_vol_window
        baseline = abs_ret.groupby(sess.to_numpy()).transform(
            lambda x: x.shift(1).rolling(win, min_periods=max(5, win // 4)).mean()
        )
        out["rel_vol"] = abs_ret / baseline.replace(0.0, np.nan)
        return self._prefixed(out)
