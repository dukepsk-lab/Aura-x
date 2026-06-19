"""Shared fixtures: deterministic synthetic OHLCV for the H4 pipeline tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurax.config import load_params


def h4_index(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    """A clean H4 (4-hour) UTC index of length ``n``."""
    return pd.date_range(start=start, periods=n, freq="4h", tz="UTC")


def bars_from_close(close: np.ndarray, index: pd.DatetimeIndex, *, seed: int = 0) -> pd.DataFrame:
    """Build a valid OHLCV frame around a close path (high≥max(o,c), low≤min(o,c))."""
    rng = np.random.default_rng(seed)
    close = np.asarray(close, dtype=float)
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = rng.uniform(0.0002, 0.0010, size=close.size)
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.uniform(800, 1200, size=close.size)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def params() -> dict:
    """Merged YAML params from the repo ``config/`` directory."""
    return load_params()


@pytest.fixture
def random_walk_bars() -> pd.DataFrame:
    """A 600-bar seeded random-walk EURUSD-like series (~1.10 level)."""
    n = 600
    rng = np.random.default_rng(42)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0015, size=n))
    return bars_from_close(close, h4_index(n), seed=1)


@pytest.fixture
def trending_bars() -> pd.DataFrame:
    """A persistent uptrend (Hurst should read > 0.5)."""
    n = 400
    rng = np.random.default_rng(7)
    drift = np.linspace(0, 0.06, n)
    close = 1.10 + drift + np.cumsum(rng.normal(0, 0.0003, size=n))
    return bars_from_close(close, h4_index(n), seed=2)


@pytest.fixture
def partner_bars() -> pd.DataFrame:
    """A GBPUSD-like partner leg on the same index as ``random_walk_bars``."""
    n = 600
    rng = np.random.default_rng(99)
    close = 1.30 + np.cumsum(rng.normal(0, 0.0018, size=n))
    return bars_from_close(close, h4_index(n), seed=3)
