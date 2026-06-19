"""L0 — raw MT5 payload normalisation (terminal-free)."""

from __future__ import annotations

import pandas as pd

from aurax.l0_data import rates_to_frame, ticks_to_frame
from aurax.l0_data.schema import BAR_COLUMNS


def test_rates_to_frame_normalises_and_sorts():
    # MT5 returns epoch-second 'time' (bar open) and 'tick_volume'.
    raw = [
        {"time": 1_577_854_800, "open": 1.1, "high": 1.2, "low": 1.05, "close": 1.15, "tick_volume": 100, "spread": 8},
        {"time": 1_577_840_400, "open": 1.0, "high": 1.1, "low": 0.95, "close": 1.08, "tick_volume": 120, "spread": 7},
    ]
    df = rates_to_frame(raw)

    assert list(df.columns) == BAR_COLUMNS
    assert df.index.is_monotonic_increasing  # sorted ascending
    assert df.index.name == "ts"
    assert df.index.tz is not None  # UTC-aware
    assert df["volume"].iloc[0] == 120  # tick_volume -> volume, earliest first


def test_ticks_to_frame_prefers_msc_and_adds_spread():
    raw = [
        {"time": 1_577_840_400, "time_msc": 1_577_840_400_500, "bid": 1.0, "ask": 1.0002, "last": 1.0001, "volume": 1},
    ]
    df = ticks_to_frame(raw)

    assert "spread" in df.columns
    assert abs(df["spread"].iloc[0] - 0.0002) < 1e-9
    assert df.index[0] == pd.Timestamp(1_577_840_400_500, unit="ms", tz="UTC")


def test_empty_inputs_return_typed_empty_frames():
    assert rates_to_frame([]).empty
    assert list(rates_to_frame([]).columns) == BAR_COLUMNS
    assert ticks_to_frame([]).empty
