"""L4 — triple-barrier labels, sample-uniqueness weights, trend-scanning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurax.l4_labeling import (
    average_uniqueness,
    num_concurrent_events,
    sample_weights,
    trend_scanning_labels,
    triple_barrier_labels,
)

ATR = 0.0010  # constant ATR → barriers at ±2·ATR = ±0.0020 for mult=2


def _bars(o, h, l, c) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(c), freq="4h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def _flat(n: int):
    """n flat bars at 1.0000 with tight wicks well inside the barriers."""
    o = [1.0000] * n
    c = [1.0000] * n
    h = [1.0010] * n
    l = [0.9990] * n
    return o, h, l, c


def _atr(bars):
    return pd.Series(ATR, index=bars.index)


def test_triple_barrier_take_profit():
    o, h, l, c = _flat(6)
    h[2] = 1.0025  # spikes through TP (1.0020) on bar 2
    bars = _bars(o, h, l, c)
    out = triple_barrier_labels(bars, _atr(bars), tp_mult=2, sl_mult=2, vertical_bars=5,
                                events=bars.index[[0]])
    row = out.iloc[0]
    assert row["label"] == 1 and row["barrier"] == "tp"
    assert row["t1"] == bars.index[2]
    assert row["ret"] == pytest.approx(0.0020, abs=1e-9)  # exit at TP level


def test_triple_barrier_stop_loss():
    o, h, l, c = _flat(6)
    l[2] = 0.9975  # breaks SL (0.9980) on bar 2
    bars = _bars(o, h, l, c)
    out = triple_barrier_labels(bars, _atr(bars), tp_mult=2, sl_mult=2, vertical_bars=5,
                                events=bars.index[[0]])
    row = out.iloc[0]
    assert row["label"] == -1 and row["barrier"] == "sl"
    assert row["ret"] == pytest.approx(-0.0020, abs=1e-9)


def test_triple_barrier_vertical_timeout_is_neutral():
    o, h, l, c = _flat(7)  # never touches either barrier within 5 bars
    bars = _bars(o, h, l, c)
    out = triple_barrier_labels(bars, _atr(bars), tp_mult=2, sl_mult=2, vertical_bars=5,
                                events=bars.index[[0]])
    row = out.iloc[0]
    assert row["label"] == 0 and row["barrier"] == "vertical"
    assert row["t1"] == bars.index[5]


def test_triple_barrier_tie_breaks_pessimistically():
    o, h, l, c = _flat(6)
    h[1], l[1] = 1.0025, 0.9975  # both barriers inside one bar
    bars = _bars(o, h, l, c)
    out = triple_barrier_labels(bars, _atr(bars), tp_mult=2, sl_mult=2, vertical_bars=5,
                                events=bars.index[[0]], tie_break="sl")
    assert out.iloc[0]["barrier"] == "sl"  # pessimistic tie-break


def test_triple_barrier_meta_mode_labels_wins():
    o, h, l, c = _flat(6)
    h[2] = 1.0025  # long TP hit
    bars = _bars(o, h, l, c)
    side = pd.Series(1.0, index=bars.index[[0]])
    out = triple_barrier_labels(bars, _atr(bars), tp_mult=2, sl_mult=2, vertical_bars=5,
                                events=bars.index[[0]], side=side)
    row = out.iloc[0]
    assert row["label"] == 1  # meta: take the bet (it won)
    assert row["side"] == 1.0


def test_skips_events_with_nan_atr():
    o, h, l, c = _flat(6)
    bars = _bars(o, h, l, c)
    a = _atr(bars).copy()
    a.iloc[0] = np.nan
    out = triple_barrier_labels(bars, a, vertical_bars=5, events=bars.index[[0]])
    assert out.empty


# --- sample uniqueness -------------------------------------------------------
def test_concurrency_and_uniqueness_exact():
    idx = pd.date_range("2020-01-01", periods=5, freq="4h", tz="UTC")
    # A spans bars 0..1; B spans bars 1..3 → bar1 is shared.
    t1 = pd.Series([idx[1], idx[3]], index=[idx[0], idx[1]])
    conc = num_concurrent_events(idx, t1)
    assert conc.loc[idx[1]] == 2.0 and conc.loc[idx[0]] == 1.0

    u = average_uniqueness(t1, conc)
    assert u.loc[idx[0]] == pytest.approx(0.75)            # mean(1/1, 1/2)
    assert u.loc[idx[1]] == pytest.approx((0.5 + 1 + 1) / 3)
    assert u.loc[idx[1]] > u.loc[idx[0]]                   # less-overlapped → more unique


def test_sample_weights_normalised_mean_one():
    idx = pd.date_range("2020-01-01", periods=5, freq="4h", tz="UTC")
    t1 = pd.Series([idx[1], idx[3]], index=[idx[0], idx[1]])
    w = sample_weights(t1, idx, apply_time_decay=False, normalize=True)
    assert w.mean() == pytest.approx(1.0)
    assert w.loc[idx[1]] > w.loc[idx[0]]


# --- trend scanning ----------------------------------------------------------
def test_trend_scanning_detects_up_and_flat():
    idx = pd.date_range("2020-01-01", periods=40, freq="4h", tz="UTC")
    up = pd.Series(np.exp(np.linspace(0, 0.5, 40)), index=idx)  # clean log-linear rise
    out = trend_scanning_labels(up, min_horizon=5, max_horizon=20, t_value_threshold=2.0,
                                events=idx[[0]])
    assert out.iloc[0]["label"] == 1
    assert out.iloc[0]["t_value"] > 2.0

    flat = pd.Series(np.full(40, 1.10), index=idx)
    out2 = trend_scanning_labels(flat, min_horizon=5, max_horizon=20, events=idx[[0]])
    assert out2.iloc[0]["label"] == 0
