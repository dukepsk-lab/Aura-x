"""L1 — feature blocks + point-in-time discipline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurax.enums import Session
from aurax.l1_features import (
    FeaturePipeline,
    atr,
    build_feature_matrix,
    kaufman_efficiency_ratio,
    yang_zhang,
)
from aurax.l1_features.session import primary_session, session_membership
from aurax.l1_features.trend_memory import rolling_hurst


def test_atr_is_positive_and_wilder_smoothed(random_walk_bars):
    a = atr(random_walk_bars, 14)
    assert a.dropna().gt(0).all()
    assert a.iloc[:13].isna().all()  # min_periods = window
    assert a.notna().iloc[14:].all()


def test_yang_zhang_and_ker_ranges(random_walk_bars):
    yz = yang_zhang(random_walk_bars, 24)
    assert yz.dropna().ge(0).all()
    ker = kaufman_efficiency_ratio(random_walk_bars["close"], 14)
    valid = ker.dropna()
    assert valid.between(0.0, 1.0).all()


def test_hurst_detects_trend(trending_bars, random_walk_bars):
    trend_h = rolling_hurst(trending_bars["close"], window=100).dropna().mean()
    # A persistent drift should read as strongly persistent (H > 0.5).
    assert trend_h > 0.5


def test_session_membership_and_priority():
    idx = pd.DatetimeIndex(
        ["2020-01-01T00:00", "2020-01-01T08:00", "2020-01-01T12:00", "2020-01-01T20:00"], tz="UTC"
    )
    dummies = session_membership(idx)
    # 12:00 is in both London (7-16) and NY (12-21).
    assert dummies.loc["2020-01-01T12:00", "is_london"] == 1.0
    assert dummies.loc["2020-01-01T12:00", "is_ny"] == 1.0
    prio = primary_session(idx)
    assert prio.tolist() == [Session.ASIAN.value, Session.LONDON.value, Session.NY.value, Session.NY.value]


def test_pipeline_columns_and_crosspair(random_walk_bars, partner_bars, params):
    matrix = build_feature_matrix(
        random_walk_bars, params, partner_close=partner_bars["close"]
    )
    assert len(matrix) == len(random_walk_bars)
    # blocks are prefixed
    assert any(c.startswith("vol_atr_") for c in matrix.columns)
    assert "tm_hurst" in matrix.columns and "tm_ker" in matrix.columns
    assert "sess_rel_vol" in matrix.columns
    assert "xpair_corr" in matrix.columns
    corr = matrix["xpair_corr"].dropna()
    assert corr.between(-1.0, 1.0).all()


def test_point_in_time_features_do_not_use_the_future(random_walk_bars, params):
    """Feature at bar t must be identical whether or not future bars exist."""
    full = FeaturePipeline.from_params(params).compute(random_walk_bars)
    k = 400
    truncated = FeaturePipeline.from_params(params).compute(random_walk_bars.iloc[:k])

    row_full = full.iloc[k - 1]
    row_trunc = truncated.iloc[k - 1]
    both = row_full.notna() & row_trunc.notna()
    assert both.any()
    np.testing.assert_allclose(
        row_full[both].to_numpy(dtype=float),
        row_trunc[both].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )
