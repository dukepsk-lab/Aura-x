"""Validation — CPCV leakage guarantees + deflated-Sharpe honesty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurax.validation import (
    CombinatorialPurgedCV,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def _overlapping_t1(n: int = 30, span: int = 3) -> pd.Series:
    """Each label at bar i ends ``span`` bars later → overlapping H4 labels."""
    idx = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    ends = [idx[min(i + span, n - 1)] for i in range(n)]
    return pd.Series(ends, index=idx)


def test_cpcv_split_count_and_disjoint():
    t1 = _overlapping_t1()
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo_pct=0.0)
    splits = list(cv.split(t1))
    assert len(splits) == cv.n_splits == 10
    for train_pos, test_pos in splits:
        assert set(train_pos).isdisjoint(set(test_pos))  # never train on test bars


def test_cpcv_purges_overlapping_labels():
    """No surviving train label may overlap any test label in time."""
    t1 = _overlapping_t1()
    starts = t1.index
    cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2, embargo_pct=0.0)
    for train_pos, test_pos in cv.split(t1):
        test_spans = [(starts[p], t1.iloc[p]) for p in test_pos]
        for p in train_pos:
            a0, a1 = starts[p], t1.iloc[p]
            for b0, b1 in test_spans:
                assert a1 < b0 or a0 > b1  # disjoint in time → leakage purged


def test_cpcv_embargo_drops_post_test_neighbours():
    t1 = _overlapping_t1(n=40, span=1)
    no_emb = CombinatorialPurgedCV(5, 1, embargo_pct=0.0)
    emb = CombinatorialPurgedCV(5, 1, embargo_pct=0.10)
    n_no = sum(len(tr) for tr, _ in no_emb.split(t1))
    n_emb = sum(len(tr) for tr, _ in emb.split(t1))
    assert n_emb < n_no  # embargo removes additional train rows


def test_deflated_sharpe_discounts_for_trials():
    sr = 0.10  # per-observation Sharpe
    psr = probabilistic_sharpe_ratio(sr, n_obs=1000, benchmark=0.0)
    dsr = deflated_sharpe_ratio(sr, n_obs=1000, n_trials=50, sharpe_std=0.05)
    assert 0.0 <= dsr <= 1.0
    assert dsr < psr  # accounting for 50 trials lowers confidence
    # expected max Sharpe grows with the number of trials
    assert expected_max_sharpe(0.05, 100) > expected_max_sharpe(0.05, 10)


def test_sharpe_ratio_basic():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, size=500)
    assert np.isfinite(sharpe_ratio(r))
    assert np.isnan(sharpe_ratio(np.array([0.01])))  # too few points
