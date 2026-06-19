"""Validation harness — splitters, cost-adjusted backtest, baselines, Go/No-Go."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurax.validation import (
    CostModel,
    PurgedKFold,
    ValidationConfig,
    WalkForwardSplit,
    holdout_split,
    majority_class_side,
    oof_predict,
    run_validation,
    strategy_returns,
)


# --- a tiny, controllable estimator (no ML dependency) -----------------------
class CorrSignEstimator:
    """Pick the feature most correlated with y on fit; predict its (signed) sign."""

    def fit(self, X, y, sample_weight=None):
        best_c, best_abs, best_sign = X.columns[0], -1.0, 1.0
        yv = np.asarray(y, dtype=float)
        for c in X.columns:
            xv = np.asarray(X[c], dtype=float)
            if xv.std() == 0 or yv.std() == 0:
                continue
            r = np.corrcoef(xv, yv)[0, 1]
            if np.isfinite(r) and abs(r) > best_abs:
                best_c, best_abs, best_sign = c, abs(r), float(np.sign(r) or 1.0)
        self.col_, self.sign_ = best_c, best_sign
        return self

    def predict(self, X):
        return np.sign(self.sign_ * np.asarray(X[self.col_], dtype=float))


def _events(n: int):
    return pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")


def _span1_t1(idx):
    return pd.Series([idx[min(i + 1, len(idx) - 1)] for i in range(len(idx))], index=idx)


# --- splitters ---------------------------------------------------------------
def test_purged_kfold_covers_every_sample_once():
    idx = _events(60)
    t1 = _span1_t1(idx)
    cv = PurgedKFold(n_splits=5, embargo_pct=0.0)
    test_union = []
    for train_pos, test_pos in cv.split(t1):
        assert set(train_pos).isdisjoint(set(test_pos))
        test_union.extend(test_pos.tolist())
    assert sorted(test_union) == list(range(60))  # exact partition → valid OOF


def test_walk_forward_train_precedes_test():
    idx = _events(60)
    t1 = _span1_t1(idx)
    for train_pos, test_pos in WalkForwardSplit(n_splits=5).split(t1):
        assert train_pos.max() < test_pos.min()  # strictly forward in time


def test_holdout_is_the_tail_and_dev_is_purged():
    idx = _events(100)
    t1 = _span1_t1(idx)
    dev_pos, hold_pos = holdout_split(t1, holdout_pct=0.2)
    assert hold_pos.tolist() == list(range(80, 100))
    assert dev_pos.max() < hold_pos.min()       # boundary label purged → no bleed


# --- backtest & baselines ----------------------------------------------------
def test_strategy_returns_apply_cost_only_when_traded():
    idx = _events(3)
    ret = pd.Series([0.01, -0.02, -0.03], index=idx)  # underlying moves (signed)
    side = pd.Series([1.0, 0.0, -1.0], index=idx)
    cm = CostModel(spread_pips=1.0, slippage_pips=0.0, pip_size=0.001)  # cost 0.001 @ price 1
    r = strategy_returns(side, ret, cost_model=cm)
    assert r.iloc[0] == pytest.approx(0.01 - 0.001)   # long, up move, charged
    assert r.iloc[1] == pytest.approx(0.0)            # flat, no cost
    assert r.iloc[2] == pytest.approx(0.03 - 0.001)   # short profits on down move


def test_majority_class_side():
    labels = pd.Series([1, 1, 1, -1, 0])
    assert majority_class_side(labels) == 1


# --- oof predictions (the L5 bridge) -----------------------------------------
def test_oof_predict_is_complete():
    n = 200
    idx = _events(n)
    rng = np.random.default_rng(0)
    f = rng.normal(size=n)
    X = pd.DataFrame({"f": f, "g": rng.normal(size=n)}, index=idx)
    y = pd.Series(np.sign(f), index=idx)
    preds = oof_predict(CorrSignEstimator, X, y, _span1_t1(idx), cv=PurgedKFold(5, 0.0))
    assert preds.notna().all()           # every sample predicted out-of-fold
    assert set(np.unique(preds)).issubset({-1.0, 0.0, 1.0})


# --- the full Go/No-Go gate --------------------------------------------------
def _dataset(n, edge: bool, seed: int):
    idx = _events(n)
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    g = rng.normal(size=n)
    signal = 0.0015 * np.tanh(f) if edge else 0.0   # f carries the edge (or not)
    ret = pd.Series(signal + rng.normal(0, 0.0008, n), index=idx)
    labels = pd.Series(np.sign(ret.to_numpy()), index=idx)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.0008, n)), index=idx)  # ~flat B&H
    X = pd.DataFrame({"f": f, "g": g}, index=idx)
    return X, labels, ret, labels.copy(), close


def test_run_validation_passes_a_real_edge():
    X, y, ret, labels, close = _dataset(1600, edge=True, seed=1)
    cfg = ValidationConfig(
        cost_model=CostModel(spread_pips=0.3, slippage_pips=0.1), n_trials=1, holdout_pct=0.2
    )
    report = run_validation(
        CorrSignEstimator, X, y, t1=_span1_t1(X.index), ret=ret, labels=labels,
        close=close, config=cfg,
    )
    assert report.holdout_sharpe > report.best_baseline
    assert report.cpcv_sharpe_mean > 0
    assert report.passed, report.summary()


def test_run_validation_rejects_noise():
    X, y, ret, labels, close = _dataset(1600, edge=False, seed=2)
    cfg = ValidationConfig(
        cost_model=CostModel(spread_pips=0.3, slippage_pips=0.1), n_trials=1, holdout_pct=0.2
    )
    report = run_validation(
        CorrSignEstimator, X, y, t1=_span1_t1(X.index), ret=ret, labels=labels,
        close=close, config=cfg,
    )
    assert not report.passed          # no real edge must NOT clear the gate
    assert report.reasons
