"""L3 — primary SIDE ensemble: members, encoding, recall rule, regime routing."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from aurax.l3_primary import (
    PROBA_COLUMNS,
    LogisticSide,
    PrimaryConfig,
    PrimarySignalModel,
    SideModel,
    probas_to_side,
)
from aurax.l3_primary.base import align_proba
from aurax.l3_primary.members import MEMBER_REGISTRY
from aurax.validation import (
    CostModel,
    PurgedKFold,
    ValidationConfig,
    oof_predict,
    run_validation,
)


def _events(n):
    return pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")


# --- members -----------------------------------------------------------------
def test_logistic_member_learns_separable_signal():
    n = 400
    idx = _events(n)
    rng = np.random.default_rng(0)
    f = rng.normal(size=n)
    X = pd.DataFrame({"f": f, "noise": rng.normal(size=n)}, index=idx)
    y = pd.Series(np.where(f > 0, 1, -1), index=idx)  # f separates up/down

    proba = LogisticSide().fit(X, y).predict_proba(X)
    assert list(proba.columns) == list(PROBA_COLUMNS)
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-9)
    # predicted up-probability tracks the feature sign
    acc = ((proba["p_up"] > proba["p_down"]).to_numpy() == (f > 0)).mean()
    assert acc > 0.9


def test_align_proba_fills_absent_class():
    # member saw only {-1, +1}; output must still expose p_flat = 0.
    idx = _events(3)
    proba = align_proba(np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]]), [-1, 1], idx)
    assert list(proba.columns) == list(PROBA_COLUMNS)
    assert (proba["p_flat"] == 0.0).all()


def test_probas_to_side_recall_threshold():
    proba = pd.DataFrame(
        {"p_down": [0.1, 0.45, 0.5], "p_flat": [0.1, 0.1, 0.2], "p_up": [0.8, 0.45, 0.3]},
        index=_events(3),
    )
    high_recall = probas_to_side(proba, flat_threshold=0.05)
    cautious = probas_to_side(proba, flat_threshold=0.30)
    assert high_recall.tolist() == [1, 0, -1]   # only the dead-heat abstains
    assert cautious.tolist() == [1, 0, 0]        # higher threshold abstains more
    assert (cautious == 0).sum() >= (high_recall == 0).sum()


# --- ensemble: regime-conditional blending -----------------------------------
class _DummyUp(SideModel):
    name = "up"

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        df = pd.DataFrame(0.0, index=X.index, columns=list(PROBA_COLUMNS))
        df["p_up"] = 1.0
        return df


class _DummyDown(SideModel):
    name = "down"

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        df = pd.DataFrame(0.0, index=X.index, columns=list(PROBA_COLUMNS))
        df["p_down"] = 1.0
        return df


@pytest.fixture
def _register_dummies():
    MEMBER_REGISTRY["up"] = _DummyUp
    MEMBER_REGISTRY["down"] = _DummyDown
    yield
    del MEMBER_REGISTRY["up"], MEMBER_REGISTRY["down"]


def test_regime_overrides_route_the_blend(_register_dummies):
    cfg = PrimaryConfig(
        members={"up": 0.5, "down": 0.5},
        regime_overrides={"trend": {"up": 1.0, "down": 0.0}, "range": {"up": 0.0, "down": 1.0}},
        flat_threshold=0.1,
    )
    idx = _events(4)
    X = pd.DataFrame({"f": np.zeros(4)}, index=idx)
    y = pd.Series([1, -1, 1, -1], index=idx)
    model = PrimarySignalModel(cfg).fit(X, y)

    # No regime → 50/50 blend → zero margin → flat everywhere.
    assert model.predict_side(X).tolist() == [0, 0, 0, 0]
    # Regime overrides flip the blend toward one member.
    regimes = pd.Series(["trend", "range", "trend", "range"], index=idx)
    assert model.predict_side(X, regimes).tolist() == [1, -1, 1, -1]


# --- sklearn-compat: through the validation plumbing -------------------------
def _edge_dataset(n, seed):
    idx = _events(n)
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    ret = pd.Series(0.0015 * np.tanh(f) + rng.normal(0, 0.0008, n), index=idx)
    labels = pd.Series(np.sign(ret.to_numpy()).astype(int), index=idx)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.0008, n)), index=idx)
    X = pd.DataFrame({"f": f, "g": rng.normal(size=n)}, index=idx)
    return X, labels, ret, close


def test_oof_predict_with_primary_is_complete():
    X, labels, ret, close = _edge_dataset(300, seed=1)
    t1 = pd.Series([X.index[min(i + 1, len(X) - 1)] for i in range(len(X))], index=X.index)
    preds = oof_predict(
        lambda: PrimarySignalModel(PrimaryConfig.lightweight()),
        X, labels, t1, cv=PurgedKFold(5, 0.0),
    )
    assert preds.notna().all()
    assert set(np.unique(preds)).issubset({-1.0, 0.0, 1.0})


def test_primary_clears_the_gate_on_a_real_edge():
    X, labels, ret, close = _edge_dataset(1600, seed=3)
    t1 = pd.Series([X.index[min(i + 1, len(X) - 1)] for i in range(len(X))], index=X.index)
    cfg = ValidationConfig(cost_model=CostModel(spread_pips=0.3, slippage_pips=0.1), n_trials=1)
    report = run_validation(
        lambda: PrimarySignalModel(PrimaryConfig.lightweight()),
        X, labels, t1=t1, ret=ret, labels=labels, close=close, config=cfg,
    )
    assert report.holdout_sharpe > report.best_baseline
    assert report.passed, report.summary()


# --- real GBM member (only if the backend is installed) ----------------------
@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed")
def test_lightgbm_member_runs():
    from aurax.l3_primary import LightGBMSide

    X, labels, ret, close = _edge_dataset(500, seed=2)
    proba = LightGBMSide(n_estimators=50).fit(X, labels).predict_proba(X)
    assert list(proba.columns) == list(PROBA_COLUMNS)
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
