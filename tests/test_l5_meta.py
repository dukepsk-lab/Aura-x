"""L5 — calibration, base learner, meta-model, threshold tuning, meta-labeling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurax.l5_meta import (
    IsotonicCalibrator,
    LogisticMeta,
    MetaConfig,
    MetaGatedPrimary,
    MetaLabelModel,
    SigmoidCalibrator,
    build_meta_features,
    meta_labels_from_sides,
    train_meta_labeler,
)
from aurax.l5_meta.calibration import sigmoid
from aurax.l8_monitoring import calibration_error
from aurax.validation import CostModel, ValidationConfig, ValidationReport, run_validation


def _events(n):
    return pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")


def _span1(idx):
    return pd.Series([idx[min(i + 1, len(idx) - 1)] for i in range(len(idx))], index=idx)


# --- calibration -------------------------------------------------------------
def test_isotonic_reduces_calibration_error_and_is_monotonic():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 4000)
    p_true = s**2  # the score systematically overstates the true probability
    y = (rng.uniform(size=4000) < p_true).astype(int)
    cal = IsotonicCalibrator().fit(s, y)
    p = cal.predict(s)
    assert calibration_error(p, y) < calibration_error(s, y)
    grid = cal.predict(np.linspace(0, 1, 50))
    assert np.all(np.diff(grid) >= -1e-9)  # monotonic non-decreasing


def test_sigmoid_calibrator_monotonic_in_unit_interval():
    rng = np.random.default_rng(1)
    s = rng.normal(size=3000)
    y = (rng.uniform(size=3000) < sigmoid(2 * s)).astype(int)
    cal = SigmoidCalibrator().fit(s, y)
    p = cal.predict(np.sort(s))
    assert np.all((p >= 0) & (p <= 1))
    assert np.all(np.diff(p) >= -1e-9)


# --- base learner + meta-model ----------------------------------------------
def test_logistic_meta_learns_separable_binary():
    rng = np.random.default_rng(2)
    f = rng.normal(size=400)
    X = pd.DataFrame({"f": f, "noise": rng.normal(size=400)})
    y = (f > 0).astype(int)
    p = LogisticMeta().fit(X, y).predict_proba(X)
    assert ((p > 0.5).astype(int) == y).mean() > 0.9


def test_meta_model_is_calibrated_and_gates():
    rng = np.random.default_rng(3)
    n = 1500
    idx = _events(n)
    f = rng.normal(size=n)
    X = pd.DataFrame({"f": f, "g": rng.normal(size=n)}, index=idx)
    y = pd.Series((rng.uniform(size=n) < sigmoid(1.5 * f)).astype(int), index=idx)
    m = MetaLabelModel(MetaConfig(calibration="isotonic", cv_folds=4)).fit(X, y, t1=_span1(idx))

    p = m.predict_proba(X)
    assert p.between(0, 1).all()
    # a calibrated model's mean probability ≈ the empirical positive rate
    assert abs(p.mean() - y.mean()) < 0.05
    assert m.gate(X, 0.5).dtype == bool


def test_tune_threshold_improves_precision():
    rng = np.random.default_rng(4)
    probs = pd.Series(rng.uniform(size=600))
    labels = pd.Series((rng.uniform(size=600) < probs).astype(int))  # calibrated
    tau = MetaLabelModel().tune_threshold(probs, labels, min_trades=20)
    selected = probs >= tau
    assert labels[selected].mean() >= labels.mean()  # gating raises precision


# --- meta-labeling logic -----------------------------------------------------
def test_meta_labels_from_sides():
    side = pd.Series([1, -1, 0, 1])
    y = pd.Series([1, 1, 0, -1])
    assert meta_labels_from_sides(side, y).tolist() == [1, 0, 0, 0]


# A controllable primary: predicts sign(signal); correct only when reliability>0.
class _StubPrimary:
    def fit(self, X, y, sample_weight=None):
        return self

    def predict_side(self, X, regimes=None):
        return pd.Series(np.sign(X["signal"].to_numpy()).astype(int), index=X.index)

    def predict_proba(self, X, regimes=None):
        side = np.sign(X["signal"].to_numpy())
        conf = sigmoid(2 * np.abs(X["signal"].to_numpy()))
        p_up = np.where(side > 0, conf, 1 - conf)
        return pd.DataFrame(
            {"p_down": 1 - p_up, "p_flat": 0.0, "p_up": p_up}, index=X.index
        )


def _reliability_dataset(n, seed):
    idx = _events(n)
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    reliability = rng.normal(size=n)
    side_guess = np.sign(signal)
    # the primary's signal works only where reliability > 0 (good regime)
    y = pd.Series(np.where(reliability > 0, side_guess, -side_guess).astype(int), index=idx)
    X = pd.DataFrame({"signal": signal, "reliability": reliability}, index=idx)
    return X, y, idx


def test_train_meta_labeler_lifts_precision():
    X, y, idx = _reliability_dataset(1500, seed=5)
    res = train_meta_labeler(
        _StubPrimary, X, y, t1=_span1(idx), context=X[["reliability"]],
        meta_config=MetaConfig(cv_folds=4), tune=True,
    )
    # the trust layer filters the bad-regime bets → precision up, trades down
    assert res["precision_gated"] > res["precision_raw"] + 0.1
    assert res["trade_rate_gated"] <= res["trade_rate_raw"]


def test_meta_gated_primary_makes_a_noisy_primary_tradeable():
    X, y, idx = _reliability_dataset(1200, seed=6)
    ret = (0.001 * y).astype(float)                       # +pnl when the side is right
    close = pd.Series(1.10 + np.cumsum(np.random.default_rng(0).normal(0, 0.0008, len(idx))), index=idx)

    def make():
        return MetaGatedPrimary(
            _StubPrimary, t1=_span1(idx), context_cols=["reliability"],
            meta_config=MetaConfig(cv_folds=4),
        )

    report = run_validation(
        make, X, y, t1=_span1(idx), ret=ret, labels=y, close=close,
        config=ValidationConfig(cost_model=CostModel(spread_pips=0.2, slippage_pips=0.1), n_trials=1),
    )
    assert isinstance(report, ValidationReport)
    assert report.holdout_sharpe > 0          # gating the bad-regime half yields edge
    assert report.holdout_turnover < 1.0       # the meta layer actually abstains


def test_build_meta_features_columns():
    idx = _events(5)
    proba = pd.DataFrame(
        {"p_down": [0.2] * 5, "p_flat": [0.1] * 5, "p_up": [0.7] * 5}, index=idx
    )
    side = pd.Series([1, 1, -1, 0, 1], index=idx)
    feats = build_meta_features(proba, side, context=pd.DataFrame({"vol": np.arange(5.0)}, index=idx))
    assert {"prim_margin", "prim_conf", "prim_side", "ctx_vol"}.issubset(feats.columns)
