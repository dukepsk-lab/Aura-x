"""Lifecycle — training pipeline + persistence, and decay-triggered retraining."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurax.l3_primary import PrimaryConfig
from aurax.l5_meta import MetaConfig
from aurax.l8_monitoring import AlertKind, DecayConfig
from aurax.lifecycle import RetrainOrchestrator, RetrainPolicy, TrainingPipeline
from aurax.validation import CostModel, ValidationConfig


class RecordingNotifier:
    def __init__(self):
        self.sent = []

    def send(self, kind, message):
        self.sent.append((kind, message))
        return True


def _events(n):
    return pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")


def _span1(idx):
    return pd.Series([idx[min(i + 1, len(idx) - 1)] for i in range(len(idx))], index=idx)


def _dataset(n, *, edge: bool, seed: int):
    idx = _events(n)
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    if edge:
        flip = rng.uniform(size=n) < 0.1
        y = np.where(flip, -np.sign(f), np.sign(f)).astype(int)   # primary ~90% right
        ret = 0.0015 * y                                           # profit if side == y
    else:
        y = np.sign(rng.normal(size=n)).astype(int)               # unrelated to f
        ret = rng.normal(0, 0.001, n)
    X = pd.DataFrame({"f": f, "g": rng.normal(size=n)}, index=idx)
    return (
        X,
        pd.Series(y, index=idx),
        pd.Series(ret, index=idx),
        pd.Series(1.10 + np.cumsum(rng.normal(0, 0.0008, n)), index=idx),
    )


def _light_validation():
    return ValidationConfig(
        cpcv_n_groups=4, cpcv_n_test_groups=1, walk_forward_splits=3,
        cost_model=CostModel(spread_pips=0.2, slippage_pips=0.1), n_trials=1,
    )


# --- training pipeline + persistence -----------------------------------------
def test_train_produces_bundle_and_round_trips(tmp_path):
    X, y, ret, _ = _dataset(400, edge=True, seed=0)
    pipe = TrainingPipeline(PrimaryConfig.lightweight(), MetaConfig(cv_folds=3))
    model = pipe.train(X, y, t1=_span1(X.index), ret=ret)

    assert model.meta is not None
    assert model.metadata["n_samples"] == 400 and model.metadata["primary_members"] == ["logistic"]
    sides = model.predict_side(X)
    assert set(np.unique(sides)).issubset({-1, 0, 1})

    path = pipe.save(model, tmp_path / "bundle.pkl")
    reloaded = TrainingPipeline.load(path)
    pd.testing.assert_series_equal(reloaded.predict_side(X), sides)


# --- decay-triggered retraining ----------------------------------------------
def _orchestrator(notifier):
    pipe = TrainingPipeline(PrimaryConfig.lightweight(), MetaConfig(cv_folds=3))
    policy = RetrainPolicy(decay=DecayConfig(sharpe_window=20, min_rolling_sharpe=0.0))
    return RetrainOrchestrator(pipe, policy, notifier=notifier)


def test_no_decay_does_not_retrain():
    rec = RecordingNotifier()
    orch = _orchestrator(rec)
    X, y, ret, close = _dataset(300, edge=True, seed=1)
    rng = np.random.default_rng(0)
    result = orch.maybe_retrain(
        monitor_returns=pd.Series(rng.normal(0.002, 0.001, 40)),     # healthy edge
        monitor_prob=np.full(50, 0.6), monitor_outcome=(rng.uniform(size=50) < 0.6).astype(int),
        X=X, y=y, t1=_span1(X.index), ret=ret, labels=y, close=close,
    )
    assert result.triggered is False and result.model is None
    assert not rec.sent


def test_decay_retrains_and_gate_promotes_a_real_edge():
    rec = RecordingNotifier()
    orch = _orchestrator(rec)
    X, y, ret, close = _dataset(1200, edge=True, seed=2)
    result = orch.maybe_retrain(
        monitor_returns=pd.Series(np.random.default_rng(0).normal(-0.002, 0.001, 40)),  # decayed
        monitor_prob=np.full(50, 0.6), monitor_outcome=np.zeros(50),                     # miscalibrated too
        X=X, y=y, t1=_span1(X.index), ret=ret, labels=y, close=close,
        validation_config=_light_validation(),
    )
    assert result.triggered and result.decay.retrain_due
    assert result.promoted and result.model is not None      # cleared the gate
    assert any(k is AlertKind.DECAY for k, _ in rec.sent)


def test_decay_retrains_but_gate_rejects_noise():
    rec = RecordingNotifier()
    orch = _orchestrator(rec)
    X, y, ret, close = _dataset(1200, edge=False, seed=3)
    result = orch.maybe_retrain(
        monitor_returns=pd.Series(np.random.default_rng(0).normal(-0.002, 0.001, 40)),
        monitor_prob=np.full(50, 0.6), monitor_outcome=np.zeros(50),
        X=X, y=y, t1=_span1(X.index), ret=ret, labels=y, close=close,
        validation_config=_light_validation(),
    )
    assert result.triggered
    assert result.promoted is False and result.model is None  # no edge → rejected by the gate
    assert any("REJECTED" in msg for _, msg in rec.sent)
