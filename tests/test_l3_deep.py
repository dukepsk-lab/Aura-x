"""L3 deep members (Roadmap v2) — windowing + CNN/PatchTST/SSM (torch-gated)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from aurax.l3_primary import PROBA_COLUMNS, make_sequences

_NO_TORCH = importlib.util.find_spec("torch") is None
_deep = pytest.mark.skipif(_NO_TORCH, reason="torch (the `deep` extra) not installed")


def _events(n):
    return pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")


# --- windowing (dependency-free) ---------------------------------------------
def test_make_sequences_shape_and_point_in_time():
    X = pd.DataFrame(np.arange(20.0).reshape(10, 2), index=_events(10))
    seqs, idx = make_sequences(X, 3)
    assert seqs.shape == (8, 3, 2)
    assert list(idx) == list(X.index[2:])           # window k ends at index 2+k
    np.testing.assert_allclose(seqs[0], X.to_numpy()[0:3])   # uses bars [0,1,2]
    np.testing.assert_allclose(seqs[-1], X.to_numpy()[7:10])


def test_make_sequences_too_short_is_empty():
    X = pd.DataFrame(np.zeros((4, 2)), index=_events(4))
    seqs, idx = make_sequences(X, 10)
    assert seqs.shape[0] == 0 and len(idx) == 0


# --- deep members ------------------------------------------------------------
def _window_signal_dataset(n=600, lookback=10, seed=0):
    """A slow cyclical feature → label = sign of the window's mean. The cycle
    dominates the in-window noise, so it's a clean, learnable target for every
    pooling architecture (CNN / PatchTST / SSM)."""
    idx = _events(n)
    rng = np.random.default_rng(seed)
    base = np.sin(2 * np.pi * np.arange(n) / 120.0)   # slow cycle
    f = base + 0.3 * rng.normal(size=n)
    signal = pd.Series(f, index=idx).rolling(lookback).mean()
    y = pd.Series(np.sign(signal.fillna(0.0)).astype(int), index=idx)
    X = pd.DataFrame({"f": f, "g": rng.normal(size=n)}, index=idx)
    return X, y, signal


@_deep
@pytest.mark.parametrize("name", ["cnn", "patchtst", "ssm"])
def test_deep_member_trains_and_outputs_valid_proba(name):
    from aurax.l3_primary import build_member

    lookback = 10
    X, y, signal = _window_signal_dataset(lookback=lookback)
    member = build_member(name, lookback=lookback, hidden=16, epochs=20, seed=0).fit(X, y)
    proba = member.predict_proba(X)

    assert list(proba.columns) == list(PROBA_COLUMNS)
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-5)
    # warm-up rows (no full window) are neutral so they don't bias the blend
    np.testing.assert_allclose(proba.iloc[: lookback - 1].to_numpy(), 1.0 / 3.0, atol=1e-6)
    # learns the window-mean sign on the valid region
    valid = signal.notna().to_numpy()
    acc = ((proba["p_up"] > proba["p_down"]).to_numpy()[valid] == (signal.to_numpy()[valid] > 0)).mean()
    assert acc > 0.6


@_deep
def test_deep_member_is_deterministic():
    X, y, _ = _window_signal_dataset(n=400)
    from aurax.l3_primary import SSMSide

    p1 = SSMSide(lookback=10, hidden=16, epochs=6, seed=1).fit(X, y).predict_proba(X)
    p2 = SSMSide(lookback=10, hidden=16, epochs=6, seed=1).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1.to_numpy(), p2.to_numpy(), atol=1e-5)


@_deep
def test_deep_member_in_ensemble_and_oof():
    from aurax.l3_primary import PrimaryConfig, PrimarySignalModel
    from aurax.validation import PurgedKFold, oof_predict

    X, y, _ = _window_signal_dataset(n=500)
    cfg = PrimaryConfig(members={"ssm": 1.0}, member_params={"ssm": {"lookback": 10, "epochs": 6, "hidden": 16}})
    model = PrimarySignalModel(cfg).fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-5)
    assert set(np.unique(model.predict(X))).issubset({-1, 0, 1})

    # deep member as an sklearn-style primary through the purged OOF bridge
    t1 = pd.Series([X.index[min(i + 1, len(X) - 1)] for i in range(len(X))], index=X.index)
    cnn_cfg = PrimaryConfig(members={"cnn": 1.0}, member_params={"cnn": {"lookback": 10, "epochs": 4}})
    preds = oof_predict(lambda: PrimarySignalModel(cnn_cfg), X, y, t1, cv=PurgedKFold(3))
    assert preds.notna().all()
