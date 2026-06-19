"""L2 — Gaussian HMM, the rule gate, state→regime mapping, shock override."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurax.enums import Regime
from aurax.l2_regime import GaussianHMM, RegimeConfig, RegimeDetector


def _events(n):
    return pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")


def _cluster_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    """Accuracy after mapping each predicted cluster to its majority true label."""
    df = pd.DataFrame({"p": pred, "t": true})
    mapping = df.groupby("p")["t"].agg(lambda s: s.value_counts().idxmax())
    return float((df["p"].map(mapping) == df["t"]).mean())


# --- the HMM -----------------------------------------------------------------
def test_gaussian_hmm_recovers_three_states():
    rng = np.random.default_rng(0)
    seg = 250
    x = np.concatenate([
        rng.normal(-3.0, 0.4, (seg, 2)),
        rng.normal(0.0, 0.4, (seg, 2)),
        rng.normal(3.0, 0.4, (seg, 2)),
    ])
    true = np.repeat([0, 1, 2], seg)
    hmm = GaussianHMM(n_states=3, seed=0).fit(x)

    assert _cluster_accuracy(hmm.decode(x), true) > 0.95
    np.testing.assert_allclose(hmm.transmat_.sum(axis=1), 1.0, atol=1e-9)


def test_hmm_posteriors_sum_to_one():
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(-2, 0.5, (100, 1)), rng.normal(2, 0.5, (100, 1))])
    proba = GaussianHMM(n_states=2, seed=0).fit(x).predict_proba(x)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


# --- rule gate ---------------------------------------------------------------
def test_rule_gate_priorities():
    det = RegimeDetector(RegimeConfig())
    assert det.gate(0.60, 0.40, 0.50) == Regime.TREND
    assert det.gate(0.60, 0.40, 0.95) == Regime.SHOCK   # shock dominates trend
    assert det.gate(0.50, 0.10, 0.50) == Regime.RANGE


def test_predict_rules_vectorised():
    idx = _events(3)
    feats = pd.DataFrame(
        {"tm_hurst": [0.6, 0.4, 0.6], "tm_ker": [0.4, 0.1, 0.4], "vol_atr_pct": [0.5, 0.5, 0.95]},
        index=idx,
    )
    assert RegimeDetector().predict_rules(feats).tolist() == ["trend", "range", "shock"]


# --- HMM detector: mapping + shock override ----------------------------------
def _regime_features(seed: int = 0) -> pd.DataFrame:
    """Three contiguous regimes separated in volatility and trend-memory."""
    rng = np.random.default_rng(seed)
    seg = 200

    def block(atr, hurst, ker, rvol):
        return pd.DataFrame({
            "vol_realized_vol": rng.normal(rvol, rvol * 0.1, seg),
            "tm_hurst": rng.normal(hurst, 0.03, seg),
            "tm_ker": rng.normal(ker, 0.03, seg),
            "vol_atr_pct": np.clip(rng.normal(atr, 0.03, seg), 0, 1),
        })

    frame = pd.concat(
        [
            block(0.30, 0.45, 0.15, 0.005),  # range: low vol, low memory
            block(0.55, 0.68, 0.45, 0.008),  # trend: mid vol, high memory
            block(0.95, 0.50, 0.30, 0.020),  # shock: high vol
        ],
        ignore_index=True,
    )
    frame.index = _events(len(frame))
    return frame


def test_detector_maps_states_and_overrides_shock():
    feats = _regime_features()
    det = RegimeDetector(RegimeConfig(seed=0)).fit(feats)

    # All three regimes were discovered and mapped.
    assert set(det._state_to_regime.values()) == {Regime.TREND, Regime.RANGE, Regime.SHOCK}

    out = det.predict_regimes(feats)
    assert set(out["regime"]).issubset({r.value for r in Regime})
    np.testing.assert_allclose(out[["p_trend", "p_range", "p_shock"]].sum(axis=1), 1.0, atol=1e-6)

    # The high-ATR-percentile block must be SHOCK (the hard stand-down override).
    shock_block = out.iloc[400:600]["regime"]
    assert (shock_block == "shock").mean() > 0.95
    # The mid blocks resolve to their HMM regimes (override doesn't fire there).
    assert (out.iloc[0:200]["regime"] == "range").mean() > 0.8
    assert (out.iloc[200:400]["regime"] == "trend").mean() > 0.8


def test_predict_returns_regime_state():
    feats = _regime_features()
    det = RegimeDetector(RegimeConfig(seed=0)).fit(feats)
    state = det.predict(feats)
    assert isinstance(state.regime, Regime)
    assert state.regime == Regime.SHOCK            # last bar is in the shock block
    assert abs(sum(state.probabilities.values()) - 1.0) < 1e-6


def test_shock_override_can_be_disabled():
    feats = _regime_features()
    det = RegimeDetector(RegimeConfig(seed=0, shock_override=False)).fit(feats)
    out = det.predict_regimes(feats)
    # Without the override the label comes purely from the HMM (still valid regimes).
    assert set(out["regime"]).issubset({r.value for r in Regime})


# --- integration: L2 regimes feed L3 -----------------------------------------
def test_regimes_feed_primary_ensemble():
    from aurax.l3_primary import PrimaryConfig, PrimarySignalModel

    feats = _regime_features()
    regimes = RegimeDetector(RegimeConfig(seed=0)).fit(feats).regime_series(feats)
    assert len(regimes) == len(feats)

    X = feats[["tm_hurst", "tm_ker", "vol_atr_pct"]]
    y = pd.Series(np.where(feats["tm_hurst"].to_numpy() > 0.5, 1, -1), index=feats.index)
    model = PrimarySignalModel(PrimaryConfig.lightweight()).fit(X, y)
    proba = model.predict_proba(X, regimes)  # L3 accepts L2's regime labels
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_fit_requires_known_features():
    bad = pd.DataFrame({"unrelated": np.arange(50.0)}, index=_events(50))
    with pytest.raises(ValueError):
        RegimeDetector().fit(bad)
